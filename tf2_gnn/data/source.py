from typing import Union, Dict, Any, Optional, Callable
import abc
import tensorflow as tf
import tensorflow_datasets as tfds
from tf2_gnn.utils.register import register_custom_object

Split = Union[str, tfds.Split]


class DataSource(abc.ABC):

    @abc.abstractmethod
    def get_dataset(self, split: Split, repeats: Optional[int] = None) -> tf.data.Dataset:
        raise NotImplementedError('Abstract method')

    @abc.abstractmethod
    def examples_per_epoch(self, split: Split) -> int:
        raise NotImplementedError('Abstract method')

    @abc.abstractmethod
    def metadata(self) -> Dict[str, Any]:
        raise NotImplementedError('Abstract method')

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config) -> 'DataSource':
        raise NotImplementedError('Abstract class method')


@register_custom_object
class TfdsSource(DataSource):

    def __init__(self,
                 builder: Union[tfds.core.DatasetBuilder, str],
                 metadata: Dict[str, Any] = {},
                 as_supervised: bool = True,
                 data_dir: Optional[str] = None,
                 in_memory: Optional[bool] = None,
                 shuffle_buffer: int = 1024):
        self._builder: tfds.core.DatasetBuilder = (tfds.builder(builder, data_dir=data_dir)
                                                   if isinstance(builder, str) else builder)
        self._metadata = metadata
        self._as_supervised = as_supervised
        self._in_memory = in_memory
        self._shuffle_buffer = shuffle_buffer

    def get_config(self):
        return dict(builder=self._builder.name,
                    metadata=self.metadata(),
                    as_supervised=self._as_supervised,
                    data_dir=self._builder.data_dir,
                    in_memory=self._in_memory,
                    shuffle_buffer=self._shuffle_buffer)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def metadata(self):
        return self._metadata.copy()  # defensive copy

    def examples_per_epoch(self, split: Split) -> int:
        return self._builder.info.splits[split].num_examples

    def get_dataset(self, split: Split, repeats: Optional[int] = None) -> tf.data.Dataset:
        shuffle = split == 'train'
        dataset = self._builder.as_dataset(split=split,
                                           shuffle_files=shuffle,
                                           as_supervised=self._as_supervised,
                                           in_memory=self._in_memory)
        if repeats != -1:
            dataset = dataset.repeat()
        if shuffle:
            dataset = dataset.shuffle(self._shuffle_buffer)
        return dataset


@register_custom_object
class MappedSource(DataSource):

    def __init__(self,
                 base: DataSource,
                 map_fn: Callable,
                 num_parallel_calls: int = tf.data.experimental.AUTOTUNE):
        self._base = base
        self._map_fn = map_fn
        self._num_parallel_calls = num_parallel_calls

    def get_config(self):
        return dict(base=tf.keras.utils.serialize_keras_object(self._base),
                    num_parallel_calls=self._num_parallel_calls)

    @classmethod
    def from_config(cls, config):
        assert cls is not MappedSource
        return cls(**config)

    def metadata(self):
        return self._base.metadata()

    def examples_per_epoch(self, split: Split) -> int:
        return self._base.examples_per_epoch(split)

    def get_dataset(self, split: Split) -> tf.data.Dataset:
        with tf.keras.backend.learning_phase_scope(split == 'train'):
            return self._base.get_dataset(self).map(self._map_fn, self._num_parallel_calls)


@register_custom_object
class GraphSource(DataSource):

    def __init__(self,
                 base: DataSource,
                 batch_size: int,
                 num_parallel_calls: int = tf.data.experimental.AUTOTUNE,
                 prefetch_buffer: int = tf.data.experimental.AUTOTUNE):
        self._base = get(base)
        self._batch_size = batch_size
        self._num_parallel_calls = num_parallel_calls
        self._prefetch_buffer = prefetch_buffer

    def get_config(self):
        return dict(
            base=tf.keras.utils.serialize_keras_object(self._base),
            batch_size=self._batch_size,
            num_parallel_calls=self._num_parallel_calls,
            prefetch_buffer=self._prefetch_buffer,
        )

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def examples_per_epoch(self, split: Split) -> int:
        return self._base.examples_per_epoch(split) // self._batch_size

    def get_dataset(self, split: Split, repeats: Optional[int] = None) -> int:
        ds = self._base.get_dataset(split=split, repeats=repeats)
        ds = ds.apply(tf.data.experimental.dense_to_ragged_batch(self._batch_size))

        def post_batch_map(inputs: Dict[str, tf.RaggedTensor], labels: tf.RaggedTensor):
            node_features = inputs['node_features']
            offsets = node_features.row_starts()
            # block-diagonalize links
            links = inputs['links']
            b = links.value_rowids()
            links = links.values + tf.expand_dims(tf.gather(offsets, b), axis=-1)
            inputs['links'] = links
            labels = labels.values
            return inputs, labels

        ds = ds.map(post_batch_map, self._num_parallel_calls)
        ds = ds.prefetch(self._prefetch_buffer)
        return ds

    def metadata(self):
        return self._base.metadata()


@register_custom_object
class NodeBatchedGraphSource(DataSource):
    """Dataset that batches graphs up to a maximum number of nodes."""

    def __init__(self,
                 base: DataSource,
                 max_nodes_per_batch: int,
                 approx_nodes_per_example: Union[int, float],
                 num_parallel_calls: int = tf.data.experimental.AUTOTUNE,
                 prefetch_buffer: int = tf.data.experimental.AUTOTUNE):
        self._base = get(base)
        self._max_nodes_per_batch = max_nodes_per_batch
        self._approx_nodes_per_example = approx_nodes_per_example
        self._num_parallel_calls = num_parallel_calls
        self._prefetch_buffer = prefetch_buffer
        self._approx_batch_size = max_nodes_per_batch / approx_nodes_per_example

    def get_config(self):
        return dict(
            base=tf.keras.utils.serialize_keras_object(self._base),
            max_nodes_per_batch=self._max_nodes_per_batch,
            approx_batch_size=self._approx_batch_size,
            num_parallel_calls=self._num_parallel_calls,
            prefetch_buffer=self._prefetch_buffer,
        )

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def examples_per_epoch(self, split: Split) -> int:
        return int(self._base.examples_per_epoch(split) / self._approx_batch_size)

    def get_dataset(self, split: Split, repeats: Optional[int] = None) -> int:
        ds = self._base.get_dataset(split=split, repeats=repeats)

        inputs_spec, labels_spec = ds.element_spec
        node_features_spec = inputs_spec['node_features']
        links_spec = inputs_spec['links']
        node_features_spec.shape.assert_has_rank(2)
        links_spec.shape.assert_has_rank(2)
        assert links_spec.shape[1] == 2

        @tf.function
        def scan_fn(old_state, input_element):
            node_features, links, labels, node_row_splits = old_state
            next_inputs, next_labels = input_element
            next_node_features = next_inputs['node_features']
            next_links = next_inputs['links']

            offset = node_row_splits[-1]
            nn = tf.shape(next_node_features, out_type=node_row_splits.dtype)[0]
            num_nodes = offset + nn
            if num_nodes > self._max_nodes_per_batch:
                output_element = (node_features, links, labels, node_row_splits, True)
                node_row_splits = tf.stack([0, nn], axis=0)
                new_state = (next_node_features, next_links, next_labels, node_row_splits)
                return new_state, output_element
            else:
                node_row_splits = tf.concat(
                    [node_row_splits, tf.expand_dims(num_nodes, axis=0)], axis=0)
                node_features = tf.concat([node_features, next_node_features], axis=0)
                links = tf.concat([links, next_links + offset], axis=0)
                labels = tf.concat([labels, next_labels], axis=0)
                new_state = (node_features, links, labels, node_row_splits)
                output_element = (node_features, links, labels, node_row_splits, False)
                return new_state, output_element

        def filter_fn(*args):
            return args[-1]

        def map_fn(node_features, links, labels, node_row_splits, done):
            # return node_features, links, labels, node_row_splits
            node_features = tf.RaggedTensor.from_row_splits(node_features, node_row_splits)
            return dict(node_features=node_features, links=links), labels

        node_features = tf.zeros((0, node_features_spec.shape[1]), dtype=node_features_spec.dtype)
        links = tf.zeros((0, 2), dtype=links_spec.dtype)
        labels = tf.zeros((0, *labels_spec.shape[1:]), dtype=labels_spec.dtype)
        node_row_splits = tf.zeros((1,), dtype=links.dtype)
        init_state = (node_features, links, labels, node_row_splits)

        ds = ds.apply(tf.data.experimental.scan(init_state, scan_fn)).filter(filter_fn).map(map_fn)
        ds = ds.prefetch(self._prefetch_buffer)
        return ds

    def metadata(self):
        return self._base.metadata()


def get(identifier) -> DataSource:
    if isinstance(identifier, DataSource):
        return identifier
    else:
        if isinstance(identifier, str):
            identifier = dict(class_name=identifier, config={})
        return tf.keras.utils.deserialize_keras_object(identifier)

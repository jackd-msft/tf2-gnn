from typing import Union, Dict, Any, Optional, Mapping, NamedTuple, Callable
import abc
from copy import deepcopy
import tensorflow as tf
import tensorflow_datasets as tfds
from ..utils import register as register_lib
from ..utils import functions

Split = Union[str, tfds.Split]


class DataSource(abc.ABC):

    @property
    def example_spec(self):
        return self.get_dataset('train').element_spec

    @abc.abstractmethod
    def get_dataset(self, split: Split, repeats: Optional[int] = None):
        raise NotImplementedError('Abstract method')

    @abc.abstractmethod
    def examples_per_epoch(self, split: Split):
        raise NotImplementedError('Abstract method')

    @property
    def meta(self) -> Dict[str, Any]:
        return {}

    def map(self, map_fn, num_parallel_calls: int = 1) -> 'DataSource':
        return MappedSource(self, map_fn, num_parallel_calls)

    def with_meta(self, meta: Mapping[str, Dict]) -> 'DataSource':
        return SourceWithMeta(self, meta)


@register_lib.register_custom_object
class TfdsSource(DataSource):

    def __init__(self,
                 builder: Union[tfds.core.DatasetBuilder, str],
                 meta: Mapping[str, Any] = {},
                 as_supervised: bool = True,
                 data_dir: Optional[str] = None,
                 in_memory: Optional[bool] = None,
                 shuffle_buffer: int = -1):
        if isinstance(builder, str):
            self._builder: tfds.core.DatasetBuilder = tfds.builder(builder, data_dir=data_dir)
            self._builder.download_and_prepare()
        else:
            self._builder: tfds.core.DatasetBuilder = builder
        self._meta = meta
        self._as_supervised = as_supervised
        self._in_memory = in_memory
        self._shuffle_buffer = (self.examples_per_epoch('train')
                                if shuffle_buffer == -1 else shuffle_buffer)

    def get_config(self):
        return dict(builder=self._builder.name,
                    metad=self.meta,
                    as_supervised=self._as_supervised,
                    data_dir=self._builder.data_dir,
                    in_memory=self._in_memory,
                    shuffle_buffer=self._shuffle_buffer)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @property
    def meta(self) -> Mapping[str, Any]:
        return deepcopy(self._meta)  # defensive copy

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


class DelegatingSource(DataSource):

    def __init__(self, base: Union[DataSource, Mapping[str, Any]]):
        self._base = get(base)

    def get_config(self):
        return dict(base=tf.keras.utils.serialize_keras_object(self._base))

    @classmethod
    def from_config(cls, config) -> 'DelegatingSource':
        return cls(**config)

    def get_dataset(self, split: Split, repeats: Optional[int] = None) -> tf.data.Dataset:
        return self._base.get_dataset(split, repeats)

    def examples_per_epoch(self, split: Split) -> int:
        return self._base.examples_per_epoch(split)


@register_lib.register_custom_object
class MappedSource(DelegatingSource):

    def __init__(self,
                 base: Union[DataSource, Mapping[str, Any]],
                 map_fn: Callable,
                 num_parallel_calls: int = tf.data.experimental.AUTOTUNE):
        super().__init__(base=base)
        self._map_fn = functions.unwrap(map_fn)
        self._num_parallel_calls = num_parallel_calls

    def get_dataset(self, split: Split, repeats: Optional[int] = None):
        return self._base.get_dataset(split, repeats).map(self._map_fn, self._num_parallel_calls)

    def get_config(self) -> Mapping[str, Any]:
        config = super().get_config()
        config.update(
            dict(map_fn=tf.keras.utils.serialize_keras_object(functions.wrap(self._map_fn)),
                 num_parallel_calls=self._num_parallel_calls))
        return config


@register_lib.register_custom_object
class SourceWithMeta(DelegatingSource):

    def __init__(self, base: Union[DataSource, Mapping[str, Any]],
                 meta: Optional[Mapping[str, Any]]):
        super().__init__(base)
        self._meta = meta

    @property
    def meta(self):
        return deepcopy(self._meta)

    def get_config(self):
        config = super().get_config()
        config['meta'] = self.meta
        return config

    def with_meta(self, meta: Mapping[str, Any]) -> DataSource:
        return self._base.with_meta(meta)


def get(identifier) -> DataSource:
    return register_lib.get(identifier, DataSource)

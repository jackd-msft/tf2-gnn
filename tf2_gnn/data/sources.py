from typing import Optional, Any, Mapping
import functools
import tensorflow as tf
from ..utils import register as register_lib
from .core import Split, DataSource, get, TfdsSource
from . import batching

AUTOTUNE = tf.data.experimental.AUTOTUNE


@register_lib.register_custom_object
class GraphSource(DataSource):

    def __init__(self,
                 base: DataSource,
                 batch_size: Optional[int] = None,
                 max_nodes_per_batch: Optional[int] = None,
                 num_parallel_calls: int = tf.data.experimental.AUTOTUNE,
                 prefetch_buffer: int = tf.data.experimental.AUTOTUNE):
        self._base = get(base)
        self._num_parallel_calls = num_parallel_calls
        self._prefetch_buffer = prefetch_buffer
        self._batch_size = batch_size
        self._max_nodes_per_batch = max_nodes_per_batch
        self._nodes_per_epoch = {}
        self._examples_per_epoch = {}

    def get_config(self):
        return dict(
            base=tf.keras.utils.serialize_keras_object(self._base),
            num_parallel_calls=self._num_parallel_calls,
            prefetch_buffer=self._prefetch_buffer,
            batch_size=self._batch_size,
            max_nodes_per_batch=self._max_nodes_per_batch,
        )

    @property
    def meta(self) -> Mapping[str, Any]:
        return self._base.meta

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_dataset(self, split: Split, repeats: Optional[int] = None):
        dataset = self._base.get_dataset(split=split, repeats=repeats)
        examples_per_epoch = self._base.examples_per_epoch(split)
        if self._batch_size is not None:
            batch_size = min(self._batch_size, examples_per_epoch)
            self._examples_per_epoch[split] = examples_per_epoch // batch_size
            return batching.batch_with_batch_size(dataset, batch_size, self._num_parallel_calls)
        else:
            assert isinstance(self._max_nodes_per_batch, int)
            if split not in self._nodes_per_epoch:
                self._nodes_per_epoch[split] = batching.count_nodes_per_epoch(
                    dataset.take(examples_per_epoch))
            nodes_per_epoch = self._nodes_per_epoch[split]
            max_nodes_per_batch = min(self._max_nodes_per_batch, nodes_per_epoch)
            self._examples_per_epoch[split] = nodes_per_epoch // max_nodes_per_batch
            return batching.batch_with_max_nodes(dataset, self._max_nodes_per_batch,
                                                 self._num_parallel_calls)

    def examples_per_epoch(self, split: Split) -> int:
        if split not in self._examples_per_epoch:
            self.get_dataset(split, repeats=-1)
        return self._examples_per_epoch[split]


@register_lib.register_custom_object
def ppi(batch_size: Optional[int] = None,
        max_nodes_per_batch: Optional[int] = None,
        num_parallel_calls: int = AUTOTUNE,
        prefetch_buffer: int = AUTOTUNE,
        download_and_prepare: bool = True,
        **builder_kwargs) -> GraphSource:
    from .builders.ppi import PPI, NUM_CLASSES
    builder = PPI(**builder_kwargs)
    if download_and_prepare:
        builder.download_and_prepare()
    base = TfdsSource(builder, meta=dict(num_classes=NUM_CLASSES))
    return GraphSource(
        base,
        batch_size=batch_size,
        max_nodes_per_batch=max_nodes_per_batch,
        num_parallel_calls=num_parallel_calls,
        prefetch_buffer=prefetch_buffer,
    )


def _gather_label(inputs, labels, target_index):
    return inputs, tf.gather(labels, target_index, axis=-1)


@register_lib.register_custom_object
def qm9(batch_size: Optional[int] = None,
        max_nodes_per_batch: Optional[int] = None,
        num_parallel_calls: int = AUTOTUNE,
        prefetch_buffer: int = AUTOTUNE,
        download_and_prepare: bool = True,
        target_index: Optional[int] = None,
        **builder_kwargs) -> GraphSource:
    from .builders.qm9 import QM9, NUM_TARGETS
    builder = QM9(**builder_kwargs)
    if download_and_prepare:
        builder.download_and_prepare()
    if target_index is not None:
        base = TfdsSource(builder).map(functools.partial(_gather_label, target_index=target_index))
    else:
        base = TfdsSource(builder, meta=dict(num_targets=NUM_TARGETS))
    return GraphSource(
        base,
        batch_size=batch_size,
        max_nodes_per_batch=max_nodes_per_batch,
        num_parallel_calls=num_parallel_calls,
        prefetch_buffer=prefetch_buffer,
    )


_sources = {
    'ppi': ppi,
    'qm9': qm9,
}


def get_source(name: str, **kwargs) -> GraphSource:
    return _sources[name](**kwargs)

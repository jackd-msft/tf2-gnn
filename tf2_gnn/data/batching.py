import abc
import collections
from typing import (
    Any,
    Generic,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import tensorflow as tf


def append_empty_element(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """
    Append an empty element to the input dataset.

    This is useful for `scan`ing datasets where the element you want to return
    may be dependent on the previous state and the current element, e.g.
    block_diagonalize_batch.

    The appended element is all zeros. Unknown dimensions (e.g. dimensions
    that will be ragged after batching) have zero entries.

    Args:
        dataset: tf.data.Dataset

    Returns:
        dataset with one extra entry filled with zeros.
    """

    def empty_element(spec):
        return tf.zeros([1, *(0 if s is None else s for s in spec.shape)],
                        spec.dtype)

    empty_elements = tf.nest.map_structure(empty_element, dataset.element_spec)
    dataset = dataset.concatenate(
        tf.data.Dataset.from_tensor_slices(empty_elements))
    return dataset


S = TypeVar('S')
E = TypeVar('E')


class Accumulator(Generic[S, E]):

    @abc.abstractmethod
    def init_state(self) -> S:
        raise NotImplementedError('Abstract method')

    def can_accumulate(self, state: S, element: E):
        return ()

    @abc.abstractmethod
    def accumulate(self, state: S, element: E) -> S:
        raise NotImplementedError('Abstract method')

    def map_state(self, state: S):
        return state

    def batch(self, dataset: tf.data.Dataset):

        def scan_fn(prev_state, element):
            can_accumulate = tf.nest.flatten(
                self.can_accumulate(prev_state, element))
            assert len(can_accumulate) > 0  # prevent infinite accumulation
            can_accumulate = tf.reduce_all(can_accumulate)
            reset = tf.logical_not(can_accumulate)
            if reset:
                state = self.init_state()
            else:
                state = prev_state
            next_state = self.accumulate(state, element)
            return (next_state, (prev_state, reset))

        def filter_fn(state, reset):
            return reset

        def map_fn(state, reset):
            return self.map_state(state)

        return dataset.apply(
            tf.data.experimental.scan(self.init_state(),
                                      scan_fn)).filter(filter_fn).map(map_fn)


def _size(x: tf.Tensor):
    return tf.shape(x, tf.int32)[0]


class FlatValuesAccumulator(Accumulator[tf.Tensor, tf.Tensor]):

    def __init__(self,
                 spec: tf.TensorSpec,
                 max_size: Optional[int],
                 allow_empty=True):
        assert spec.shape.ndims > 0 and spec.shape[0] is None
        self._spec = spec
        self._max_size = max_size
        self._allow_empty = allow_empty

    def init_state(self):
        values = tf.zeros((0, *self._spec.shape[1:]), dtype=self._spec.dtype)
        return values

    def can_accumulate(self, state: tf.Tensor, element: tf.Tensor):
        size = _size(element)
        if self._max_size is None:
            conds = ()
        else:
            conds = (_size(state) + size <= self._max_size),
        if not self._allow_empty:
            conds = (*conds, size > 0)
        return conds

    def accumulate(self, state: tf.Tensor, element: tf.Tensor):
        return tf.concat((state, element), axis=0)


class TensorAccumulator(Accumulator[tf.Tensor, tf.Tensor]):

    def __init__(self, spec: tf.TensorSpec):
        self._spec = spec

    def init_state(self):
        values = tf.zeros(dtype=self._spec.dtype, shape=(0, *self._spec.shape))
        return values

    def accumulate(self, state, element):
        return tf.concat((state, tf.expand_dims(element, 0)), axis=0)


class RaggedState(NamedTuple):
    flat_values: tf.Tensor
    row_splits: tf.Tensor


class RaggedAccumulator(Accumulator[RaggedState, tf.Tensor]):

    def __init__(self,
                 spec: tf.TensorSpec,
                 max_size: Optional[int],
                 allow_empty: bool = True):
        self._values_acc = FlatValuesAccumulator(spec,
                                                 max_size=max_size,
                                                 allow_empty=allow_empty)
        self._row_splits_acc = TensorAccumulator(
            tf.TensorSpec(shape=(), dtype=tf.int32))

    def init_state(self):
        vals = self._values_acc.init_state()
        rs = self._row_splits_acc.accumulate(self._row_splits_acc.init_state(),
                                             _size(vals))
        return RaggedState(vals, rs)

    def can_accumulate(self, state, element):
        return self._values_acc.can_accumulate(state.flat_values, element)

    def accumulate(self, state: RaggedState, element: tf.Tensor):
        vals = self._values_acc.accumulate(state.flat_values, element)
        rs = self._row_splits_acc.accumulate(state.row_splits, _size(vals))
        # tf.print(rs)
        return RaggedState(vals, rs)

    def map_state(self, state: RaggedState):
        return tf.RaggedTensor.from_row_splits(
            *state, validate=True)  # TODO: turn off validation?


class MappingAccumulator(Accumulator[Mapping, Mapping],
                         collections.abc.Mapping):

    def __init__(self, accumulators: Mapping[Any, Accumulator]):
        self._accumulators = {
            k: packed_accumulator(v) for k, v in accumulators.items()
        }

    def init_state(self):
        return {k: v.init_state() for k, v in self._accumulators.items()}

    def can_accumulate(self, state, element):
        return {
            k: v.can_accumulate(state[k], element[k])
            for k, v in self._accumulators.items()
        }

    def accumulate(self, state, element):
        return {
            k: v.accumulate(state[k], element[k])
            for k, v in self._accumulators.items()
        }

    def map_state(self, state):
        return {k: v.map_state(state[k]) for k, v in self._accumulators.items()}

    def __getitem__(self, key):
        return self._accumulators[key]

    def __iter__(self):
        return iter(self._accumulators)

    def __len__(self):
        return len(self._accumulators)

    def __contains__(self, key):
        return key in self._accumulators


class TupleAccumulator(Accumulator[Tuple, Tuple], collections.abc.Sequence):

    def __init__(self, accumulators: Tuple):
        self._accumulators = tuple(packed_accumulator(a) for a in accumulators)

    def init_state(self):
        return tuple(acc.init_state() for acc in self._accumulators)

    def can_accumulate(self, state, element):
        return tuple(
            acc.can_accumulate(s, el)
            for acc, s, el in zip(self._accumulators, state, element))

    def accumulate(self, state, element):
        return tuple(
            acc.accumulate(s, el)
            for acc, s, el in zip(self._accumulators, state, element))

    def map_state(self, state):
        return tuple(
            acc.map_state(s) for acc, s in zip(self._accumulators, state))

    def __getitem__(self, i):
        return self._accumulators[i]

    def __len__(self):
        return len(self._accumulators)


def packed_accumulator(acc) -> Accumulator:
    if isinstance(acc, Accumulator):
        return acc
    if isinstance(acc, Mapping):
        return MappingAccumulator(
            {k: packed_accumulator(v) for k, v in acc.items()})
    if isinstance(acc, Tuple):
        return TupleAccumulator(tuple(packed_accumulator(a) for a in acc))
    raise ValueError(f'acc must be an accumulator, mapping or tuple, got {acc}')


class GraphState(NamedTuple):
    node_state: RaggedState
    links_state: Tuple[tf.Tensor, ...]
    labels_state: Union[tf.Tensor, tf.TensorArray]


class GraphAccumulator(Accumulator):

    def __init__(self,
                 element_spec,
                 max_nodes: int,
                 max_links: Union[None, int, Sequence[int]] = None,
                 allow_empty: bool = False):
        link_accs: Tuple[Accumulator, ...]
        labels_acc: Accumulator

        node_spec, link_specs, labels_spec = self._unpack(element_spec)
        for spec in link_specs:
            assert tuple(spec.shape) == (None, 2)
            assert spec.dtype.is_integer
        link_specs = tuple(
            tf.TensorSpec((None, 2), tf.int32) for _ in link_specs)

        node_acc = RaggedAccumulator(node_spec,
                                     max_nodes,
                                     allow_empty=allow_empty)

        if max_links is None or isinstance(max_links, int):
            max_links_seq = max_links,
        else:
            max_links_seq = max_links
        link_accs = tuple(
            FlatValuesAccumulator(spec, ml)
            for spec, ml in zip(link_specs, max_links_seq))
        if labels_spec.shape.ndims > 0 and labels_spec.shape[0] is None:
            labels_acc = FlatValuesAccumulator(labels_spec, max_nodes)
        else:
            labels_acc = TensorAccumulator(labels_spec)
        self._accs = TupleAccumulator((node_acc, link_accs, labels_acc))

    def _unpack(self, element):
        if len(element) == 3:
            return element
        inputs, labels = element
        if isinstance(inputs, Mapping):
            assert len(inputs) == 2
            nodes = inputs['node_features']
            links = inputs['links']
        else:
            nodes, links = inputs
        if not isinstance(links, tuple):
            links = links,
        return nodes, links, labels

    def _repack(self, nodes, links, labels):
        return (nodes, links), labels

    def init_state(self):
        return GraphState(*self._accs.init_state())

    def can_accumulate(self, state, element):
        return self._accs.can_accumulate(state, self._unpack(element))

    def accumulate(self, state, element):
        nodes, links, labels = self._unpack(element)
        offset = _size(state.node_state.flat_values)
        links = tuple(tf.cast(link, tf.int32) + offset for link in links)
        return GraphState(*self._accs.accumulate(state, (nodes, links, labels)))

    def map_state(self, state):
        return self._repack(*self._accs.map_state(state))


def block_diagonal_batch_with_max_nodes(
        dataset: tf.data.Dataset,
        max_nodes: int,
        max_links: Union[None, int, Sequence[int]] = None):
    """
    Batch the input dataset block diagonally up to the speicified max nodes.

    All examples are assumed to have at least 1 node.

    Args:
        dataset: tf.data.Dataset with spec ((nodes, (link*)), labels).
            nodes: [V?, ...] node features.
            link: [E?, 2] int edge/link indices.
            labels: [V?, ...] or [...] label data.
        max_nodes: maximum number of nodes allowed in each batch.

    Returns:
        dataset with spec:
            nodes: [B, V?, ...] ragged node features.
            links: [E, 2] indices into flattened nodes.
            labels: [BV, ...] or [B, ...]
        BV <= max_nodes is the total number not nodes.
    """
    accumulator = GraphAccumulator(dataset.element_spec,
                                   max_nodes,
                                   max_links,
                                   allow_empty=False)
    dataset = append_empty_element(dataset)
    return accumulator.batch(dataset)


def _block_diagonalize_batched(inputs, labels):
    if isinstance(inputs, dict):
        inputs = (inputs['node_features'], inputs['links'])
    node_features, links = inputs
    if isinstance(links, tf.RaggedTensor):
        links = links,
    offset = tf.expand_dims(node_features.row_splits, axis=-1)
    links = tuple(
        tf.cast(link.values, tf.int32) + tf.gather(offset, link.value_rowids())
        for link in links)
    return (node_features, links), labels.values


def block_diagonal_batch_with_batch_size(dataset: tf.data.Dataset,
                                         batch_size: int):
    """
    Batch the input dataset block diagonally up to the given batch size.

    Args:
        dataset: tf.data.Dataset with spec ((nodes, (link*)), labels).
            nodes: [V?, ...] node features.
            link: [E?, 2] int edge/link indices.
            labels: [V?, ...] or [...] label data.
        batch_size: number of examples in the resulting batch.

    Returns:
        dataset with spec:
            nodes: [B, V?, ...] ragged node features.
            links: [E, 2] indices into flattened nodes.
            labels: [BV, ...] or [B, ...]
        B = batch_size
        BV = sum_b V_b
    """
    dataset = dataset.map(
        lambda inputs, labels: dict(inputs=inputs, labels=labels))

    dataset = dataset.apply(
        tf.data.experimental.dense_to_ragged_batch(batch_size,
                                                   row_splits_dtype=tf.int32))
    return dataset.map(lambda kwargs: _block_diagonalize_batched(**kwargs))


def block_diagonal_batch(dataset: tf.data.Dataset,
                         max_nodes: Optional[int] = None,
                         batch_size: Optional[int] = None,
                         **kwargs) -> tf.data.Dataset:
    """
    Batch the input dataset block diagonally to batch_size or mx_nodes.

    Only

    Args:
        dataset: tf.data.Dataset with spec ((nodes, (link*)), labels).
            nodes: [V?, ...] node features.
            link: [E?, 2] int edge/link indices.
            labels: [V?, ...] or [...] label data.
        max_nodes: maximum number of nodes allowed in each batch.
        batch_size: batch size of resulting dataset.
        **kwargs: passed to one of
            block_diagonal_batch_with_batch_size
            block_diagonal_batch_with_max_nodes

    Returns:
        dataset with spec:
            nodes: [B, V?, ...] ragged node features.
            links: [E, 2] indices into flattened nodes.
            labels: [BV, ...] or [B, ...]
        B = batch_size (if batch_size is given, else dynamic)
        BV = sum_b V_b <= max_nodes (if max_nodes is given)

    Raises:
        ValueError if both ma_nodes and batch_size are given.

    See also:
        block_diagonal_batch_with_batch_size
        block_diagonal_batch_with_max_nodes
    """
    if max_nodes is None and batch_size is not None:
        return block_diagonal_batch_with_batch_size(dataset,
                                                    batch_size=batch_size,
                                                    **kwargs)
    elif max_nodes is not None and batch_size is None:
        return block_diagonal_batch_with_max_nodes(dataset,
                                                   max_nodes=max_nodes,
                                                   **kwargs)
    else:
        raise ValueError(
            'Exactly one of `max_nodes` and `batch_size` must be given')

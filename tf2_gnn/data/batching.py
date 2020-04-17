from typing import NamedTuple, Optional, Callable, Union, Tuple
import collections
import tensorflow as tf


def batch_with_max_nodes(dataset: tf.data.Dataset,
                         max_nodes_per_batch: int,
                         post_batch_map_func: Optional[Callable] = None,
                         num_parallel_calls=tf.data.experimental.AUTOTUNE,
                         row_splits_dtype=tf.int64,
                         dataset_size: Optional[int] = None,
                         skip_excessive_examples: bool = True
                        ) -> tf.data.Dataset:
    """
    Batch the given dataset until the first element has up to max_nodes_per_batch values.

    Current limitations:
        * dataset must contain only tensors - i.e. no SparseTensors or RaggedTensors.
        * if dataset_size is not given, elements of the partial final batch will be discarded
        * if dataset_size is given, the final example will be discarded if it would push the
            current batch over the max_nodes_per_batch limit.

    Args:
        dataset: tf.data.Dataset. The first entry of the flattened examples must have a dynamic
            leading dimension (otherwise use one of Dataset.batch, Dataset.padded_batch,
            data.experimental.dense_to_ragged_batch).
        max_nodes_per_batch: maximum leading dimension of the batched leading elements values.
        dataset_size: total number of examples in the dataset. If not given, all examples in the
            final incomplete batch are disgarded.
        post_batch_map_func: if given, this is applied after batching.
        num_parallel_calls: used in the post batch `Dataset.map` call.
        row_splits_dtype: dtype of the ragged row_splits.
        skip_excessive_examples: if True, examples with more nodes than max_nodes_per_batch are
            skipped completely. If False, they will make up their own batch.

    Returns:
        batched dataset, where any elements of the input dataset with dynamic leading dim are
            ragged tensors, or the rest of `post_batch_map_func` on those elements if one is
            provided.
    """

    class RaggedComponents(NamedTuple):
        values: tf.Tensor
        row_splits: tf.Tensor

    def append_tensor(acc: tf.Tensor, element: tf.Tensor):
        return tf.concat((acc, tf.expand_dims(element, axis=0)), axis=0)

    def append_ragged(acc: 'RaggedComponents',
                      element: tf.Tensor) -> 'RaggedComponents':
        values = acc.values
        row_splits = acc.row_splits
        values = tf.concat((values, element), axis=0)
        total_size = tf.shape(values, out_type=row_splits.dtype)[0]
        row_splits = append_tensor(row_splits, total_size)
        return RaggedComponents(values, row_splits)

    def append(acc: Union[tf.Tensor, 'RaggedComponents'], element: tf.Tensor):
        if isinstance(acc, tf.Tensor):
            return append_tensor(acc, element)
        else:
            return append_ragged(acc, element)

    def init_accumulator(spec: tf.TensorSpec):
        if spec.shape[0] is None:
            return RaggedComponents(
                tf.zeros((0, *spec._shape[1:]), dtype=spec._dtype),
                tf.zeros((1,), row_splits_dtype))
        else:
            return tf.zeros((0, *spec.shape), dtype=spec.dtype)

    def accumulator_from_element(element: tf.Tensor):
        if element.shape[0] is None:
            row_splits = tf.stack((
                tf.zeros((), dtype=row_splits_dtype),
                tf.shape(element, row_splits_dtype)[0],
            ))
            return RaggedComponents(element, row_splits)
        else:
            return tf.expand_dims(element, axis=0)

    def _apply_func(func: Optional[Callable], elements):
        if func is None:
            return elements
        if isinstance(elements, (tf.Tensor, tf.RaggedTensor)):
            return func(elements)
        elif isinstance(elements, collections.Mapping):
            return func(**elements)
        elif isinstance(elements, collections.Sequence):
            return func(*elements)
        else:
            raise ValueError(f'Unrecognized elements type {elements}')

    ########################
    # Implementation start #
    ########################
    specs = dataset.element_spec
    flat_specs = tf.nest.flatten(specs)
    if flat_specs[0].shape[0] is not None:
        raise ValueError(
            'leading value must have dynamic leading dimension but has shape '
            f'{flat_specs[0].shape}')

    def initial_state():
        return (tuple(init_accumulator(s) for s in flat_specs), 0)

    def leading_dim(x):
        return tf.shape(x, out_type=row_splits_dtype)[0]

    @tf.function
    def scan_fn(old_state: Tuple[Tuple, tf.Tensor], input_element):
        accs, count = old_state
        elements = tf.nest.flatten(input_element)

        count = count + 1

        size = leading_dim(elements[0])

        if skip_excessive_examples and size > max_nodes_per_batch:
            # example is too big - skip it
            return (accs, count), (accs, False)

        total_size = accs[0].row_splits[-1] + size
        if total_size > max_nodes_per_batch:
            output_element = (accs, True)
            new_accs = tuple(accumulator_from_element(el) for el in elements)
            return (new_accs, count), output_element
        else:
            accs = tuple(append(*args) for args in zip(accs, elements))
            final_batch = (dataset_size is not None and count == dataset_size)
            state = initial_state() if final_batch else (accs, count)
            return state, (accs, final_batch)

    def filter_fn(state, ret):
        del state
        return ret

    def map_func(elements, ret):
        del ret
        elements = tuple(
            tf.RaggedTensor.from_row_splits(
                *el) if isinstance(el, RaggedComponents) else el
            for el in elements)
        elements = tf.nest.pack_sequence_as(specs, elements)
        elements = _apply_func(post_batch_map_func, elements)
        return elements

    return dataset.apply(tf.data.experimental.scan(
        initial_state(),
        scan_fn)).filter(filter_fn).map(map_func, num_parallel_calls)


def batch_with_batch_size(
        dataset: tf.data.Dataset,
        batch_size: int,
        post_batch_map_func: Optional[Callable] = None,
        num_parallel_calls: int = tf.data.experimental.AUTOTUNE,
        row_splits_dtype=tf.int64,
        drop_remainder=False):
    spec = dataset.element_spec
    flat_spec = tf.nest.flatten(spec)

    dataset = dataset.apply(
        tf.data.experimental.dense_to_ragged_batch(
            batch_size, drop_remainder=drop_remainder))

    if all(s.shape[0] is None
           for s in flat_spec) and post_batch_map_func is None:
        return dataset

    if isinstance(spec, collections.Mapping):

        def map_func(**elements):
            elements = tf.nest.map_structure(
                lambda e, s: e
                if s.shape[0] is None else e.to_tensor(), elements, spec)
            if post_batch_map_func is not None:
                elements = post_batch_map_func(**elements)
            return elements
    elif isinstance(spec, collections.Sequence):

        def map_func(*elements):
            elements = tf.nest.map_structure(
                lambda e, s: e
                if s.shape[0] is None else e.to_tensor(), elements, spec)
            if post_batch_map_func is not None:
                elements = post_batch_map_func(*elements)
            return elements
    elif isinstance(spec, tf.TensorSpec):
        if post_batch_map_func is None:
            return dataset

        def map_func(el):
            return post_batch_map_func(el)
    else:
        raise ValueError(f'Unrecognized dataset spec {spec}')

    dataset = dataset.map(map_func, num_parallel_calls)
    return dataset


if __name__ == '__main__':
    import numpy as np
    data = (
        # np.zeros((3,)),
        # np.zeros((3,)),
        # np.zeros((3,)),
        # np.zeros((5,)),
        # np.zeros((10,)),
        # np.zeros((3,)),
        # np.zeros((4,)),
        dict(
            x=np.zeros((3,)),
            y=np.zeros((3,)),
        ),
        dict(
            x=np.zeros((3,)),
            y=np.zeros((3,)),
        ),
        dict(
            x=np.zeros((3,)),
            y=np.zeros((3,)),
        ),
        dict(
            x=np.zeros((2,)),
            y=np.zeros((2,)),
        ),
    )

    def gen():
        return data

    base = tf.data.Dataset.from_generator(
        gen, tf.nest.map_structure(lambda x: tf.float64, data[0]),
        tf.nest.map_structure(lambda x: (None,), data[0]))

    def map_fn(x, y):
        return x + y

    batched = batch_with_max_nodes(base,
                                   6,
                                   dataset_size=len(data),
                                   skip_excessive_examples=True,
                                   post_batch_map_func=map_fn)
    for example in batched:
        print(tf.nest.map_structure(lambda x: x.values.shape, example))

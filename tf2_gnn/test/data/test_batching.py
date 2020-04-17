from typing import Tuple
import numpy as np
import tensorflow as tf
from tf2_gnn.data.batching import batch_with_max_nodes


def get_dataset(data: Tuple, shapes=None):
    if shapes is None:
        shapes = tf.nest.map_structure(lambda x: (None,), data[0])
    return tf.data.Dataset.from_generator(
        lambda: data, tf.nest.map_structure(lambda x: x.dtype, data[0]), shapes)


def get_batched_sizes(data, max_nodes_per_batch, shapes=None, **kwargs):
    def values(x):
        return x.values
    batched = batch_with_max_nodes(
        get_dataset(data, shapes), max_nodes_per_batch,
        post_batch_map_func=values, **kwargs)
    return [b.shape[0] for b in batched]


def test_basic_max_node_batching():
    data = (
        np.arange(2),
        np.arange(3),
        np.arange(5),
        np.arange(4),
    )
    sizes = get_batched_sizes(data, 5)
    # we lose the final incomplete batch (a single example) without dataset_size
    np.testing.assert_equal(sizes, [5, 5])
    sizes = get_batched_sizes(data, 5, dataset_size=len(data))
    # we lose the last element when it occurs immediately after a returned value
    # [5, 5, 4] would be nicer, but [5, 5] is what's detailed in the docs
    np.testing.assert_equal(sizes, [5, 5])

    data = (
        np.arange(2),
        np.arange(3),
        np.arange(5),
        np.arange(2),
        np.arange(2),
    )
    sizes = get_batched_sizes(data, 5, dataset_size=len(data))
    np.testing.assert_equal(sizes, [5, 5, 4])

    # we lose the final incomplete batch without dataset_size, even when there
    # are multiple elements
    sizes = get_batched_sizes(data, 5)
    np.testing.assert_equal(sizes, [5, 5])


def test_sequence_batching():
    data = (
        (np.arange(2), np.zeros((2,))),
        (np.arange(3), np.zeros((2,))),
        (np.arange(5), np.zeros((2,))),
        (np.arange(4), np.zeros((2,))),
    )
    dataset = get_dataset(data, shapes=((None,), (2,)))

    batched = list(batch_with_max_nodes(dataset, 5))
    np.testing.assert_equal(len(batched), 2)
    
    a, b = batched[0]
    np.testing.assert_equal(a.values.shape.as_list(), [5])
    np.testing.assert_equal(a.row_splits.numpy(), [0, 2, 5])
    np.testing.assert_equal(b.shape.as_list(), [2, 2])
    a, b = batched[1]
    np.testing.assert_equal(a.values.shape.as_list(), [5])
    np.testing.assert_equal(a.row_splits.numpy(), [0, 5])
    np.testing.assert_equal(b.shape.as_list(), [1, 2])


def test_dict_batching():
    data = (
        dict(a=np.arange(2), b=np.zeros((2,))),
        dict(a=np.arange(3), b=np.zeros((2,))),
        dict(a=np.arange(5), b=np.zeros((2,))),
        dict(a=np.arange(4), b=np.zeros((2,))),
    )
    dataset = get_dataset(data, shapes=dict(a=(None,), b=(2,)))

    batched = list(batch_with_max_nodes(dataset, 5))
    np.testing.assert_equal(len(batched), 2)
    
    a, b = tf.nest.flatten(batched[0])
    np.testing.assert_equal(a.values.shape.as_list(), [5])
    np.testing.assert_equal(a.row_splits.numpy(), [0, 2, 5])
    np.testing.assert_equal(b.shape.as_list(), [2, 2])
    a, b = tf.nest.flatten(batched[1])
    np.testing.assert_equal(a.values.shape.as_list(), [5])
    np.testing.assert_equal(a.row_splits.numpy(), [0, 5])
    np.testing.assert_equal(b.shape.as_list(), [1, 2])



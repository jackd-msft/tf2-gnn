import numpy as np
import tensorflow as tf

from tf2_gnn.data import batching


def test_flat_values_accumulator():

    def gen():
        yield tf.ones((2, 2))
        yield tf.ones((3, 2))

        yield tf.ones((4, 2))

        yield tf.ones((2, 2))
        yield tf.ones((2, 2))

        yield tf.ones((2, 2))

    base = tf.data.Dataset.from_generator(gen, tf.float32, (None, 2))
    acc = batching.FlatValuesAccumulator(spec=tf.TensorSpec((None, 2),
                                                            dtype=tf.float32),
                                         max_size=5)

    batched = tuple(acc.batch(base))
    np.testing.assert_equal(len(batched), 3)
    np.testing.assert_equal(tuple(batched[0].shape), (5, 2))
    np.testing.assert_equal(tuple(batched[1].shape), (4, 2))
    np.testing.assert_equal(tuple(batched[2].shape), (4, 2))

    acc = batching.FlatValuesAccumulator(spec=tf.TensorSpec((None, 2),
                                                            dtype=tf.float32),
                                         max_size=5,
                                         allow_empty=False)
    batched = tuple(acc.batch(batching.append_empty_element(base)))
    np.testing.assert_equal(len(batched), 4)
    np.testing.assert_equal(tuple(batched[3].shape), (2, 2))


def test_ragged_accumulator():

    def gen():
        yield tf.ones((2, 2))
        yield tf.ones((3, 2))

        yield tf.ones((4, 2))

        yield tf.ones((2, 2))
        yield tf.ones((2, 2))

        yield tf.ones((2, 2))

    base = tf.data.Dataset.from_generator(gen, tf.float32, (None, 2))
    acc = batching.RaggedAccumulator(spec=tf.TensorSpec((None, 2),
                                                        dtype=tf.float32),
                                     max_size=5)

    batched = tuple(acc.batch(base))
    np.testing.assert_equal(len(batched), 3)
    np.testing.assert_equal(tuple(batched[0].values.shape), (5, 2))
    np.testing.assert_equal(batched[0].row_splits.numpy(), (0, 2, 5))
    np.testing.assert_equal(tuple(batched[1].values.shape), (4, 2))
    np.testing.assert_equal(batched[1].row_splits.numpy(), (0, 4))
    np.testing.assert_equal(tuple(batched[2].values.shape), (4, 2))
    np.testing.assert_equal(batched[2].row_splits.numpy(), (0, 2, 4))

    acc = batching.RaggedAccumulator(spec=tf.TensorSpec((None, 2),
                                                        dtype=tf.float32),
                                     max_size=5,
                                     allow_empty=False)
    batched = tuple(acc.batch(batching.append_empty_element(base)))
    np.testing.assert_equal(len(batched), 4)
    np.testing.assert_equal(tuple(batched[3].values.shape), (2, 2))


def test_tuple_accumulator():

    def gen():
        yield tf.ones((2, 2)), tf.ones((2, 2))
        yield tf.ones((3, 2)), tf.ones((2, 2))

        yield tf.ones((4, 2)), tf.ones((2, 2))

    base = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32),
                                          ((None, 2), (2, 2)))

    flat_acc = batching.FlatValuesAccumulator(spec=tf.TensorSpec(
        (None, 2), dtype=tf.float32),
                                              max_size=5,
                                              allow_empty=False)
    rest = batching.TensorAccumulator(
        spec=tf.TensorSpec(shape=(2, 2), dtype=tf.float32))
    acc = batching.TupleAccumulator((flat_acc, rest))
    flat, reg = tuple(zip(*acc.batch(batching.append_empty_element(base),)))

    np.testing.assert_equal(len(flat), 2)
    np.testing.assert_equal(tuple(flat[0].shape), (5, 2))
    np.testing.assert_equal(tuple(flat[1].shape), (4, 2))
    np.testing.assert_equal(tuple(reg[0].shape), (2, 2, 2))
    np.testing.assert_equal(tuple(reg[1].shape), (1, 2, 2))


def test_dict_accumulator():

    def gen():
        yield dict(x=tf.ones((2, 2)), y=tf.ones((2, 2)))
        yield dict(x=tf.ones((3, 2)), y=tf.ones((2, 2)))

        yield dict(x=tf.ones((4, 2)), y=tf.ones((2, 2)))

    spec = dict(x=tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
                y=tf.TensorSpec(shape=(2, 2), dtype=tf.float32))

    base = tf.data.Dataset.from_generator(
        gen, tf.nest.map_structure(lambda s: s.dtype, spec),
        tf.nest.map_structure(lambda s: s.shape, spec))

    flat_acc = batching.FlatValuesAccumulator(spec=spec['x'],
                                              max_size=5,
                                              allow_empty=False)
    rest = batching.TensorAccumulator(spec=spec['y'])
    acc = batching.MappingAccumulator(dict(x=flat_acc, y=rest))
    res = tuple(acc.batch(batching.append_empty_element(base)))

    np.testing.assert_equal(len(res), 2)
    np.testing.assert_equal(tuple(res[0]['x'].shape), (5, 2))
    np.testing.assert_equal(tuple(res[1]['x'].shape), (4, 2))
    np.testing.assert_equal(tuple(res[0]['y'].shape), (2, 2, 2))
    np.testing.assert_equal(tuple(res[1]['y'].shape), (1, 2, 2))


def test_block_diagonal_batch():
    kwargs = dict(max_links=10,)

    def gen():
        nodes = (
            tf.range(3, dtype=tf.float32),
            tf.range(2, dtype=tf.float32),
            tf.range(2, dtype=tf.float32),
        )
        edges = (
            tf.constant([
                [0, 1],
                [0, 2],
                [1, 2],
            ], dtype=tf.int32),
            tf.constant([
                [0, 1],
            ], dtype=tf.int32),
            tf.constant([
                [1, 0],
            ], dtype=tf.int32),
        )

        labels = (0, 1, 2)
        for i in range(3):
            yield (nodes[i], (edges[i],)), labels[i]

    dtypes = (tf.float32, (tf.int32,)), tf.int64
    shapes = (tf.TensorShape((None,)), (tf.TensorShape((None, 2)),)), ()

    dataset = tf.data.Dataset.from_generator(gen, dtypes, shapes)

    # final batch completes on second last element
    max_nodes = 5
    batched = batching.block_diagonal_batch_with_max_nodes(dataset,
                                                           max_nodes=max_nodes,
                                                           **kwargs)
    data = tuple(batched)
    np.testing.assert_equal(len(data), 2)

    (node, (links,)), labels = data[0]
    np.testing.assert_equal(node.values.numpy(), [0, 1, 2, 0, 1])
    np.testing.assert_equal(node.row_splits.numpy(), [0, 3, 5])
    np.testing.assert_equal(links.numpy(), [
        [0, 1],
        [0, 2],
        [1, 2],
        [3, 4],
    ])
    np.testing.assert_equal(labels.numpy(), [0, 1])

    (node, (links,)), labels = data[1]
    np.testing.assert_equal(node.values.numpy(), [0, 1])
    np.testing.assert_equal(node.row_splits.numpy(), [0, 2])
    np.testing.assert_equal(links.numpy(), [
        [1, 0],
    ])
    np.testing.assert_equal(labels.numpy(), [2])

    # 0, (1, 2) grouping
    max_nodes = 4
    batched = batching.block_diagonal_batch_with_max_nodes(dataset,
                                                           max_nodes=max_nodes,
                                                           **kwargs)
    data = tuple(batched)
    np.testing.assert_equal(len(data), 2)
    (node, (links,)), labels = data[0]
    np.testing.assert_equal(node.values.numpy(), [0, 1, 2])
    np.testing.assert_equal(node.row_splits.numpy(), [0, 3])
    np.testing.assert_equal(links.numpy(), [[0, 1], [0, 2], [1, 2]])
    np.testing.assert_equal(labels.numpy(), [0])

    (node, (links,)), labels = data[1]
    np.testing.assert_equal(node.values.numpy(), [0, 1, 0, 1])
    np.testing.assert_equal(node.row_splits.numpy(), [0, 2, 4])
    np.testing.assert_equal(links.numpy(), [[0, 1], [3, 2]])
    np.testing.assert_equal(labels.numpy(), [1, 2])

    # (0, 1), (2,)) grouping
    # here we lose the final element
    max_nodes = 6
    batched = batching.block_diagonal_batch_with_max_nodes(dataset,
                                                           max_nodes=max_nodes,
                                                           **kwargs)
    data = tuple(batched)
    np.testing.assert_equal(len(data), 2)
    (node, (links,)), labels = data[0]
    np.testing.assert_equal(node.values.numpy(), [0, 1, 2, 0, 1])
    np.testing.assert_equal(node.row_splits.numpy(), [0, 3, 5])
    np.testing.assert_equal(links.numpy(), [
        [0, 1],
        [0, 2],
        [1, 2],
        [3, 4],
    ])
    np.testing.assert_equal(labels.numpy(), [0, 1])

    (node, (links,)), labels = data[1]
    np.testing.assert_equal(node.values.numpy(), [0, 1])
    np.testing.assert_equal(node.row_splits.numpy(), [0, 2])
    np.testing.assert_equal(links.numpy(), [
        [1, 0],
    ])

    # (0, 1, 2)) grouping
    # here we lose the final element
    max_nodes = 10
    batched = batching.block_diagonal_batch_with_max_nodes(dataset,
                                                           max_nodes=max_nodes,
                                                           **kwargs)
    data = tuple(batched)
    np.testing.assert_equal(len(data), 1)
    (node, (links,)), labels = data[0]
    np.testing.assert_equal(node.values.numpy(), [0, 1, 2, 0, 1, 0, 1])
    np.testing.assert_equal(node.row_splits.numpy(), [0, 3, 5, 7])
    np.testing.assert_equal(links.numpy(), [
        [0, 1],
        [0, 2],
        [1, 2],
        [3, 4],
        [6, 5],
    ])
    np.testing.assert_equal(labels.numpy(), [0, 1, 2])

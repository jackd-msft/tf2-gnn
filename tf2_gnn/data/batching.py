"""
Batching functions for graph networks.

Input datasets are assumed to have elements (inputs, labels), where
    inputs: dict with (node_feature, links) keys. Links may be a tuple or single [num_edges, 2]
    labels: node-wise or graph-wise labels. If leading dimension is unknown/dynamic it is assumed
        to match with that of node_features.

After batching,
    inputs = dict
        node_features: [batch_size, num_nodes?, ...]
        links: [total_num_edges, 2] (or tuple) with indices into flat node features
        labels: [total_num_nodes, ...] or [batch_size, ...]
"""
from typing import Tuple
import tensorflow as tf


def dataset_mean(dataset: tf.data.Dataset):

    def reduce_fn(old_state, n):
        total, count = old_state
        return total + n, count + 1

    total, count = dataset.reduce((0, 0), reduce_fn)
    return total / count


def dataset_sum(dataset: tf.data.Dataset):

    def reduce_fn(old_state, n):
        return old_state + n

    return dataset.reduce(0, reduce_fn)


AUTOTUNE = tf.data.experimental.AUTOTUNE


def _count_nodes_per_epoch(dataset: tf.data.Dataset):
    return dataset_sum(dataset.map(lambda i, l: tf.shape(i['node_features'])[0]))


def count_nodes_per_epoch(dataset: tf.data.Dataset):
    if tf.executing_eagerly():
        return _count_nodes_per_epoch(dataset).numpy()
    count = _count_nodes_per_epoch(dataset)
    with tf.compat.v1.Session() as sess:
        return sess.run(count)


def _validate_graph_spec(element_spec) -> Tuple[bool, bool]:
    inputs_spec, labels_spec = element_spec

    node_features_spec = inputs_spec['node_features']
    link_specs = inputs_spec['links']
    node_features_spec.shape.assert_has_rank(2)
    single_edge_type = isinstance(link_specs, tf.TensorSpec)
    if single_edge_type:
        link_specs = link_specs,
    for spec in link_specs:
        spec.shape.assert_has_rank(2)
        assert spec.shape[1] == 2

    labels_shape = labels_spec.shape
    is_node_labels = labels_shape.ndims > 0 and labels_shape[0] is None
    return is_node_labels, single_edge_type


def batch_with_batch_size(dataset: tf.data.Dataset,
                          batch_size: int,
                          num_parallel_calls: int = AUTOTUNE):
    is_node_labels, single_edge_type = _validate_graph_spec(dataset.element_spec)

    def pre_batch_map(inputs, labels):
        if single_edge_type:
            inputs['links'] = inputs['links'],
        lengths = tf.nest.map_structure(lambda x: tf.shape(x)[0], inputs)
        return inputs, labels, lengths

    def post_batch_map(inputs, labels, lengths):
        node_features = tf.RaggedTensor.from_tensor(inputs['node_features'],
                                                    lengths['node_features'])
        links = inputs['links']
        assert isinstance(links, tuple)
        dtype = links[0].dtype
        offsets = tf.cast(node_features.row_starts(), dtype)
        offsets = tf.expand_dims(offsets, axis=-1)

        def block_diagonalize(links, lengths):
            links = tf.RaggedTensor.from_tensor(links, lengths)
            return links.values + tf.gather(offsets, links.value_rowids())

        links = tf.nest.map_structure(block_diagonalize, links, lengths['links'])
        links = tuple(tf.cast(l, tf.int32) for l in links)
        inputs = dict(node_features=node_features, links=links)
        if is_node_labels:
            labels = tf.boolean_mask(labels, tf.sequence_mask(lengths['node_features']))

        return inputs, labels

    dataset = dataset.map(pre_batch_map)
    dataset = dataset.padded_batch(batch_size,
                                   tf.nest.map_structure(lambda x: x.shape, dataset.element_spec))
    dataset = dataset.map(post_batch_map, num_parallel_calls)
    return dataset


def batch_with_max_nodes(dataset: tf.data.Dataset,
                         max_nodes_per_batch: int,
                         num_parallel_calls: int = AUTOTUNE) -> tf.data.Dataset:

    input_spec, labels_spec = dataset.element_spec
    node_features_spec = input_spec['node_features']
    is_node_labels, single_edge_type = _validate_graph_spec((input_spec, labels_spec))
    link_specs = input_spec['links']
    if single_edge_type:
        link_specs = link_specs,
    link_dtype = link_specs[0].dtype
    num_edge_types = len(link_specs)

    @tf.function
    def scan_fn(old_state, input_element):
        node_features, links, labels, node_row_splits = old_state
        next_inputs, next_labels = input_element
        next_node_features = next_inputs['node_features']
        next_links = next_inputs['links']
        if single_edge_type:
            next_links = next_links,

        if not is_node_labels:
            next_labels = tf.expand_dims(next_labels, axis=0)

        offset = node_row_splits[-1]
        nn = tf.shape(next_node_features, out_type=node_row_splits.dtype)[0]
        num_nodes = offset + nn

        if num_nodes > max_nodes_per_batch:
            output_element = (node_features, links, labels, node_row_splits, True)
            node_row_splits = tf.stack([0, nn], axis=0)
            new_state = (next_node_features, next_links, next_labels, node_row_splits)
            return new_state, output_element
        else:
            node_row_splits = tf.concat(
                [node_row_splits, tf.expand_dims(num_nodes, axis=0)], axis=0)
            node_features = tf.concat([node_features, next_node_features], axis=0)
            links = tuple(
                tf.concat([link, next_link + offset], axis=0)
                for link, next_link in zip(links, next_links))
            labels = tf.concat([labels, next_labels], axis=0)
            new_state = (node_features, links, labels, node_row_splits)
            output_element = (node_features, links, labels, node_row_splits, False)
            return new_state, output_element

    def filter_fn(*args):
        return args[-1]

    def map_fn(node_features, links, labels, node_row_splits, done):
        # return node_features, links, labels, node_row_splits
        node_features = tf.RaggedTensor.from_row_splits(node_features, node_row_splits)
        links = tuple(tf.cast(l, tf.int32) for l in links)
        return dict(node_features=node_features, links=links), labels

    node_features = tf.zeros((0, node_features_spec.shape[1]), dtype=node_features_spec.dtype)
    link = tf.zeros((0, 2), dtype=link_dtype)
    labels = tf.zeros((0, *labels_spec.shape[1 if is_node_labels else 0:]), dtype=labels_spec.dtype)
    node_row_splits = tf.zeros((1,), dtype=link.dtype)
    links = (link,) * num_edge_types
    init_state = (node_features, links, labels, node_row_splits)

    return dataset.apply(tf.data.experimental.scan(init_state,
                                                   scan_fn)).filter(filter_fn).map(map_fn)

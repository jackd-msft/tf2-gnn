import numpy as np
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt

import tensorflow as tf
from tf2_gnn.data import source
from tf2_gnn.data import ppi

builder = ppi.PPI()
builder.download_and_prepare()
base = source.TfdsSource(builder, metadata=dict(num_classes=ppi.NUM_CLASSES))
ds = base.get_dataset('train', repeats=-1)


def reduce_fn(old_state, element):
    total, count = old_state
    n = tf.shape(element[0]['node_features'])[0]
    return total + n, count + 1


total, count = ds.reduce((0, 0), reduce_fn)
mean_nodes_per_batch = total / count
print(f'Mean nodes per batch: {mean_nodes_per_batch.numpy()}')
print('{} training examples'.format(base.examples_per_epoch('train')))
print('{} validation examples'.format(base.examples_per_epoch('validation')))
print('{} test examples'.format(base.examples_per_epoch('test')))

batched = source.NodeBatchedGraphSource(base,
                                        max_nodes_per_batch=8000,
                                        approx_nodes_per_example=2250)

for example in batched.get_dataset('train'):
    # node_features, links, labels, node_row_splits = example
    # print('---')
    # print(node_features.shape)
    # print(labels.shape)
    # print(node_row_splits.shape)
    # print(node_row_splits)
    # print(tf.reduce_min(links), tf.reduce_max(links))
    # print(links.shape)
    inputs, labels = example
    node_features = inputs['node_features']
    print(node_features.row_splits.numpy())
    links = inputs['links']
    num_nodes = node_features.row_splits[-1].numpy()
    num_links = links.shape[0]
    print(f'batch_size:  {node_features.nrows().numpy()}')
    print(f'num_nodes:   {num_nodes}')
    print(f'num_links:   {num_links}')
    print(f'mean_degree: {num_links / num_nodes}')
    i, j = tf.unstack(links, axis=-1)
    vals = np.ones(i.shape.as_list(), dtype=np.bool)
    plt.spy(coo_matrix((vals, (i.numpy(), j.numpy()))), markersize=0.01)
    plt.show()

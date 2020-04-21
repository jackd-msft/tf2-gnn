from absl import app, flags
import numpy as np
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt

import tensorflow as tf
from tf2_gnn.data import builders
from tf2_gnn.data.batching import block_diagonal_batch

flags.DEFINE_string('builder', default='ppi', help='one of ppi, qm9')
flags.DEFINE_integer('max_nodes', default=None, help='maximum number of nodes')
flags.DEFINE_string('split',
                    default='train',
                    help='dataset split, one of train, validation, test')


def main(_):
    FLAGS = flags.FLAGS
    builder = FLAGS.builder
    max_nodes = FLAGS.max_nodes

    constructor, max_nodes, marker_size = {
        'ppi': (builders.PPI, 8000, 0.02),
        'qm9': (builders.QM9, 100, 1.0)
    }[FLAGS.builder]
    builder = constructor()
    if FLAGS.max_nodes is not None:
        max_nodes = FLAGS.max_nodes

    builder.download_and_prepare()
    dataset = builder.as_dataset(
        split=FLAGS.split, as_supervised=True).shuffle(20).repeat()
    dataset = block_diagonal_batch(
        dataset,
        max_nodes)
    
    for example in dataset:
        inputs, labels = example
        node_features, links = inputs
        num_nodes = node_features.row_splits[-1].numpy()
        print(f'batch_size:   {node_features.nrows().numpy()}')
        print(f'num_nodes:    {num_nodes}')
        num_edge_types = len(links)
        for i, l in enumerate(links):
            num_links = l.shape[0]
            print(f' link {i+1} / {num_edge_types}')
            print(f'  num_links:    {num_links}')
            print(f'  mean_degree:  {num_links / num_nodes}')   
        print(f'labels_shape: {labels.shape}')

        _, axes = plt.subplots(1, len(links))
        if num_edge_types == 1:
            axes = axes,

        for l, ax in zip(links, axes):
            if l.shape[0] == 0:
                continue
            i, j = tf.unstack(l, axis=-1)
            vals = np.ones(i.shape.as_list(), dtype=np.bool)
            ax.spy(coo_matrix((vals, (i.numpy(), j.numpy()))),
                   markersize=marker_size)
        plt.show()


if __name__ == '__main__':
    app.run(main)

from typing import Callable, Optional

import tensorflow as tf
from absl import app, flags

flags.DEFINE_string('problem', default='ppi', help='one of ppi, qm9')
flags.DEFINE_bool('tfds', default=False, help='use tfds-based builder')
flags.DEFINE_integer('max_nodes', default=10000, help='maximum number of nodes')
flags.DEFINE_integer('batch_size', default=None, help='maximum number of nodes')
flags.DEFINE_string('split', default='train', help='train, validation, test')
flags.DEFINE_integer('burn_iters', default=10, help='burn iterations')
flags.DEFINE_integer('min_iters', default=50, help='minimum run iters')

FLAGS = flags.FLAGS


def summarize(result, print_fn=print):
    """
    Args:
        result: output of a tf.test.Benchmark.run_op_benchmark call.
        print_fn: print-like function.
    """
    print_fn('Wall time (ms): {}'.format(result['wall_time'] * 1000))
    gpu_mem = result['extras'].get('allocator_maximum_num_bytes_GPU_0_bfc', 0)
    print_fn('Memory (Mb):    {}'.format(gpu_mem / 1024**2))


def summarize_all(*args, print_fn=print):
    """
    Applies `summarize` to (name, result) pairs.

    Args:
        *args: (name, result) pairs
        print_fn: print-like function.
    """
    for name, result in args:
        print_fn(name)
        summarize(result, print_fn)


def benchmark_dataset(dataset_fn: Callable[[], tf.data.Dataset],
                      burn_iters: int, min_iters: int):
    with tf.Graph().as_default() as graph:
        dataset = dataset_fn()
        element = tf.compat.v1.data.make_one_shot_iterator(
            dataset.repeat()).get_next()
        with tf.compat.v1.Session(graph=graph) as sess:
            bm = tf.test.Benchmark()
            print('Starting benchmarking...')
            result = bm.run_op_benchmark(sess,
                                         element,
                                         burn_iters=burn_iters,
                                         min_iters=min_iters)
            summarize(result)


def get_tfds_dataset_fn(problem: str,
                        split: str,
                        max_nodes: Optional[int] = None,
                        batch_size: Optional[int] = None):
    from tf2_gnn.data import builders
    from tf2_gnn.data.batching import block_diagonal_batch_with_max_nodes
    from tf2_gnn.data.batching import block_diagonal_batch_with_batch_size
    builder = {
        'ppi': builders.PPI,
        'qm9': builders.QM9,
    }[problem]()
    builder.download_and_prepare()

    def add_back_edges(inputs, labels):
        nodes, links = inputs
        links = links + tuple(tf.reverse(l, axis=[1]) for l in links)
        return (nodes, links), labels

    def f():
        dataset = builder.as_dataset(split=split,
                                     as_supervised=True).shuffle(256)

        if batch_size is None:
            assert isinstance(max_nodes, int)
            dataset = block_diagonal_batch_with_max_nodes(dataset, max_nodes)
        else:
            dataset = block_diagonal_batch_with_batch_size(dataset, batch_size)
        dataset = dataset.map(add_back_edges)  # for comparison with base
        return dataset

    return f


def get_baseline_dataset_fn(problem: str, split: str, max_nodes: int):
    from tf2_gnn import data
    from dpu_utils.utils import RichPath
    graph_dataset_cls = {
        'ppi': data.PPIDataset,
        'qm9': data.QM9Dataset
    }[problem]
    params = graph_dataset_cls.get_default_hyperparameters()
    params.update(
        dict(
            max_nodes=max_nodes,
            add_self_loop_edges=False,
            tie_fwd_bkwd_edges=False,
        ))
    graph_dataset = graph_dataset_cls(params)
    graph_dataset.load_data(RichPath.create(f'data/{problem}'))
    fold = {
        'train': data.DataFold.TRAIN,
        'validation': data.DataFold.VALIDATION,
        'test': data.DataFold.TEST,
    }[split]

    def f():
        return graph_dataset.get_tensorflow_dataset(fold)

    return f


def main(_):
    FLAGS = flags.FLAGS
    kwargs = dict(problem=FLAGS.problem, split=FLAGS.split)

    max_nodes = FLAGS.max_nodes
    batch_size = FLAGS.batch_size
    if batch_size is None:
        kwargs['max_nodes'] = max_nodes
    else:
        if not FLAGS.tfds:
            raise NotImplementedError(
                'batch_size implementation only available for tfds '
                'implementations.')
        kwargs['batch_size'] = batch_size
    if FLAGS.tfds:
        dataset_fn = get_tfds_dataset_fn(**kwargs)
    else:
        dataset_fn = get_baseline_dataset_fn(**kwargs)
    benchmark_dataset(dataset_fn,
                      burn_iters=FLAGS.burn_iters,
                      min_iters=FLAGS.min_iters)
    if not FLAGS.tfds:
        print('Benchmark complete. Please ignore possible GeneratorDataset '
              'iterator errors and kill program')
    exit(0)


if __name__ == '__main__':
    app.run(main)

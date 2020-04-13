from absl import app, flags, logging
import tensorflow as tf
from tf2_gnn.data import get_source, DataSource

flags.DEFINE_string('source', default='ppi', help='one of ppi, qm9')
flags.DEFINE_integer('batch_size', default=None, help='batch size')
flags.DEFINE_integer('max_nodes', default=None, help='maximum number of nodes')
flags.DEFINE_string('split', default='train', help='trai, eval, test')
flags.DEFINE_integer('burn_iters', default=50, help='burn iterations')
flags.DEFINE_integer('min_iters', default=500, help='minimum run iters')


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


def benchmark_source(source: DataSource, split: str, burn_iters: int, min_iters: int):
    with tf.Graph().as_default() as graph:
        dataset = source.get_dataset(split, repeats=None)
        inputs, labels = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
        count = tf.Variable(0, dtype=tf.int64)
        update_count = count.assign(count + inputs['node_features'].nrows())
        op = (inputs, labels, update_count)
        bm = tf.test.Benchmark()
        with tf.compat.v1.Session(graph=graph) as sess:
            logging.info('Starting benchmarking...')
            sess.run(count.initializer)
            result = bm.run_op_benchmark(sess, op, burn_iters=burn_iters, min_iters=min_iters)
            summarize(result)
            c = sess.run(count)
        logging.info(f'Total examples: {c}')


def main(_):
    FLAGS = flags.FLAGS
    source = get_source(FLAGS.source,
                        batch_size=FLAGS.batch_size,
                        max_nodes_per_batch=FLAGS.max_nodes)
    benchmark_source(source, FLAGS.split, burn_iters=FLAGS.burn_iters, min_iters=FLAGS.min_iters)


if __name__ == '__main__':
    app.run(main)

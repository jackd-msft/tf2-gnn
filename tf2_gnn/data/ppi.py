
from typing import Tuple
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from dpu_utils.utils import RichPath

NUM_FEATURES = 50
NUM_CLASSES = 121

CITATION = '''\
@article{zitnik2017predicting,
  title={Predicting multicellular function through multi-layer tissue networks},
  author={Zitnik, Marinka and Leskovec, Jure},
  journal={Bioinformatics},
  volume={33},
  number={14},
  pages={i190--i198},
  year={2017},
  publisher={Oxford University Press}
}
'''


def raggedify_batched_graphs(node_graph_ids: np.ndarray, node_features: np.ndarray,
                             node_labels: np.ndarray, links: np.ndarray) -> Tuple[
                                 tf.RaggedTensor, tf.RaggedTensor, tf.RaggedTensor]:
    node_features = tf.RaggedTensor.from_value_rowids(node_features, node_graph_ids)
    offset = node_features.row_starts()
    link_graph_ids = node_graph_ids[links[:, 0]]
    links = tf.RaggedTensor.from_value_rowids(links, link_graph_ids)
    links -= tf.reshape(offset, (-1, 1, 1))
    # node_labels = node_features.with_values(node_labels)
    node_labels = tf.RaggedTensor.from_row_splits(node_labels, node_features.row_splits)
    return node_features, node_labels, links


class PPI(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('0.0.1')

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.core.features.FeaturesDict(
                dict(graph=dict(node_features=tfds.core.features.Tensor(shape=(None, NUM_FEATURES),
                                                                        dtype=tf.float32),
                                links=tfds.core.features.Tensor(shape=(None, 2), dtype=tf.int64)),
                     node_labels=tfds.core.features.Tensor(shape=(None, NUM_CLASSES),
                                                           dtype=tf.bool))),
            supervised_keys=('graph', 'node_labels'),
            citation=CITATION)

    def _split_generators(self, dl_manager):
        data_dir: str = dl_manager.download_and_extract(
            'https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/ppi.zip')
        data_dir = RichPath.create(data_dir)
        return [
            tfds.core.SplitGenerator(name=tfds.Split.TRAIN,
                                     gen_kwargs=dict(data_dir=data_dir, data_name='train')),
            tfds.core.SplitGenerator(name=tfds.Split.VALIDATION,
                                     gen_kwargs=dict(data_dir=data_dir, data_name='valid')),
            tfds.core.SplitGenerator(name=tfds.Split.TEST,
                                     gen_kwargs=dict(data_dir=data_dir, data_name='test')),
        ]

    def _generate_examples(self, data_dir: RichPath, data_name: str):
        graph_data = data_dir.join(f"{data_name}_graph.json").read_by_file_suffix()
        node_features = data_dir.join(f"{data_name}_feats.npy").read_by_file_suffix()
        node_labels = data_dir.join(f"{data_name}_labels.npy").read_by_file_suffix()
        # 1-based node value_rowids
        node_graph_ids = data_dir.join(f"{data_name}_graph_id.npy").read_by_file_suffix()
        node_graph_ids -= 1  # make zero-based

        links = graph_data['links']
        links = np.array(tuple((link['source'], link['target']) for link in links), dtype=np.int64)
        node_features, node_labels, links = raggedify_batched_graphs(
            node_graph_ids, node_features, node_labels, links)
        for i in range(node_features.nrows().numpy()):
            yield i, dict(graph=dict(node_features=node_features[i], links=links[i]),
                          node_labels=node_labels[i])


if __name__ == '__main__':
    import tf2_gnn.data.checksums
    # config = tfds.core.download.DownloadConfig(register_checksums=True)
    config = None
    ppi = PPI()
    ppi.download_and_prepare(download_config=config)

import collections
from typing import Tuple
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from dpu_utils.utils import RichPath

CITATION = '''\
@article{ramakrishnan2014quantum,
  title={Quantum chemistry structures and properties of 134 kilo molecules},
  author={Ramakrishnan, Raghunathan and Dral, Pavlo O and Rupp, Matthias and Von Lilienfeld, O Anatole},
  journal={Scientific data},
  volume={1},
  pages={140022},
  year={2014},
  publisher={Nature Publishing Group}
}
'''

NUM_TARGETS = 13
NUM_FEATURES = 15
NUM_EDGE_TYPES = 4


class FeatureTuple(tfds.core.features.FeatureConnector, collections.abc.Sequence):

    def __init__(self, features: tfds.core.features.FeatureConnector):
        self._features = features

    def __getitem__(self, s):
        return self._features[s]

    def __len__(self):
        return len(self._features)

    def _assert_same_length(self, values):
        if len(values) != len(self):
            raise ValueError(f'values must have {len(self)} entries, got {len(values)}')

    def __flatten(self, x):
        return tuple(x[i] for i in range(len(self)))

    def __nest(self, x):
        return {i: x for i, x in enumerate(x)}

    def get_tensor_info(self):
        return self.__flatten(self.get_serialized_info())

    def get_serialized_info(self):
        return self.__nest(f.get_serialized_info() for f in self._features)

    def encode_example(self, example_data):
        self._assert_same_length(example_data)
        return self.__nest(f.encode_example(x) for f, x in zip(self._features, example_data))

    def decode_example(self, tfexample_data):
        self._assert_same_length(tfexample_data)
        return tuple(
            f.decode_example(e) for f, e in zip(self._features, self.__flatten(tfexample_data)))


class Qm9(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('0.0.1')

    def _info(self):
        link_features = tfds.core.features.Tensor(shape=(None, 2), dtype=tf.int64)
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.core.features.FeaturesDict(
                dict(graph=dict(node_features=tfds.core.features.Tensor(shape=(None, NUM_FEATURES),
                                                                        dtype=tf.float32),
                                links=FeatureTuple((link_features,) * NUM_EDGE_TYPES)),
                     targets=tfds.core.features.Tensor(shape=(NUM_TARGETS,), dtype=tf.float32),
                     example_id=tfds.core.features.Tensor(shape=(), dtype=tf.int64))),
            supervised_keys=('graph', 'targets'),
            citation=CITATION)

    def _split_generators(self, dl_manager):
        # xyz_url = "https://ndownloader.figshare.com/files/3195389"
        url = 'https://github.com/microsoft/tf-gnn-samples/raw/master/data/qm9/{split}.jsonl.gz'
        return [
            tfds.core.SplitGenerator(name=tfds.Split.TRAIN,
                                     gen_kwargs=dict(path=url.format(split='train'))),
            tfds.core.SplitGenerator(name=tfds.Split.VALIDATION,
                                     gen_kwargs=dict(path=url.format(split='valid'))),
            tfds.core.SplitGenerator(name=tfds.Split.TEST,
                                     gen_kwargs=dict(path=url.format(split='test'))),
        ]

    def _generate_examples(self, path: str):
        data = RichPath.create(path).read_as_jsonl()  # tfds mangles path, obscures extension
        for d in data:
            src, edge_id, dst = np.array(d['graph'], dtype=np.int64).T  # pylint:disable=unpacking-non-sequence
            edge_id -= 1
            links = tf.dynamic_partition(np.stack((src, dst), axis=-1),
                                         edge_id,
                                         num_partitions=NUM_EDGE_TYPES)
            node_features = np.array(d['node_features'], dtype=np.float32)
            graph = dict(node_features=node_features, links=links)
            targets = np.squeeze(np.array(d['targets'], dtype=np.float32), axis=-1)
            example_id = int(d['id'][4:])
            yield example_id, dict(graph=graph, targets=targets, example_id=example_id)


QM9 = Qm9  # hack naming issues

if __name__ == '__main__':
    import tf2_gnn.data.checksums  # ensure CHECKSUMS_DIR is registered
    # config = None
    config = tfds.core.download.DownloadConfig(register_checksums=True)
    qm9 = QM9()
    qm9.download_and_prepare(download_config=config)

    dataset = qm9.as_dataset(split='train')
    print(dataset.element_spec)

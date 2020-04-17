import os
import tensorflow_datasets as tfds

CHECKSUMS_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), 'checksums_dir'))
tfds.core.download.add_checksums_dir(CHECKSUMS_DIR)

import time
import os
import gin

from tf2_gnn import models
from tf2_gnn.data import sources

gin.register(sources.ppi, module='tf2_gnn.sources')
gin.register(sources.qm9, module='tf2_gnn.sources')

for fn in (models.graph_binary_classifier, models.graph_regressor,
           models.node_multiclass_classifier):
    gin.register(fn, module='tf2_gnn.models', blacklist=['input_specs', 'metadata'])

gin.register(models.get_optimizer, module='tf2_gnn.utils')


@gin.register
def path(parent: str, folder):
    if not isinstance(folder, str):
        folder = folder.__name__
    return os.path.join(parent, folder)


def _time_default(config):
    key = ('run_id', 'gin.macro')
    if not key in config:
        run_id = f'run_{time.strftime("%Y_%m_%d_%H_%M_%S")}_{os.getpid()}'
        gin.bind_parameter((*key, 'value'), run_id)


# register_finalize_hook appends the hook - but we want it to happen before validating macros
gin.config._FINALIZE_HOOKS.insert(0, _time_default)

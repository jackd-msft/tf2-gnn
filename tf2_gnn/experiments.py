from typing import Any, Mapping, Union, Callable, Optional, Tuple
import tensorflow as tf
from . import data
from .utils import register
from .utils import functions

Config = Mapping[str, Any]


@register.register_custom_object
class Experiment(object):

    def __init__(self, source: Union[data.DataSource, Config], model_fn: Union[Callable, Config]):
        self._source = data.get(source)
        self._model_fn = functions.unwrap(functions.get(model_fn))

    def get_config(self):
        return dict(
            source=tf.keras.utils.serialize_keras_object(self._source),
            model_fn=tf.keras.utils.serialize_keras_object(functions.wrap(self._model_fn)),
        )

    @classmethod
    def from_config(cls, config: Config) -> 'Experiment':
        return cls(**config)

    @property
    def source(self) -> data.DataSource:
        return self._source

    @property
    def model_fn(self) -> Callable:
        return self._model_fn


def fit(experiment: Experiment,
        epochs: int,
        model_dir: Optional[str] = None,
        validation_freq: int = 1) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    source = experiment.source
    train_ds, train_steps = source.epoch_data('train')
    validation_ds, val_steps = source.epoch_data('validation')

    model = experiment.model_fn(train_ds.element_spec, source.metadata())

    if model_dir is None:
        callbacks = []
        initial_epoch = 0
    else:
        callbacks = [
            tf.keras.callbacks.TensorBoard(model_dir),
        ]
        initial_epoch = 0

    history = model.fit(
        train_ds,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=validation_ds,
        validation_steps=val_steps,
        callbacks=callbacks,
        intitial_epoch=initial_epoch,
        validation_freq=validation_freq,
    )

    return model, history

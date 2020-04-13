from typing import Union, Dict, Any, Optional, Sequence
import tensorflow as tf

PACKAGE = 'tf2_gnn'

register_custom_object = tf.keras.utils.register_keras_serializable(PACKAGE)


def get(identifier: Union[str, Dict[str, Any]],
        accepted_types=Optional[Union[type, Sequence[type]]]):
    if accepted_types is not None and isinstance(identifier, accepted_types):
        return identifier
    if isinstance(identifier, str):
        identifier = dict(class_name=identifier, config={})
    assert isinstance(identifier, dict)
    class_name = identifier['class_name']
    if '>' not in class_name:
        identifier['class_name'] = f'{PACKAGE}>{class_name}'
    out = tf.keras.utils.deserialize_keras_object(identifier)
    if accepted_types is not None and not isinstance(out, accepted_types):
        raise ValueError(f'Expected type {accepted_types} but got {out} with type {type(out)}')
    return out

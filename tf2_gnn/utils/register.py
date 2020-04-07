from typing import Callable
import tensorflow as tf

_custom_objects = tf.keras.utils.get_custom_objects()


def register_custom_object(fn_or_class: Callable):
    """
    Register fn_or_class as a keras custom_object.

    This allows it to be used with `tf.keras.*.get` methods and
    `tf.keras.utils.deserialize_keras_object`

    Example usage:
    ```python
    @register
    def custom_activation(x):
        return x + 1

    ones = tf.keras.activations.get('custom_activation')(tf.zeros(5, 4))
    layer = tf.keras.layers.Dense(activation='custom_activation')
    ```

    """
    if not callable(fn_or_class):
        raise ValueError(f'fn_or_class must be callable, got {fn_or_class}')

    name = fn_or_class.__name__
    if name in _custom_objects:
        raise KeyError(f'Cannot register {name} - key already present')
    _custom_objects[name] = fn_or_class
    return fn_or_class

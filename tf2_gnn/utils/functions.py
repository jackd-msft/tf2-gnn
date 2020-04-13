from typing import Callable, Mapping, Any, Union
import tensorflow as tf
import functools
import importlib
from . import register


@register.register_custom_object
class Function(object):

    def __init__(self, func: Callable):
        for attr in ('__module__', '__name__'):
            if not hasattr(func, attr):
                raise ValueError(f'func must have a {attr} attribute')

        self._func = func

    def get_config(self):
        return dict(
            module=self._func.__module__,
            name=self._func.__name__,
        )

    @classmethod
    def from_config(cls, config):
        module = config['module']
        name = config['name']
        return getattr(importlib.import_module(module), name)

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    @property
    def func(self):
        return self._func


@register.register_custom_object
class Partial(object):

    def __init__(self, func: Union[Callable, Mapping[str, Any]], *args, **kwargs):
        self._func = unwrap(get(func))
        self._args = args
        self._kwargs = kwargs

    @staticmethod
    def from_functools(partial) -> 'Partial':
        assert isinstance(partial, functools.partial)
        return Partial(partial.func, *partial.args, **partial.keywords)

    def to_functools(self):
        return functools.partial(self.func, *self.args, **self.keywords)

    @property
    def func(self):
        return self._func

    @property
    def args(self):
        return self._args

    @property
    def keywords(self):
        return self._kwargs

    def __call__(self, *args, **kwargs):
        return functools.partial(self._func, *self._args, **self._kwargs)(*args, **kwargs)

    def get_config(self):
        return dict(func=tf.keras.utils.serialize_keras_object(Function(self._func)),
                    args=tuple(self._args),
                    kwargs=self._kwargs.copy())

    @classmethod
    def from_config(cls, config):
        assert len(config) == 3
        return cls(config['func'], *config['args'], **config['kwargs'])


def get(identifier) -> Callable:
    if callable(identifier):
        return identifier
    else:
        out = register.get(identifier)
        assert callable(out)
        return get


def wrap(func: Callable) -> Callable:
    if not hasattr(func, 'get_config'):
        if isinstance(func, functools.partial):
            func = Partial.from_functools(func)
        else:
            func = Function(func)
    return func


def unwrap(func: Callable) -> Callable:
    while isinstance(func, Function):
        func = func.func
    if isinstance(func, Partial):
        func = func.to_functools()
    return func

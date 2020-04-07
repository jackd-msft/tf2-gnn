"""Functions to convert from string parameters to their values."""
import tensorflow as tf

from .activation import gelu


def get_aggregation_function(aggregation_fn_name: str):
    """Convert from an aggregation function name to the function itself."""
    string_to_aggregation_fn = {
        "sum": tf.math.unsorted_segment_sum,
        "max": tf.math.unsorted_segment_max,
        "mean": tf.math.unsorted_segment_mean,
        "sqrt_n": tf.math.unsorted_segment_sqrt_n,
    }
    aggregation_fn = string_to_aggregation_fn.get(aggregation_fn_name)
    if aggregation_fn is None:
        raise ValueError(f"Unknown aggregation function: {aggregation_fn_name}")
    return aggregation_fn


get_activation_function = tf.keras.activations.get

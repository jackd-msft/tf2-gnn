import tensorflow as tf
from tf2_gnn.layers.message_passing.ggnn import GGNN, MessagePassingInput

layer = GGNN()
inp = MessagePassingInput(tf.zeros((10, 3), dtype=tf.float32), (tf.zeros((0, 2), dtype=tf.int32),))

# build needs name scoping :S
# layer(inp)
with tf.name_scope(layer.name):
    layer.build(tf.nest.map_structure(lambda i: i.shape, inp))
print([w.name for w in layer.weights])

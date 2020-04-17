from typing import List, Optional, Tuple, Union
import tensorflow as tf

from tf2_gnn import GNNInput, GNN
from tf2_gnn.layers import WeightedSumGraphRepresentation

K = tf.keras.backend


class ModifiedMicroF1(tf.keras.metrics.Metric):
    """This is different to tensorflow-addons implementation, hence the `Modified`."""
    def __init__(self, name='custom_micro_f1'):
        super().__init__(name=name)
        self.total = self.add_weight('total', initializer='zeros', dtype=tf.float32)
        self.counter = self.add_weight('counter', initializer='zeros', dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        if sample_weight is not None:
            raise NotImplementedError()
        predicted = tf.cast(y_pred > 0, tf.int32)
        labels = tf.cast(y_true, tf.int32)

        true_pos = tf.math.count_nonzero(predicted * labels)
        false_pos = tf.math.count_nonzero(predicted * (labels - 1))
        false_neg = tf.math.count_nonzero((predicted - 1) * labels)

        true_pos = tf.cast(true_pos, tf.float32)
        false_neg = tf.cast(false_neg, tf.float32)
        false_pos = tf.cast(false_pos, tf.float32)

        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        fmeasure = (2 * precision * recall) / (precision + recall)
        self.total.assign_add(fmeasure)
        self.counter.assign_add(1)

    def result(self):
        return self.total / tf.cast(self.counter, tf.float32)


class ModifiedBinaryCrossentropy(tf.keras.losses.Loss):

    def __init__(self, from_logits: bool = True, **kwargs):
        if not from_logits:
            raise NotImplementedError()
        self.from_logits = from_logits
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true)
        # tf.keras.losses.BinaryCrossetnropy finds the mean over the class axis
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
        # return K.sum(K.binary_crossentropy(
        #     y_true, y_pred, from_logits=self.from_logits), axis=-1)


def maybe_block_diagonalize(links: Union[tf.Tensor, tf.RaggedTensor], offset: tf.Tensor):
    if links.shape.ndims == 2:
        assert isinstance(links, tf.Tensor)
        return links
    links.shape.assert_has_rank(3)
    assert links.dtype == offset.dtype
    if isinstance(links, tf.RaggedTensor):
        batch_index = links.value_rowids()
        return links.values + tf.expand_dims(tf.gather(offset, batch_index), axis=-1)
    else:
        b, ij = tf.split(links, [1, 2], axis=-1)
        return ij + tf.gather(offset, b)


def as_gnn_inputs(
        node_features: tf.RaggedTensor, links: Tuple[Union[tf.Tensor, tf.RaggedTensor], ...],
        tie_fwd_bkwd_edges: bool=False, add_self_loop_edges: bool=False) -> GNNInput:
    node_features.shape.assert_has_rank(3)
    assert node_features.ragged_rank == 1
    assert (len(links) >= 1)
    row_splits = node_features.row_splits
    links_dtype = links[0].dtype
    if row_splits.dtype != links_dtype:
        row_splits = tf.cast(row_splits, links_dtype)
    links = tuple(maybe_block_diagonalize(l, row_splits) for l in links)
    for l in links:
        l.shape.assert_has_rank(2)
        assert l.shape[1] == 2
    node_features, node_to_graph_map, num_graphs = tf.keras.layers.Lambda(
        lambda x: (tf.identity(x.values), x.value_rowids(), x.nrows()))(node_features)
    adjacency_lists = list(links)
    back_adjacency_lists = (tf.reverse(l, axis=[1]) for l in links)
    if tie_fwd_bkwd_edges:
        adjacency_lists.extend(back_adjacency_lists)
    else:
        # concat reversed
        adjacency_lists = [
            tf.concat((fwd, bkwd), axis=0)
            for fwd, bkwd in zip(adjacency_lists, back_adjacency_lists)
        ]

    if add_self_loop_edges:

        def get_self_loops(flat_node_features, dtype):
            num_nodes = tf.shape(flat_node_features, out_type=dtype)[0]
            return tf.tile(tf.expand_dims(tf.range(num_nodes), axis=-1), (1, 2))

        self_loops = tf.keras.layers.Lambda(get_self_loops,
                                            arguments=dict(dtype=links_dtype))(node_features)
        adjacency_lists.append(self_loops)

    return GNNInput(node_features=node_features,
                    adjacency_lists=tuple(adjacency_lists),
                    node_to_graph_map=node_to_graph_map,
                    num_graphs=num_graphs)


def _spec_to_input(spec):
    if isinstance(spec, tf.RaggedTensorSpec):
        return tf.keras.Input(batch_shape=spec._shape, dtype=spec._dtype, ragged=True)
    return tf.keras.Input(batch_shape=spec.shape, dtype=spec.dtype)


def spec_to_input(spec_structure):
    return tf.nest.map_structure(_spec_to_input, spec_structure)


def get_optimizer(
        learning_rate=0.001,
        optimizer='Adam',
        learning_rate_decay=0.98,
        momentum=0.85,
        gradient_clip_value=1.0,
):
    """Create fresh optimizer."""
    optimizer_name = optimizer.lower()
    if optimizer_name == "sgd":
        return tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=momentum,
            clipvalue=gradient_clip_value,
        )
    elif optimizer_name == "rmsprop":
        return tf.keras.optimizers.RMSprop(
            learning_rate=learning_rate,
            decay=learning_rate_decay,
            momentum=momentum,
            clipvalue=gradient_clip_value,
        )
    elif optimizer_name == "adam":
        return tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipvalue=gradient_clip_value,
        )
    else:
        raise Exception(f'Unknown optimizer {optimizer}".')


def _gnn_outputs(gnn_input: GNNInput, use_intermediate_gnn_results: bool,
                 message_calculation_class: str, add_self_loop_edges: bool,
                 tie_fwd_bkwd_edges: bool, **kwargs):
    assert all(n.startswith('gnn_') for n in kwargs)
    kwargs = {k[4:]: v for k, v in kwargs.items()}
    gnn_outputs = GNN(dict(message_calculation_class=message_calculation_class,
                      add_self_loop_edges=add_self_loop_edges,
                      tie_fwd_bkwd_edges=tie_fwd_bkwd_edges,
                      **kwargs))(gnn_input, return_all_representations=use_intermediate_gnn_results)
    return gnn_outputs


def _graph_outputs(model_input,
                   graph_aggregation_num_heads: int,
                   graph_aggregation_hidden_layers: List[int],
                   graph_aggregation_dropout_rate: float,
                   use_intermediate_gnn_results: bool,
                   add_self_loop_edges: bool = True,
                   tie_fwd_bkwd_edges: bool = True,
                   **gnn_kwargs):
    gnn_input = as_gnn_inputs(**model_input)
    gnn_outputs = _gnn_outputs(gnn_input,
                               use_intermediate_gnn_results,
                               add_self_loop_edges=add_self_loop_edges,
                               tie_fwd_bkwd_edges=tie_fwd_bkwd_edges,
                               **gnn_kwargs)
    per_graph_results = WeightedSumGraphRepresentation(
        graph_representation_size=graph_aggregation_num_heads,
        num_heads=graph_aggregation_num_heads,
        scoring_mlp_layers=graph_aggregation_hidden_layers,
        scoring_mlp_dropout_rate=graph_aggregation_dropout_rate,
        transformation_mlp_layers=graph_aggregation_hidden_layers,
        transformation_mlp_dropout_rate=graph_aggregation_dropout_rate,
    )(node_embeddings=tf.concat([gnn_input.node_features, gnn_outputs], axis=-1),
      node_to_graphmap=gnn_input.node_to_graph_map,
      num_graphs=gnn_input.num_graphs)
    return per_graph_results


def graph_binary_classifier(input_specs,
                            optimizer,
                            graph_aggregation_num_heads: int = 16,
                            graph_aggregation_hidden_layers: List[int] = [128],
                            graph_aggregation_dropout_rate: float = 0.2,
                            use_intermediate_gnn_results: bool = False,
                            add_self_loop_edges: bool = True,
                            tie_fwd_bkwd_edges: bool = True,
                            **gnn_kwargs) -> tf.keras.Model:
    model_input = spec_to_input(input_specs)
    per_graph_results = _graph_outputs(
        model_input,
        graph_aggregation_num_heads=graph_aggregation_num_heads,
        graph_aggregation_hidden_layers=graph_aggregation_hidden_layers,
        graph_aggregation_dropout_rate=graph_aggregation_dropout_rate,
        use_intermediate_gnn_results=use_intermediate_gnn_results,
        add_self_loop_edges=add_self_loop_edges,
        tie_fwd_bkwd_edges=tie_fwd_bkwd_edges,
        **gnn_kwargs)
    logits = tf.keras.layers.Dense(units=1, activation=None, use_bias=True)(per_graph_results)
    model = tf.keras.Model(inputs=model_input, outputs=logits)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.0)])
    return model


def graph_regressor(input_specs,
                    num_targets: Optional[int] = None,
                    optimizer=None,
                    graph_aggregation_num_heads: int = 16,
                    graph_aggregation_hidden_layers: List[int] = [128],
                    graph_aggregation_dropout_rate: float = 0.2,
                    use_intermediate_gnn_results: bool = False,
                    add_self_loop_edges: bool = True,
                    tie_fwd_bkwd_edges: bool = True,
                    **gnn_kwargs) -> tf.keras.Model:

    model_input = spec_to_input(input_specs)
    per_graph_results = _graph_outputs(
        model_input,
        use_intermediate_gnn_results=use_intermediate_gnn_results,
        add_self_loop_edges=add_self_loop_edges,
        tie_fwd_bkwd_edges=tie_fwd_bkwd_edges,
        graph_aggregation_num_heads=graph_aggregation_num_heads,
        graph_aggregation_hidden_layers=graph_aggregation_hidden_layers,
        graph_aggregation_dropout_rate=graph_aggregation_dropout_rate,
        **gnn_kwargs)

    if num_targets is not None:
        output = tf.keras.layers.Dense(num_targets)(per_graph_results)
    else:
        output = tf.reduce_sum(per_graph_results, axis=-1)  # shape [G]
    model = tf.keras.Model(inputs=model_input, outputs=output)
    model.compile(optimizer=optimizer or get_optimizer(),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])
    return model


def node_multiclass_classifier(input_specs,
                               num_classes: int,
                               optimizer,
                               graph_aggregation_num_heads: int = 16,
                               graph_aggregation_hidden_layers: List[int] = [128],
                               graph_aggregation_dropout_rate: float = 0.2,
                               use_intermediate_gnn_results: bool = False,
                               add_self_loop_edges: bool = True,
                               tie_fwd_bkwd_edges: bool = False,
                               **gnn_kwargs) -> tf.keras.Model:
    # from tensorflow_addons.metrics import F1Score
    model_input = spec_to_input(input_specs)
    gnn_input = as_gnn_inputs(**model_input)
    gnn_outputs = _gnn_outputs(gnn_input,
                               use_intermediate_gnn_results,
                               add_self_loop_edges=add_self_loop_edges,
                               tie_fwd_bkwd_edges=tie_fwd_bkwd_edges,
                               **gnn_kwargs)
    per_node_logits = tf.keras.layers.Dense(num_classes)(gnn_outputs)
    model = tf.keras.Model(inputs=model_input, outputs=per_node_logits)
    model.compile(
        optimizer=optimizer,
        #   loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        loss=ModifiedBinaryCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(threshold=0.0),
            #   F1Score(num_classes=num_classes, threshod=0.0, average='micro')
            ModifiedMicroF1()
        ])
    return model

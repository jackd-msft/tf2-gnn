from typing import Any, Dict, List, Optional, Tuple
import tensorflow as tf

from tf2_gnn import GNNInput, GNN
from tf2_gnn.layers import WeightedSumGraphRepresentation


def as_gnn_inputs(node_features: tf.RaggedTensor, links: tf.Tensor, add_self_loop_edges: bool,
                  tie_fwd_bkwd_edges: bool) -> GNNInput:
    node_features.shape.assert_has_rank(3)
    assert node_features.ragged_rank == 1
    links.shape.assert_has_rank(2)
    assert links.shape[1] == 2
    node_features, node_to_graph_map, num_graphs = tf.keras.layers.Lambda(
        lambda x: (tf.identity(x.values), x.value_rowids(), x.nrows()))(node_features)
    adjacency_lists = [links]
    if add_self_loop_edges:

        def get_self_loops(flat_node_features, dtype):
            num_nodes = tf.shape(flat_node_features, out_type=dtype)[0]
            return tf.tile(tf.expand_dims(tf.range(num_nodes), axis=-1), (1, 2))

        self_loops = tf.keras.layers.Lambda(get_self_loops,
                                            arguments=dict(dtype=links.dtype))(node_features)
        adjacency_lists.append(self_loops)
    if tie_fwd_bkwd_edges:
        adjacency_lists.append(tf.reverse(links, axis=[1]))
    return GNNInput(node_features=node_features,
                    adjacency_lists=tuple(adjacency_lists),
                    node_to_graph_map=node_to_graph_map,
                    num_graphs=num_graphs)


def _spec_to_input(spec: tf.TensorSpec):
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


def _gnn_outputs(gnn_input: GNNInput, use_intermediate_gnn_results=False, **kwargs):
    assert all(n.startswith('gnn_') for n in kwargs)
    kwargs = {k[4:]: v for k, v in kwargs.items()}
    gnn_outputs = GNN(**kwargs)(gnn_input, return_all_representations=use_intermediate_gnn_results)
    return gnn_outputs


def _graph_outputs(model_input,
                   graph_aggregation_num_heads: int,
                   graph_aggregation_hidden_layers: List[int],
                   graph_aggregation_dropout_rate: float,
                   use_intermediate_gnn_results: bool,
                   add_self_loop_edges: bool = True,
                   tie_fwd_bkwd_edges: bool = True,
                   **gnn_kwargs):
    gnn_input = as_gnn_inputs(**model_input,
                              add_self_loop_edges=add_self_loop_edges,
                              tie_fwd_bkwd_edges=tie_fwd_bkwd_edges)
    gnn_outputs = _gnn_outputs(gnn_input, use_intermediate_gnn_results, **gnn_kwargs)
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
                            metadata: Dict[str, Any] = {},
                            optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
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
    model.compile(optimizer=get_optimizer() if optimizer is None else optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.0)])
    return model


def graph_regressor(input_specs,
                    metadata: Dict[str, Any] = {},
                    optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
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
        **gnn_kwargs)

    output = tf.reduce_sum(per_graph_results, axis=-1)  # shape [G]
    model = tf.keras.Model(inputs=model_input, outputs=output)
    model.compile(optimizer=get_optimizer() if optimizer is None else optimizer,
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])
    return model


def node_multiclass_classifier(input_specs,
                               metadata: Dict[str, Any] = {},
                               optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
                               graph_aggregation_num_heads: int = 16,
                               graph_aggregation_hidden_layers: List[int] = [128],
                               graph_aggregation_dropout_rate: float = 0.2,
                               use_intermediate_gnn_results: bool = False,
                               add_self_loop_edges: bool = True,
                               tie_fwd_bkwd_edges: bool = False,
                               **gnn_kwargs) -> tf.keras.Model:
    from tensorflow_addons.metrics import F1Score
    num_classes = metadata.get('num_classes')
    if num_classes is None:
        raise ValueError('num_classes must be a key of metadata')
    model_input = spec_to_input(input_specs)
    gnn_input = as_gnn_inputs(**model_input,
                              add_self_loop_edges=add_self_loop_edges,
                              tie_fwd_bkwd_edges=tie_fwd_bkwd_edges)
    gnn_outputs = _gnn_outputs(gnn_input, use_intermediate_gnn_results, **gnn_kwargs)
    per_node_logits = tf.keras.layers.Dense(num_classes)(gnn_outputs)
    model = tf.keras.Model(inputs=model_input, outputs=per_node_logits)
    model.compile(optimizer=get_optimizer() if optimizer is None else optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[
                      tf.keras.metrics.BinaryAccuracy(threshold=0.0),
                      F1Score(num_classes=num_classes, threshod=0.0)
                  ])
    return model

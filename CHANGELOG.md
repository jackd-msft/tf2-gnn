* Added functional `keras.Model` implementation with consistent losses / metrics.
* Added `tensorflow-datasets` builders and batching functions.
* Added `GraphGlobalExchange.reset_recurrent_state` which redirects to `tf.keras.layers.GRUCell.reset_dropout_mask` where relevant.
* Changed `training: bool = False` -> `training: Optional[bool] = None`. This usage with keras models easier, since `None` signals the implementation to look at `tf.keras.backend.learning_phase()`
    - `dpu_utils` does not do this, so we manually call `learning_phase()` when using `MLP`s.

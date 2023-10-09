
�root"_tf_keras_model*�{"name": "model1_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Model1", "config": {"name": "model1_6", "trainable": true, "dtype": "float32", "input_sizes": {"group_feature_size": 463, "technique_feature_size": 837}, "config": {"group_nn_hidden_layer_widths": 32, "group_nn_hidden_layer_depth": 3, "technique_nn_hidden_layer_widths": 62, "technique_nn_hidden_layer_depth": 3, "nn_output_size": 16, "regularizer": null, "regularizer_weight": null, "initializer": null, "dropout_rate": 0.3}}, "shared_object_id": 0, "build_input_shape": {"input_Group": {"class_name": "TensorShape", "items": [null, 463]}, "input_Technique": {"class_name": "TensorShape", "items": [null, 837]}}, "is_graph_network": false, "full_save_spec": {"class_name": "__tuple__", "items": [[{"input_Group": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 463]}, "float32", "input_Group"]}, "input_Technique": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 837]}, "float32", "input_Technique"]}}], {}]}, "save_spec": {"input_Group": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 463]}, "float32", "input_Group"]}, "input_Technique": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 837]}, "float32", "input_Technique"]}}, "keras_version": "2.12.0", "backend": "tensorflow", "model_config": {"class_name": "Model1", "config": {"name": "model1_6", "trainable": true, "dtype": "float32", "input_sizes": {"group_feature_size": 463, "technique_feature_size": 837}, "config": {"group_nn_hidden_layer_widths": 32, "group_nn_hidden_layer_depth": 3, "technique_nn_hidden_layer_widths": 62, "technique_nn_hidden_layer_depth": 3, "nn_output_size": 16, "regularizer": null, "regularizer_weight": null, "initializer": null, "dropout_rate": 0.3}}}, "training_config": {"loss": {"class_name": "BinaryCrossentropy", "config": {"reduction": "auto", "name": "binary_crossentropy", "from_logits": true, "label_smoothing": 0.0, "axis": -1}, "shared_object_id": 1}, "metrics": [[{"class_name": "AUC", "config": {"name": "auc-pr", "dtype": "float32", "num_thresholds": 200, "curve": "PR", "summation_method": "interpolation", "multi_label": false, "num_labels": null, "label_weights": null, "from_logits": true}, "shared_object_id": 2}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Custom>Adam", "config": {"name": "Adam", "weight_decay": null, "clipnorm": null, "global_clipnorm": null, "clipvalue": null, "use_ema": false, "ema_momentum": 0.99, "ema_overwrite_frequency": null, "jit_compile": false, "is_legacy_optimizer": false, "learning_rate": 9.999999974752427e-07, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}2
�1
�1	root.Technique_NN"_tf_keras_sequential*�1{"name": "Technique_NN", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": false, "class_name": "customNN", "config": {"name": "Technique_NN", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 837]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_63_input"}}, {"class_name": "Dense", "config": {"name": "dense_63", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 837]}, "units": 62, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": 13}}, {"class_name": "Dense", "config": {"name": "dense_64", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": 13}}, {"class_name": "Dense", "config": {"name": "dense_65", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": 13}}, {"class_name": "Dense", "config": {"name": "output_NN", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 37, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 837]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 837]}, "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 837]}, "float32", "dense_63_input"]}], {}]}, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 837]}, "float32", "dense_63_input"]}, "keras_version": "2.12.0", "backend": "tensorflow", "model_config": {"class_name": "customNN", "config": {"name": "Technique_NN", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 837]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_63_input"}, "shared_object_id": 21}, {"class_name": "Dense", "config": {"name": "dense_63", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 837]}, "units": 62, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 24}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": 13}, "shared_object_id": 25}, {"class_name": "Dense", "config": {"name": "dense_64", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 26}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 28}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": 13}, "shared_object_id": 29}, {"class_name": "Dense", "config": {"name": "dense_65", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 30}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 31}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 32}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": 13}, "shared_object_id": 33}, {"class_name": "Dense", "config": {"name": "output_NN", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 34}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 36}]}}}2
�
root.dot_product"_tf_keras_layer*�{"name": "dot_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dot", "config": {"name": "dot_6", "trainable": true, "dtype": "float32", "axes": 1, "normalize": false}, "shared_object_id": 39, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 16]}, {"class_name": "TensorShape", "items": [null, 16]}]}2
�	*"root.Group_NN.layer_with_weights-0"_tf_keras_layer*�{"name": "dense_60", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 463]}, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense_60", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 463]}, "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 463}}, "shared_object_id": 40}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 463]}}2
�+root.Group_NN.layer-1"_tf_keras_layer*�{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": 13}, "shared_object_id": 7, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}2
�,"root.Group_NN.layer_with_weights-1"_tf_keras_layer*�{"name": "dense_61", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense_61", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 41}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}2
�-root.Group_NN.layer-3"_tf_keras_layer*�{"name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": 13}, "shared_object_id": 11, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}2
�."root.Group_NN.layer_with_weights-2"_tf_keras_layer*�{"name": "dense_62", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense_62", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 42}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}2
�/root.Group_NN.layer-5"_tf_keras_layer*�{"name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": 13}, "shared_object_id": 15, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}2
�0"root.Group_NN.layer_with_weights-3"_tf_keras_layer*�{"name": "output_NN", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "output_NN", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 43}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}2
�	7&root.Technique_NN.layer_with_weights-0"_tf_keras_layer*�{"name": "dense_63", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 837]}, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense_63", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 837]}, "units": 62, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 24, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 837}}, "shared_object_id": 44}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 837]}}2
�8root.Technique_NN.layer-1"_tf_keras_layer*�{"name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": 13}, "shared_object_id": 25, "build_input_shape": {"class_name": "TensorShape", "items": [null, 62]}}2
�9&root.Technique_NN.layer_with_weights-1"_tf_keras_layer*�{"name": "dense_64", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense_64", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 26}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 28, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 62}}, "shared_object_id": 45}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 62]}}2
�:root.Technique_NN.layer-3"_tf_keras_layer*�{"name": "dropout_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": 13}, "shared_object_id": 29, "build_input_shape": {"class_name": "TensorShape", "items": [null, 62]}}2
�;&root.Technique_NN.layer_with_weights-2"_tf_keras_layer*�{"name": "dense_65", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense_65", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 30}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 31}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 32, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 62}}, "shared_object_id": 46}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 62]}}2
�<root.Technique_NN.layer-5"_tf_keras_layer*�{"name": "dropout_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": 13}, "shared_object_id": 33, "build_input_shape": {"class_name": "TensorShape", "items": [null, 62]}}2
�=&root.Technique_NN.layer_with_weights-3"_tf_keras_layer*�{"name": "output_NN", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "output_NN", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 34}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 36, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 62}}, "shared_object_id": 47}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 62]}}2
�Rroot.keras_api.metrics.0"_tf_keras_metric*�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 48}2
�Sroot.keras_api.metrics.1"_tf_keras_metric*�{"class_name": "AUC", "name": "auc-pr", "dtype": "float32", "config": {"name": "auc-pr", "dtype": "float32", "num_thresholds": 200, "curve": "PR", "summation_method": "interpolation", "multi_label": false, "num_labels": null, "label_weights": null, "from_logits": true}, "shared_object_id": 2}2
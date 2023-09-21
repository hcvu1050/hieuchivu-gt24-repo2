import tensorflow as tf
from keras.losses import BinaryCrossentropy
from tensorflow import keras

class customNN(keras.Sequential):
    def __init__(self, 
                 regularizer: str|None, regularizer_weight: int|None,
                 name,
                 input_size, 
                 output_size,
                 widths: int|list,
                 depth: int,
                 hidden_layer_activation = 'relu',
                 output_layer_activation = 'linear',
                 ):
        super().__init__(name = name)
        """
        when there is a single int for `widths`, all the Dense layers are identical in size
        `widths` only indicates the widths in the middle layer, NOT the last layer. 
        The last layer's widths is indicated by `output_size`
        """
        if isinstance (widths, int):
            # First dense layer defined with input shape
            self.add(keras.layers.Dense(widths, input_shape=(input_size,)))
            # Custom layer of hidden Dense layer
            for _ in range (depth-2):
                self.add (keras.layers.Dense(widths,activation= hidden_layer_activation))
            # Output layer
            self.add (keras.layers.Dense (output_size, activation = output_layer_activation, name = 'output_NN'))  
        elif isinstance (widths, list) and len(widths) == (depth-2):
            self.add(keras.layers.Dense(widths[0], input_shape=(input_size,)))
            for width in widths[1:]:
                self.add (keras.layers.Dense(width,activation= hidden_layer_activation))
            self.add (keras.layers.Dense (output_size, activation = output_layer_activation, name = 'output_NN'))  
        else: raise Exception ("CustomNN: widths and depths are set incorrectly.\n Most likely becase: widths is a list, and len(width) == (depth-2) is False")
        
        if regularizer == 'l2':
            for layer in self.layers:
                layer.kernel_regularizer = tf.keras.regularizers.l2(regularizer_weight)


class Model1(keras.Model):
    def __init__(self, input_sizes, name = None,
                 group_nn_widths = None, group_nn_depth = None, 
                 technique_nn_widths = None, technique_nn_depth = None,
                 nn_output_size = None, config = None,
                 *args, **kwargs):
        super().__init__(name = name, *args, **kwargs)
        
        if config != None:
            print ('---model built from config')
            group_nn_widths = config['group_nn_widths']
            group_nn_depth = config['group_nn_depth']
            technique_nn_widths = config['technique_nn_widths']
            technique_nn_depth = config['technique_nn_depth']
            nn_output_size = config['nn_output_size']
            regularizer = config['regularizer']
            regularizer_weight = config['regularizer_weight']
            
        group_input_size = input_sizes['group_feature_size']
        technique_input_size = input_sizes['technique_feature_size']
        
        self.input_Group = keras.layers.Input (shape= (group_input_size,), name = 'input_Group')
        self.input_Technique = keras.layers.Input (shape= (technique_input_size,), name = 'input_Technique')
        self.Group_NN = customNN(input_size =  group_input_size,
                                 output_size = nn_output_size,
                                 widths = group_nn_widths,
                                 depth =group_nn_depth,
                                 name = 'Group_NN', 
                                 regularizer= regularizer, regularizer_weight= regularizer_weight)
        self.Technique_NN = customNN(input_size = technique_input_size,
                                 output_size = nn_output_size,
                                 widths = technique_nn_widths,
                                 depth = technique_nn_depth,
                                 name = 'Technique_NN', 
                                 regularizer=regularizer, regularizer_weight= regularizer_weight)
        
        self.dot_product = keras.layers.Dot(axes= 1)
    
    def call(self, inputs):
        # self.input_NN_1, self.input_NN_2 = inputs[0], inputs[1]
        self.input_Group = inputs['input_Group']
        self.input_Technique = inputs['input_Technique']
        
        output_Group = self.Group_NN(self.input_Group)
        output_Technique = self.Technique_NN(self.input_Technique)
        
        norm_output_Group = tf.linalg.l2_normalize (output_Group, axis = 1)
        norm_output_Technique = tf.linalg.l2_normalize (output_Technique, axis = 1)
        
        dot_product = self.dot_product ([norm_output_Group, norm_output_Technique])
        return dot_product

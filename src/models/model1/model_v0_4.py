import tensorflow as tf
from keras.losses import BinaryCrossentropy
from tensorflow import keras

class customNN(keras.Sequential):
    def __init__(self, 
                 regularizer: str|None, regularizer_weight: int|None,
                 name,
                 input_size, 
                 output_size,
                 hidden_layer_widths: int|list,
                 hidden_layer_depth: int| None,
                 hidden_layer_activation = 'relu',
                 output_layer_activation = 'linear',
                 ):
        super().__init__(name = name)
        """
        when `widths` is an `int`, all the Dense layers are identical in size
        `widths` only indicates the widths in the middle layer, NOT the last layer. 
        The last layer's widths is indicated by `output_size`
        """
        if isinstance (hidden_layer_widths, int) and isinstance (hidden_layer_depth, int):
            # First dense layer defined with input shape
            self.add(keras.layers.Dense(hidden_layer_widths, input_shape=(input_size,)))
            # Custom layer of hidden Dense layer
            for _ in range (hidden_layer_depth-1):
                self.add (keras.layers.Dense(hidden_layer_widths,activation= hidden_layer_activation))
            # Output layer
            self.add (keras.layers.Dense (output_size, activation = output_layer_activation, name = 'output_NN'))  
        
        elif isinstance (hidden_layer_widths, list) and hidden_layer_depth == None:
            self.add(keras.layers.Dense(hidden_layer_widths[0], input_shape=(input_size,)))
            for width in hidden_layer_widths[1:]:
                self.add (keras.layers.Dense(width,activation= hidden_layer_activation))
            self.add (keras.layers.Dense (output_size, activation = output_layer_activation, name = 'output_NN'))  
        else: raise Exception ("CustomNN: widths and depths are set incorrectly.\n Correct cases: (widths is int and depth is int) OR (widths is list and depth is None)")
        
        if regularizer == 'l2':
            for layer in self.layers:
                layer.kernel_regularizer = tf.keras.regularizers.l2(l2=regularizer_weight)


class Model1(keras.Model):
    def __init__(self, input_sizes, name = None,
                 group_nn_hidden_layer_widths = None, group_nn_hidden_layer_depth = None, 
                 technique_nn_hidden_layer_widths = None, technique_nn_hidden_layer_depth = None,
                 nn_output_size = None, config = None,
                 *args, **kwargs):
        super().__init__(name = name, *args, **kwargs)
        
        if config != None:
            print ('---model built from config')
            group_nn_hidden_layer_widths = config['group_nn_hidden_layer_widths']
            group_nn_hidden_layer_depth = config['group_nn_hidden_layer_depth']
            technique_nn_hidden_layer_widths = config['technique_nn_hidden_layer_widths']
            technique_nn_hidden_layer_depth = config['technique_nn_hidden_layer_depth']
            nn_output_size = config['nn_output_size']
            regularizer = config['regularizer']
            regularizer_weight = config['regularizer_weight']
            if regularizer_weight != None: regularizer_weight = float (regularizer_weight)
            
        group_input_size = input_sizes['group_feature_size']
        technique_input_size = input_sizes['technique_feature_size']
        
        self.input_Group = keras.layers.Input (shape= (group_input_size,), name = 'input_Group')
        self.input_Technique = keras.layers.Input (shape= (technique_input_size,), name = 'input_Technique')
        self.Group_NN = customNN(input_size =  group_input_size,
                                 output_size = nn_output_size,
                                 hidden_layer_widths = group_nn_hidden_layer_widths,
                                 hidden_layer_depth =group_nn_hidden_layer_depth,
                                 name = 'Group_NN', 
                                 regularizer= regularizer, regularizer_weight= regularizer_weight)
        self.Technique_NN = customNN(input_size = technique_input_size,
                                 output_size = nn_output_size,
                                 hidden_layer_widths = technique_nn_hidden_layer_widths,
                                 hidden_layer_depth = technique_nn_hidden_layer_depth,
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

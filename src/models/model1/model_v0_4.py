import tensorflow as tf
from keras.losses import BinaryCrossentropy
from tensorflow import keras

class ContentBasedFiltering(keras.Model):
    def __init__(self, 
                 num_G_features, 
                 num_T_features,
                 Group_NN_width,
                 Group_NN_depth,
                 Technique_NN_width,
                 Technique_NN_depth,
                 name = 'content_based_filtering',
                 **kwargs):
        super().__init__(name = name, **kwargs)
        
        self.seed = 17
        self.set_random_seed()
        
        # input / output config
        self.num_G_features = num_G_features
        self.num_T_features = num_T_features
        self.num_outputs = 32
        
        # neural network config
        self.Group_NN_width = Group_NN_width
        self.Group_NN_depth = Group_NN_depth
        self.Technique_NN_width = Technique_NN_width
        self.Technique_NN_depth = Technique_NN_depth
        
        self.early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Monitor validation loss for early stopping
        patience=32,           # Number of epochs with no improvement before stopping
        restore_best_weights=True  # Restore model weights from the epoch with the best validation loss
        ) 
        
        self.input_Group = tf.keras.layers.Input(shape = (self.num_G_features), name = "input_Group")
        self.input_Technique = tf.keras.layers.Input (shape= (self.num_T_features), name = "input_Technique")
        
        self.Group_NN = self.build_Group_NN ()
        self.vg = self.Group_NN(self.input_Group)
        
        self.Technique_NN = self.build_Technique_NN ()
        self.vt = self.Technique_NN (self.input_Technique)
        
        
        self.output_layer = tf.keras.layers.Dot (axes=1)(inputs= [self.vg, self.vt], )
        
        self.inputs = [self.input_Group, self.input_Technique]
        self.outputs = self.output_layer
        print (self.input_Group.shape)
        print (self.input_Technique.shape)
        self.build ([tuple(self.input_Group.shape), tuple(self.input_Technique.shape)])
    
    
    def set_random_seed(self):
        tf.random.set_seed(self.seed)
        
    def build_hidden_layers(self, width, depth):
        self.set_random_seed()
        layers = []
        for _ in range (depth):
            layers.append(tf.keras.layers.Dense (width, activation = 'relu'))
        return layers
    
    def build_Group_NN(self):
        self.set_random_seed()
        Group_NN = tf.keras.models.Sequential (
            layers= self.build_hidden_layers (width= self.Group_NN_width, depth=self.Group_NN_depth) + 
            [tf.keras.layers.Dense (self.num_outputs, activation='linear', name = 'output_Group')],
            name = 'Group_NN'
        )
        return Group_NN
    
    def build_Technique_NN(self):
        self.set_random_seed()
        Technique_NN = tf.keras.models.Sequential (
            layers= self.build_hidden_layers (width= self.Technique_NN_width, depth=self.Technique_NN_depth) + 
            [tf.keras.layers.Dense (self.num_outputs, activation='linear', name = 'output_Tecnique')],
            name = 'Technique_NN'
        )
        return Technique_NN
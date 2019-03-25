'''
Relational Reasoning Network for Keras.

This model is specifically designed to conduct visual question-answering with a 
particular focus on rational reasoning questions.  It is based on the "simple neural 
network module for relational reasoning" paper by Deepmind.  However, it loosely 
incorporates elements from Deepmind's related paper on "Visual Interaction Networks".


# References:
    https://arxiv.org/pdf/1706.01427v1.pdf
    https://arxiv.org/pdf/1706.01433.pdf
'''


# Import statements
import sys, warnings
import math, os
import pickle

import numpy as np

import keras
import keras.backend as K

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Embedding, \
                            LSTM, Bidirectional, Lambda, Concatenate, Add
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization, regularizers
from keras.optimizers import Adam, SGD

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../")

from state_of_art_cnns.densenet import densenet  # Requires 'sys.path' call above

from dl_utilities.general import general as gen_utils  # Requires 'sys.path' call above
from dl_utilities.callbacks import callback_utils as cb_utils  # Requires 'sys.path' call above
from dl_utilities.datasets import dataset_utils as ds_utils  # Requires 'sys.path' call above
from dl_utilities.layers import general as dl_layers  # Requires 'sys.path' call above
from dl_utilities.vqa import create_artificial_ds_utils as create_ds_utils  # Requires 'sys.path' call above
from dl_utilities.vqa import cornell_nlvr_utils as cornell_nlvr_utils  # Requires 'sys.path' call above
from dl_utilities.nlp import word_vec_utils as wv_utils  # Requires 'sys.path' call above



# Define global variables
MAX_SPATIAL_DIM_WARNING = 50
L2_NORM_VAL = 1e-4



# Helper function to get list of layers in a simple FF network
def get_layers_for_forward_feed(num_layers, init_MLP_units, 
                                final_MLP_units, id_string):    
                                
    # Get needed variables
    if num_layers == 0:
        raise ValueError("The 'num_layers' parameter should be greater than 0.")
        
    step_val = int(math.floor((final_MLP_units - init_MLP_units) / (num_layers + 1)))
    first_val = init_MLP_units + step_val
    
    
    # Create list with number units in each layer
    layer_vals = []
    
    cur_val = first_val
    for i in range(1, num_layers+1):
        if i == num_layers:
            cur_val = final_MLP_units
        else:
            cur_val += step_val
    
        layer_vals.append(cur_val)    
    
    
    # Initial Dense layer without pre-ReLU (bc it is already there from CNN)
    layers = []

    layers.append(Dense(first_val, name=("ff_%s_dense_%d" % (id_string, 0))))    
    
    # Loop for number of desired layers
    for i, tmp_layers in enumerate(layer_vals):    
        if i % 2 == 0:
            layers.append(BatchNormalization(
                            name=("ff_%s_bn_%d" % (id_string, i+1))))
                            
        layers.append(Activation('relu'))
        layers.append(Dense(tmp_layers, 
                        name=("ff_%s_dense_%d" % (id_string, i+1))))


    # Return layers as list (for fuureu execution)            
    return layers

    
# Helper function to execute a list of layers
def ff_layer_executor(x, ff_layers):
    for cur_layer in ff_layers:
        x = cur_layer(x)
    
    return x    
    
    
    
# Main Relational Reasoning Network model
def RRN(final_img_dims, q_input_len, output_size,
        use_position_channels_with_img=True,
        dn_parms={'block_sizes': [12, 20, 28, 20, 10],
                    'growth_rate': 8},
        RNN_parms=None, 
        relational_MLP_parms={'layers': 5, 'units': 256}, 
        self_analysis_MLP_num_layers=0,
        final_MLP_parms={'layers': 3, 'units': 192}, 
        dropout_rate=0.25):
        
    '''
        Instaniate the Relational Reasoning Network architecture with options for 
        specifying MLP sizes, including a self-analysis MLP, changing dropout rate, 
        type of question vector (e.g. either a sentence or pre-computed vector 
        representing a question), and image processing CNN size.
        
        Parameters:
            final_img_dims: 2-D tuple for the input image shape 
                -should be formatted (width, height)
            q_input_len: the length of the input question 
                -should be a single integer representing 
                    either number of words in the question or 
                    dimensions of the question embedding
            use_position_channels_with_img: flag for augmenting RBG color channels 
                -with 2 position channels (one for the x-axis and one for y-axis)
            dn_parms: dictionary with key parameters for the DenseNet model
                -'block_sizes': list of number of layers in each "block" of DenseNet 
                -'growth_rate': value for increase of size of  
                -Recommended that there are at enough blocks (e.g. 4 or 5) so 
                    that the spatial area (width x height) of thee DenseNet 
                    embeddings is at most 50 units
            RNN_parms: dictionary used only if question input requires an RNN
                -'word_embedding_dim': desired size for word embeddings
                -'rnn_output_size': size of RNN output embedding (representing word question)
                -'word_index': parameter with dictionary mapping of words to encoded indices
                -'use_pretrained_embeddings': whether to use pre-trained word embeddings
            relational_MLP_parms: dictionary for parameters of relational MLP
                -'num_layers': number of layers in MLP
                -'units': output embedding size of each layer
            self_analysis_MLP_num_layers:  number of layers in the self-analysis MLP
                -not included by default (e.g. set to 0)
            final_MLP_parms:  dictionary for parameters of final MLP
                -'num_layers':  number of layers in MLP
                -'units': output embedding size of each layer
            dropout_rate: the rate of dropout to apply to the layers in the final_MLP
    '''
    
    # Define inputs
    final_img_dims += (3,)
        
    image_input = Input(final_img_dims)
    question_input = Input((q_input_len,))
    

    # August images with position
    final_image_input = image_input
    if use_position_channels_with_img:
        final_image_input = dl_layers.AddPositionalChannels()(final_image_input)
        final_img_dims = final_img_dims[:2] + (5,)
    

    # Define CNN model and get image features
    dn_block_sizes = dn_parms['block_sizes']
    dn_growth_rate = dn_parms['growth_rate']
    
    if type(dn_block_sizes) is not list:
        raise ValueError("The 'dn_block_sizes' variable should be a list of integers.")
        
    dn_num_blocks = len(dn_block_sizes)
    dn_depth = int(np.sum(np.array(dn_block_sizes))) + dn_num_blocks + 1
    
    cnn_model = densenet.DenseNet(final_img_dims, depth=dn_depth, 
                                    nb_dense_block=dn_num_blocks, 
                                    nb_layers_per_block=dn_block_sizes,
                                    bottleneck=False, reduction=0.5,        # No bottle to improve gradient flow 
                                    growth_rate=dn_growth_rate, 
                                    final_dropout=0.0, include_top=False)                                        
    
    cnn_features = cnn_model(final_image_input)

    
    # Get question embedding using an RNN
    q_embedding = question_input
    if RNN_parms is not None:
        embedding_dim = RNN_parms['word_embedding_dim']
        word_index = RNN_parms['word_index']
        
        # Get word embeddings 
        if RNN_parms['use_pretrained_embeddings']:
            embedding_layer = wv_utils.embed_layer_with_pretrained_word_vecs(word_index, 
                                                                embedding_dim, q_input_len)
        else:
            num_words = len(word_index) + 1
            embedding_layer = Embedding(num_words, embedding_dim, input_length=q_input_len)
        
        q_embedding = embedding_layer(q_embedding)
        
        # Pass them thru RNN
        rnn_units = RNN_parms['rnn_output_size']
        rnn_model = LSTM(rnn_units, implementation=2, return_sequences=False,
                                        recurrent_regularizer=regularizers.l2(L2_NORM_VAL), 
                                        recurrent_dropout=0.25)
        
        bi_rnn_model = Bidirectional(rnn_model)
        q_embedding = bi_rnn_model(q_embedding)
    
    else:
        image_embed_size = int(K.int_shape(cnn_features)[-1])
        q_scale_factor = 1.0
        q_num_layers = 3
        
        init_embed_size = K.int_shape(q_embedding)[-1]
        ideal_q_embed_size = int(image_embed_size * q_scale_factor)
        
        q_layers = get_layers_for_forward_feed(q_num_layers, init_embed_size, 
                                                ideal_q_embed_size, "q_embed")
        q_embedding = ff_layer_executor(q_embedding, q_layers)
        
        
    # Get combination of spatial features with question embedding
    shapes = K.int_shape(cnn_features)    
    h, w = shapes[1], shapes[2]      
    num_cnn_channels = shapes[-1]      
    
    combined_dims = h * w
    if combined_dims > MAX_SPATIAL_DIM_WARNING:
        warnings.warn("By using more than %d spatial embeddings, training will take a very long time "
                                "and may not even possible depending on your system specs." % MAX_SPATIAL_DIM_WARNING, 
                                RuntimeWarning)


    features = []
    for h_val in range(h):
        for w_val in range(w):
            features.append(
                    dl_layers.GetSpecificSpatialFeatures(h_val, w_val)(cnn_features))
                    
    relations = []
    for i, feature1 in enumerate(features):
        for feature2 in features[i+1:]:
            relations.append(Concatenate()([feature1, feature2, q_embedding]))
            
        
    # Pass relationships through simple forward feed network    
    num_layers = relational_MLP_parms['num_layers']
    hidden_units = relational_MLP_parms['units']
    init_units = (2 * num_cnn_channels) + K.int_shape(q_embedding)[-1]
    
    rel_layers = get_layers_for_forward_feed(num_layers, init_units, hidden_units, "rel")

    processed_relations = []
    for cur_relation in relations:
        processed_relations.append(ff_layer_executor(cur_relation, rel_layers))    

        
    # Incorporate self-analysis if desired        
    if self_analysis_MLP_num_layers > 0:
        relations = []
        for cur_feature in enumerate(features):        
            relations.append(Concatenate()([cur_feature, q_embedding]))
        
        init_units -= num_cnn_channels
        self_analysis_layers = get_layers_for_forward_feed(self_analysis_MLP_num_layers, 
                                                        init_units, hidden_units, "self")
        
        for cur_relation in relations:
            processed_relations.append(ff_layer_executor(cur_relation, self_analysis_layers))    
        
        
    # Sum them up and then pass through a couple more layers
    x = Add()(processed_relations)
    
    num_final_iters = final_MLP_parms['num_layers']
    final_MLP_units = final_MLP_parms['units']

    for i in range(num_final_iters):
        if i % 2 == 0:
            x = BatchNormalization()(x)
            
        x = Dense(final_MLP_units)(x)
        x = Activation('relu')(x)
        
        if dropout_rate > 0.0:
            x = Dropout(dropout_rate)(x)

    pred = Dense(output_size, activation='softmax')(x)

    
    # Build and return model
    model = Model(inputs=[image_input, question_input], outputs=pred)
    return model
    
    
    
    
    
    


#############   MAIN ROUTINE   #############	

if __name__ == '__main__':
    
    # Set desired type of training
    use_simple_dataset = False
        
    
    if use_simple_dataset:
        # Dataset variables
        image_side_size = 64
        shape_size = 5
        
        final_img_dims = (image_side_size, image_side_size)
        ques_embedding_size = create_ds_utils.question_dim
        
        train_set_size = 20000
        test_set_size = 2000
        num_q_per_image = 5
        
        cnn_block_sizes = [4, 6, 8, 5, 3]
        dn_growth_rate = 14
        
        rel_MLP_layers = 5
        rel_MLP_units = 256

        final_MLP_layers = 5
        final_MLP_units = 192
        final_MLP_dropout = 0.3
        
        
        # Get training and test data
        train_data, test_data = create_ds_utils.get_simple_vqa_artificial_dataset(
                                                                    train_size=train_set_size, 
                                                                    test_size=test_set_size,
                                                                    num_q_per_image=num_q_per_image, 
                                                                    img_side_size=image_side_size, 
                                                                    half_shape_size=shape_size,
                                                                    force_new=True)
        
        rel_train, non_rel_train = train_data
        train_imgs, train_qs, train_labels = create_ds_utils.reformat_dataset(
                                                            rel_train + non_rel_train)                        
       
        
        rel_test, non_rel_test = test_data
        test_imgs, test_qs, test_labels = create_ds_utils.reformat_dataset(
                                                            rel_test + non_rel_test)                                                

        
        # Perform normalization/standardization to images
        train_imgs, test_imgs = ds_utils.simple_image_preprocess(train_imgs, test_imgs)

        
        # Get Relational Reasoning model
        rr_model = RRN(final_img_dims, ques_embedding_size, 
                            train_labels.shape[-1],
                            use_position_channels_with_img=False,
                            dn_parms=
                                {'block_sizes': cnn_block_sizes,
                                 'growth_rate': dn_growth_rate},
                            relational_MLP_parms=
                                {'num_layers': rel_MLP_layers, 'units': rel_MLP_units}, 
                            self_analysis_MLP_num_layers=0,
                            final_MLP_parms=
                                {'num_layers': final_MLP_layers, 'units': final_MLP_units},
                            dropout_rate=final_MLP_dropout)
        rr_model.summary()
        
        
        # Begin training and evaluation
        epochs = 50
        batch_size = 256
        init_lr_val = 0.2
        
        callbacks = [ cb_utils.CosineLRScheduler(init_lr_val, epochs) ]       
        
        rr_model.compile(optimizer=SGD(lr=init_lr_val), loss='categorical_crossentropy', metrics=['accuracy'])
        rr_model.fit([train_imgs, train_qs], train_labels, 
                        validation_data=[[test_imgs, test_qs], test_labels],
                        epochs=epochs, batch_size=batch_size, callbacks=callbacks)
                        
    else:
        # Dataset variables
        final_img_dims = (32, 128)
        max_seq_len = 32
        
        cnn_block_sizes = [8, 10, 12, 8, 6]
        dn_growth_rate = 12
        
        embedding_dim = 160
        rnn_units = 112       # Should be between 0.5 to 0.75 of the size of CNN output channels

        rel_MLP_layers = 5
        rel_MLP_units = 256

        final_MLP_layers = 6
        final_MLP_units = 192
        final_MLP_dropout = 0.1

        
        # Get training and test data
        word_index, train_components, test_components = (
                                            cornell_nlvr_utils.get_data(final_img_dims, max_seq_len))
        
        train_imgs, train_word_seqs, train_labels = train_components
        test_imgs, test_word_seqs, test_labels = test_components
        
        # Perform normalization/standardization to images
        train_imgs, test_imgs = ds_utils.normal_image_preprocess(train_imgs, test_imgs)

        
        # Get Relational Reasoning model
        rr_model = RRN(final_img_dims, max_seq_len, 
                            train_labels.shape[-1],
                            use_position_channels_with_img=True,
                            dn_parms=
                                {'block_sizes': cnn_block_sizes, 
                                 'growth_rate': dn_growth_rate},
                            RNN_parms=
                                {'word_embedding_dim': embedding_dim, 
                                 'rnn_output_size': rnn_units, 
                                 'word_index': word_index, 
                                 'use_pretrained_embeddings': True},
                            relational_MLP_parms=
                                {'num_layers': rel_MLP_layers, 'units': rel_MLP_units}, 
                            self_analysis_MLP_num_layers=0,
                            final_MLP_parms=
                                {'num_layers': final_MLP_layers, 'units': final_MLP_units},
                            dropout_rate=final_MLP_dropout)
        rr_model.summary()

        
        # Begin training and evaluation
        epochs = 75
        batch_size = 128
        init_lr_val = 0.2
        
        callbacks = [ cb_utils.CosineLRScheduler(init_lr_val, epochs) ]
        multi_input_gen = cornell_nlvr_utils.CornellDataAugmentor(train_imgs, train_word_seqs, 
                                                                    word_index, train_labels, 
                                                                    batch_size=batch_size)
        
        rr_model.compile(optimizer=SGD(lr=init_lr_val), loss='categorical_crossentropy', 
                            metrics=['accuracy'])
        rr_model.fit_generator(multi_input_gen, 
                                validation_data=[[test_imgs, test_word_seqs], test_labels],
                                steps_per_epoch=(train_imgs.shape[0] // batch_size),
                                epochs=epochs, 
                                callbacks=callbacks)
    
    
    
    # Exit successfully
    exit(0)

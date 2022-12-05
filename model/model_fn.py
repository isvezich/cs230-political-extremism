import tensorflow as tf
from keras import layers
from keras.layers import Embedding, Input, Layer, TextVectorization
from keras.models import Model
import string
from nltk.corpus import stopwords
import re
import numpy as np
from sentence_transformers import SentenceTransformer

from keras import backend as K
from keras.losses import BinaryCrossentropy

from model.utils import read_glove_vecs, sentence_to_avg, sentences_to_indices, sentence_to_sbert_avg, \
    sentence_to_sbert_seq


# define evaluation metrics
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# clean up junk from string
def custom_standardization(input_data):
    cachedStopWords = stopwords.words("english")

    lowercase = tf.strings.lower(input_data)
    for word in cachedStopWords:
        lowercase = tf.strings.regex_replace(lowercase, word, '')
    lowercase = tf.strings.regex_replace(lowercase, 'nan', '')
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.

    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """

    vocab_size = len(word_to_index) + 1  # adding 1 to fit Keras embedding (requirement)
    any_word = list(word_to_vec_map.keys())[0]
    emb_dim = word_to_vec_map[any_word].shape[0]  # define dimensionality of your GloVe word vectors (= 50)

    ### START CODE HERE ###
    # Step 1
    # Initialize the embedding matrix as a numpy array of zeros.
    # See instructions above to choose the correct shape.
    emb_matrix = np.zeros((vocab_size, emb_dim))

    # Step 2
    # Set each row "idx" of the embedding matrix to be
    # the word vector representation of the idx'th word of the vocabulary
    for word, idx in word_to_index.items():
        emb_matrix[idx, :] = word_to_vec_map[word]

    # Step 3
    # Define Keras embedding layer with the correct input and output sizes
    # Make it non-trainable.
    embedding_layer = Embedding(vocab_size, emb_dim)
    ### END CODE HERE ###

    # Step 4 (already done for you; please do not modify)
    # Build the embedding layer, it is required before setting the weights of the embedding layer.
    embedding_layer.build((None,))  # Do not modify the "None".  This line of code is complete as-is.

    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


def mlp_model(params, vectorize_layer=None):
    if params.embeddings:
        print('params.embeddings: model fn mlp model - 96')
        inputs = Input(shape=(params.embedding_size,), dtype='float64')
        X_inp = Layer()(inputs)
    else:
        print('no params.embeddings: model fn mlp model - 99')
        inputs = Input(shape=(), dtype='string')
        X_inp = vectorize_layer(inputs)
        X_inp = layers.Embedding(
            input_dim=len(vectorize_layer.get_vocabulary()),
            output_dim=params.embedding_size,
            # Use masking to handle the variable sequence lengths
            mask_zero=True)(X_inp)
        X_inp = layers.GlobalAveragePooling1D()(X_inp)
    X = layers.Dense(params.h1_units,
                           activation='relu',
                           kernel_regularizer=tf.keras.regularizers.L2(params.l2_reg_lambda),
                           kernel_initializer=tf.keras.initializers.HeUniform())(X_inp)
    X = layers.BatchNormalization()(X)
    X = layers.Dense(params.h2_units,
                           activation='relu',
                           kernel_regularizer=tf.keras.regularizers.L2(params.l2_reg_lambda),
                           kernel_initializer=tf.keras.initializers.HeUniform())(X)
    X = layers.BatchNormalization()(X)
    outputs = layers.Dense(1)(X)
    model = Model(inputs, outputs)
    return model


def rnn_model(params, maxLen=None, word_to_vec_map=None, word_to_index=None, vectorize_layer=None):
    if params.embeddings == 'GloVe':
        inputs = Input(shape=(maxLen,), dtype='int32')

        # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
        embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

        # Propagate sentence_indices through your embedding layer
        # (See additional hints in the instructions).
        X_inp = embedding_layer(inputs)
    elif params.embeddings == 'SBERT':
        inputs = Input(shape=(maxLen, params.embedding_size), dtype='float64')
        X_inp = Layer()(inputs)
    else:
        inputs = Input(shape=(), dtype='string')
        X_inp = vectorize_layer(inputs)
        X_inp = layers.Embedding(
            input_dim=len(vectorize_layer.get_vocabulary()),
            output_dim=params.embedding_size,
            # Use masking to handle the variable sequence lengths
            mask_zero=True)(X_inp)
    # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
    X = layers.GRU(params.h1_units, return_sequences=True, dropout=params.dropout_rate, recurrent_regularizer=tf.keras.regularizers.L2(params.l2_reg_lambda))(X_inp)
    # The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
    X = layers.SimpleRNN(params.h2_units, dropout=params.dropout_rate, recurrent_regularizer=tf.keras.regularizers.L2(params.l2_reg_lambda))(X)
    outputs = layers.Dense(1)(X)
    model = Model(inputs, outputs)

    return model


def model_fn(inputs, params, embeddings_path=None):
    """Compute logits of the model (output distribution)

    Args:
        inputs: features & labels
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        embeddings_path: (string) relative path to pre-trained embeddings if they are to be used, else None

    Returns:
        output: (keras.Sequential) Sequential model
    """

    # set up model architecture
    if params.model_version == 'mlp':
        print('model version is mlp: model_fn - 159')
        if params.embeddings == 'GloVe':
            print('glove embeddings: model_fn - 161')
            params.embedding_size = 50
            # inputs['train'][0] = tf.map_fn(sentence_to_avg, inputs['train'][0])
            # inputs['val'][0] = tf.map_fn(sentence_to_avg, inputs['val'][0])
            # inputs['test'][0] = tf.map_fn(sentence_to_avg, inputs['test'][0])
            str_feat_train = []
            str_feat_val = []
            str_feat_test = []
            for i in range(inputs['train'][0].shape[0]):
                str_feat_train.append(sentence_to_avg(inputs['train'][0][i]))
            print('finished sentence_to_avg for train')
            for i in range(inputs['val'][0].shape[0]):
                str_feat_val.append(sentence_to_avg(inputs['val'][0][i]))
            print('finished sentence_to_avg for val')
            for i in range(inputs['test'][0].shape[0]):
                str_feat_test.append(sentence_to_avg(inputs['test'][0][i]))
            print('finished sentence_to_avg for test')
            inputs['train'][0] = tf.cast(tf.stack(str_feat_train), 'float64')
            inputs['val'][0] = tf.cast(tf.stack(str_feat_val), 'float64')
            inputs['test'][0] = tf.cast(tf.stack(str_feat_test), 'float64')
            model = mlp_model(params)
        elif params.embeddings == 'SBERT':
            print('sbert embeddings: model fn - 194')
            params.embedding_size = 384
            # inputs['train'][0] = tf.vectorized_map(sentence_to_sbert_avg, inputs['train'][0])
            # inputs['val'][0] = tf.vectorized_map(sentence_to_sbert_avg, inputs['val'][0])
            # inputs['test'][0] = tf.vectorized_map(sentence_to_sbert_avg, inputs['test'][0])
            str_feat_train = []
            str_feat_val = []
            str_feat_test = []
            for i in range(inputs['train'][0].shape[0]):
                str_feat_train.append(sentence_to_sbert_avg(inputs['train'][0][i]))
            for i in range(inputs['val'][0].shape[0]):
                str_feat_val.append(sentence_to_sbert_avg(inputs['val'][0][i]))
            print('finished sentence embeddings for val')
            for i in range(inputs['test'][0].shape[0]):
                str_feat_test.append(sentence_to_sbert_avg(inputs['test'][0][i]))
            print('finished sentence embeddings for test')
            inputs['train'][0] = tf.cast(tf.stack(str_feat_train), 'float64')
            inputs['val'][0] = tf.cast(tf.stack(str_feat_val), 'float64')
            inputs['test'][0] = tf.cast(tf.stack(str_feat_test), 'float64')
            model = mlp_model(params)
        elif params.embeddings == None:
            print('no embeddings path: model fn - 181')
            # instantiate embedding layer
            vectorize_layer = TextVectorization(
                standardize=custom_standardization,
                max_tokens=params.max_features,
                output_mode='int')
            vectorize_layer.adapt(inputs['train'][0])
            model = mlp_model(params, vectorize_layer=vectorize_layer)
        else:
            raise NotImplementedError("invalid embedding type")
    elif params.model_version == 'rnn':
        if params.embeddings == 'GloVe':
            params.embedding_size = 50
            maxLen = len(max(inputs['train'][0].numpy(), key=len).split())
            words_to_index, index_to_words, word_to_vec_map = read_glove_vecs(embeddings_path)
            inputs['train'][0] = sentences_to_indices(inputs['train'][0], words_to_index, maxLen)
            print('finished sentences_to_indices for train')
            inputs['val'][0] = sentences_to_indices(inputs['val'][0], words_to_index, maxLen)
            print('finished sentences_to_indices for val')
            inputs['test'][0] = sentences_to_indices(inputs['test'][0], words_to_index, maxLen)
            print('finished sentences_to_indices for test')
            model = rnn_model(params, maxLen, word_to_vec_map, words_to_index)
        elif params.embeddings == 'SBERT':
            print('sbert embeddings: model fn - 194')
            params.embedding_size = 384
            # model = SentenceTransformer('all-MiniLM-L6-v2')
            # inputs['train'][0] = tf.vectorized_map(sentence_to_sbert_seq, inputs['train'][0])
            # inputs['val'][0] = tf.vectorized_map(sentence_to_sbert_seq, inputs['val'][0])
            # inputs['test'][0] = tf.vectorized_map(sentence_to_sbert_seq, inputs['test'][0])
            str_feat_train = []
            str_feat_val = []
            str_feat_test = []
            for i in range(inputs['train'][0].shape[0]):
                str_feat_train.append(sentence_to_sbert_seq(inputs['train'][0][i]))
            print('finished sentence embeddings for train')
            for i in range(inputs['val'][0].shape[0]):
                str_feat_val.append(sentence_to_sbert_seq(inputs['val'][0][i]))
            print('finished sentence embeddings for val')
            for i in range(inputs['test'][0].shape[0]):
                str_feat_train.append(sentence_to_sbert_seq(inputs['test'][0][i]))
            print('finished sentence embeddings for test')
            inputs['train'][0] = tf.cast(tf.stack(str_feat_train), 'float64')
            inputs['val'][0] = tf.cast(tf.stack(str_feat_val), 'float64')
            inputs['test'][0] = tf.cast(tf.stack(str_feat_test), 'float64')
            maxLen = len(max(inputs['train'][0].numpy(), key=len).split())
            model = rnn_model(params, maxLen=maxLen)
        elif params.embeddings == None:
            # instantiate embedding layer
            vectorize_layer = TextVectorization(
                standardize=custom_standardization,
                max_tokens=params.max_features,
                output_mode='int')
            vectorize_layer.adapt(inputs['train'][0])
            model = rnn_model(params, vectorize_layer=vectorize_layer)
        else:
            raise NotImplementedError("invalid embedding type")
    else:
        raise NotImplementedError("Unknown model version: {}".format(params.model_version))

    # compile model
    model.compile(loss=BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=params.learning_rate),
                  metrics=[tf.metrics.BinaryAccuracy(threshold=0.0), f1_m, precision_m, recall_m#,
                           ])
                           #tf.metrics.AUC(from_logits=True)])
    print(model.summary())

    return model, inputs

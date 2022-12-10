import tensorflow as tf
from keras import layers
from keras.layers import Embedding, Input, Layer, TextVectorization
from keras.models import Model
import string
from nltk.corpus import stopwords
import re
import numpy as np

from keras import backend as K
from keras.losses import BinaryCrossentropy

from model.sentence_bert_lstm import SentenceBertLSTM
from model.sentence_bert_rnn import SentenceBertRNN
from model.sentence_bert_mlp import SentenceBertMLP
import tensorflow_models as tfm
nlp = tfm.nlp


# define evaluation metrics
def recall_m(y_true, y_pred):
    y_pred = tf.math.sigmoid(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    y_pred = tf.math.sigmoid(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    y_pred = tf.math.sigmoid(y_pred)
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

    vocab_size = len(word_to_index) + 1
    any_word = list(word_to_vec_map.keys())[0]
    emb_dim = word_to_vec_map[any_word].shape[0]

    emb_matrix = np.zeros((vocab_size, emb_dim))

    # Set each row "idx" of the embedding matrix to be
    # the word vector representation of the idx'th word of the vocabulary
    for word, idx in word_to_index.items():
        emb_matrix[idx, :] = word_to_vec_map[word]

    # Define Keras embedding layer with the correct input and output sizes
    # Make it non-trainable.
    embedding_layer = Embedding(vocab_size, emb_dim, trainable=False)

    # Build the embedding layer, it is required before setting the weights of the embedding layer.
    embedding_layer.build((None,))

    # Set the weights of the embedding layer to the embedding matrix
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer

def create_vectorized_layer(words, max_features):
    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode='int')
    vectorize_layer.adapt(words)
    return vectorize_layer


def word_mlp_model(params, vectorize_layer=None):
    if params.embeddings == 'GloVe':
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
    outputs = layers.Dense(1, activation='sigmoid')(X)
    model = Model(inputs, outputs)
    return model


def word_rnn_model(params, word_to_vec_map=None, word_to_index=None, vectorize_layer=None):
    if params.embeddings == 'GloVe':
        inputs = Input(shape=(params.max_word_length,), dtype='int32')

        # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
        embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

        # Propagate sentence_indices through your embedding layer
        # (See additional hints in the instructions).
        X_inp = embedding_layer(inputs)
    else:
        inputs = Input(shape=(), dtype='string')
        X_inp = vectorize_layer(inputs)
        X_inp = layers.Embedding(
            input_dim=len(vectorize_layer.get_vocabulary()),
            output_dim=params.embedding_size,
            # Use masking to handle the variable sequence lengths
            mask_zero=True)(X_inp)
    # The output of RNN will be a 3D tensor of shape (batch_size, timesteps, 64)
    X = layers.SimpleRNN(params.h1_units, recurrent_dropout=params.dropout_rate, recurrent_regularizer=tf.keras.regularizers.L2(params.l2_reg_lambda))(X_inp)
    outputs = layers.Dense(1, activation='sigmoid')(X)
    model = Model(inputs, outputs)

    return model

def word_lstm_model(params, vectorize_layer=None, maxLen=None, word_to_vec_map=None, word_to_index=None):
    if params.embeddings == 'GloVe':
        inputs = Input(shape=(maxLen,), dtype='int32')

        # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
        embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

        # Propagate sentence_indices through your embedding layer
        # (See additional hints in the instructions).
        X_inp = embedding_layer(inputs)
    else:
        inputs = Input(shape=(), dtype='string')
        X_inp = vectorize_layer(inputs)
        X_inp = layers.Embedding(
            input_dim=len(vectorize_layer.get_vocabulary()),
            output_dim=params.embedding_size,
            # Use masking to handle the variable sequence lengths
            mask_zero=True)(X_inp)
    X = layers.LSTM(params.h1_units, return_sequences=True, recurrent_dropout=params.dropout_rate, recurrent_regularizer=tf.keras.regularizers.L2(params.l2_reg_lambda))(X_inp)
    X = layers.LSTM(params.h2_units, recurrent_dropout=params.dropout_rate, recurrent_regularizer=tf.keras.regularizers.L2(params.l2_reg_lambda))(X)
    outputs = layers.Dense(1, activation='sigmoid')(X)
    model = Model(inputs, outputs)

    return model

def bert_to_lstm_model(params):
    inputs = Input(shape=(None, None), dtype='string')
    X = SentenceBertLSTM(params.model_id, params)(inputs)
    outputs = layers.Dense(1, activation='sigmoid')(X)
    print(f"output shape: {X.shape}")
    model = Model(inputs, outputs)

    return model

def bert_to_rnn_model(params):
    inputs = Input(shape=(None, ), dtype='string')
    X = SentenceBertRNN(params.model_id, params)(inputs)
    outputs = layers.Dense(1, activation='sigmoid')(X)
    print(f"output shape: {X.shape}")
    model = Model(inputs, outputs)

    return model

def bert_to_mlp_model(params):
    inputs = Input(shape=(None, ), dtype='string')
    X = SentenceBertMLP(params.model_id, params)(inputs)
    X = tf.keras.layers.Dense(params.h1_units,
                          activation='relu',
                          kernel_regularizer=tf.keras.regularizers.L2(params.l2_reg_lambda),
                          kernel_initializer=tf.keras.initializers.HeUniform())(X)
    X = layers.BatchNormalization()(X)
    X = layers.Dense(params.h2_units,
                           activation='relu',
                           kernel_regularizer=tf.keras.regularizers.L2(params.l2_reg_lambda),
                           kernel_initializer=tf.keras.initializers.HeUniform())(X)
    X = layers.BatchNormalization()(X)
    outputs = layers.Dense(1, activation='sigmoid')(X)
    print(f"output shape: {X.shape}")
    model = Model(inputs, outputs)

    return model

def model_fn(inputs, params):
    """Compute logits of the model (output distribution)

    Args:
        inputs: features & labels
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        output: (keras.Sequential) Sequential model
    """

    # set up model architecture


    if params.model_version == 'mlp':
        print('model version is mlp: model_fn - 159')
        if params.embeddings == 'GloVe':
            # Force glove embedding size to be 50
            params.embedding_size = 50
            model = word_mlp_model(params)
        elif params.embeddings == 'None':
            # instantiate embedding layer
            vectorize_layer = create_vectorized_layer(inputs['train'][0], params.max_features)
            model = word_mlp_model(params, vectorize_layer=vectorize_layer)
        else:
            raise NotImplementedError("invalid embedding type")
    elif params.model_version == 'rnn':
        if params.embeddings == 'GloVe':
            # Force glove embedding size to be 50
            params.embedding_size = 50
            model = word_rnn_model(params, inputs['word_to_vec_map'], inputs['words_to_index'])
        elif params.embeddings == 'None':
            # instantiate embedding layer
            vectorize_layer = create_vectorized_layer(inputs['train'][0], params.max_features)
            model = word_rnn_model(params, vectorize_layer=vectorize_layer)
        else:
            raise NotImplementedError("invalid embedding type")
    elif params.model_version == 'lstm':
        if params.embeddings == 'GloVe':
            model = word_lstm_model(params, inputs['word_to_vec_map'], inputs['words_to_index'])
        elif params.embeddings == 'None':
            # instantiate embedding layer
            vectorize_layer = create_vectorized_layer(inputs['train'][0], params.max_features)
            model = word_lstm_model(params, vectorize_layer=vectorize_layer)
        else:
            raise NotImplementedError("invalid embedding type")
    elif params.model_version == 'BERT_LSTM':
        model = bert_to_lstm_model(params)
    elif params.model_version == 'BERT_RNN':
        model = bert_to_rnn_model(params)
    elif params.model_version == 'BERT_MLP':
        model = bert_to_mlp_model(params)
    else:
        raise NotImplementedError("Unknown model version: {}".format(params.model_version))

    # compile model
    model.compile(loss=BinaryCrossentropy(from_logits=False),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=params.learning_rate, clipnorm=1.0),
                  metrics=[tf.metrics.BinaryAccuracy(),
                           f1_m,
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall(),
                           ])
                           #tf.metrics.AUC()])
    print(model.summary())

    return model, inputs
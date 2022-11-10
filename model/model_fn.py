import tensorflow as tf
from keras import layers

from keras import backend as K
from keras.losses import BinaryCrossentropy


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


def mlp_model(params):
    model = tf.keras.Sequential([
        layers.Embedding(params.max_features + 1, params.embedding_size),
        layers.GlobalAveragePooling1D(),
        layers.Dense(params.h1_units,
                     activation='relu',
                     kernel_regularizer=tf.keras.regularizers.L2(params.l2_reg_lambda),
                     kernel_initializer=tf.keras.initializers.HeUniform()),
        layers.BatchNormalization(),
        layers.Dense(params.h2_units,
                     activation='relu',
                     kernel_regularizer=tf.keras.regularizers.L2(params.l2_reg_lambda),
                     kernel_initializer=tf.keras.initializers.HeUniform()),
        layers.BatchNormalization(),
        layers.Dense(1)])
    return model


def rnn_model(params):
    model = tf.keras.Sequential()
    model.add(layers.Embedding(params.max_features + 1, params.embedding_size))

    # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
    model.add(layers.GRU(params.h1_units, return_sequences=True))

    # The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
    model.add(layers.SimpleRNN(params.h2_units))

    model.add(layers.Dense(1))

    return model


def model_fn(params):
    """Compute logits of the model (output distribution)

    Args:
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        output: (tf.Tensor) output of the model
    """

    # set up model architecture
    if params.model_version == 'mlp':
        model = mlp_model(params)
    elif params.model_version == 'rnn':
        model = rnn_model(params)
    else:
        raise NotImplementedError("Unknown model version: {}".format(params.model_version))

    # compile model
    model.compile(loss=BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=params.learning_rate),
                  metrics=[tf.metrics.BinaryAccuracy(threshold=0.0), f1_m, precision_m, recall_m,
                           tf.metrics.AUC(from_logits=True)])

    return model

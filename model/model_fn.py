import tensorflow as tf
from keras import layers

from keras import backend as K
from keras.losses import BinaryCrossentropy

from model.input_fn import custom_standardization


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


def mlp_model(params, vocab):
    inputs = {
        "words": tf.keras.Input(shape=(), dtype="string"),
        "score": tf.keras.Input(shape=(), dtype="float64"),
        "num_replies": tf.keras.Input(shape=(), dtype="float64")
    }

    text_vectorizer = layers.TextVectorization(
        standardize=custom_standardization,
        split='whitespace',
        output_mode='int',
        vocabulary=vocab
    )

    vocab_size=len(vocab)

    # text_vectorizer.adapt(inputs["words"])
    # vocab_size = text_vectorizer.vocabulary_size()

    # instantiate embedding layer
    words_vector = text_vectorizer(inputs["words"])
    words_embedding = layers.Embedding(vocab_size, params.embedding_size)(words_vector)
    words_pooled = layers.GlobalAveragePooling1D()(words_embedding)
    print(f"words_pooled: {words_pooled.shape}")

    outputs = tf.keras.layers.Concatenate()([
        words_pooled,
        tf.expand_dims(inputs["score"], -1),
        tf.expand_dims(inputs["num_replies"], -1)
    ])

    # outputs = layers.BatchNormalization()(outputs)
    outputs = layers.Dense(params.h1_units,
                           activation='relu',
                           kernel_regularizer=tf.keras.regularizers.L2(params.l2_reg_lambda),
                           kernel_initializer=tf.keras.initializers.HeUniform())(outputs)
    # outputs = layers.BatchNormalization()(outputs)
    outputs = layers.Dense(params.h2_units,
                           activation='relu',
                           kernel_regularizer=tf.keras.regularizers.L2(params.l2_reg_lambda),
                           kernel_initializer=tf.keras.initializers.HeUniform())(outputs)
    # outputs = layers.BatchNormalization()(outputs)
    outputs = layers.Dense(1)(outputs)

    model = tf.keras.Model(inputs, outputs)
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


def model_fn(inputs, params):
    """Compute logits of the model (output distribution)

    Args:
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        inputs: Features & labels

    Returns:
        output: (keras.Sequential) Sequential model
    """
    text_vectorizer = layers.TextVectorization(
        standardize=custom_standardization,
        split='whitespace',
        output_mode='int'
    )

    text_vectorizer.adapt(inputs["train"]["features"]["words"])
    vocab = text_vectorizer.get_vocabulary()

    # set up model architecture
    if params.model_version == 'mlp':
        model = mlp_model(params, vocab)
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

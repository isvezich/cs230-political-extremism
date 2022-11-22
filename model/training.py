import os

from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

import tensorflow as tf


def train_and_evaluate(inputs, model, params):
    """Evaluate the model

    Args:
        inputs: (dict) contains the inputs of the graph (features, labels...)
        model: (keras.Sequential) keras model with pre-defined layers
        params: (Params) contains hyperparameters of the model.
                Must define: num_epochs, train_size, batch_size, eval_size, save_summary_steps
    """

    logdir = os.path.join("logs")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=params.early_stopping_patience),
        ModelCheckpoint(filepath='best_model.h5', monitor='val_loss',
                        save_best_only=True),
        TensorBoard(logdir, histogram_freq=1)
    ]

    # text_vectorizer = text_vectorize(inputs['train']['features']['words'])
    # vocab = text_vectorizer.get_vocabulary()
    #
    # text_vectorizer = layers.TextVectorization(
    #     standardize=custom_standardization,
    #     split='whitespace',
    #     output_mode='int',
    #     vocabulary=vocab
    # )
    #
    # inputs['train']['features']['words'] = text_vectorizer(inputs['train']['features']['words']).numpy()
    # inputs['val']['features']['words'] = text_vectorizer(inputs['val']['features']['words']).numpy()
    # inputs['test']['features']['words'] = text_vectorizer(inputs['test']['features']['words']).numpy()
    #
    # print(type(inputs['test']['features']['words']))

    train_dataset = tf.data.Dataset.from_tensor_slices((inputs['train']['features'], inputs['train']['labels']))
    val_dataset = tf.data.Dataset.from_tensor_slices((inputs['val']['features'], inputs['val']['labels']))
    test_dataset = tf.data.Dataset.from_tensor_slices((inputs['test']['features'], inputs['test']['labels']))

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        callbacks=callbacks,
        epochs=params.num_epochs)

    loss, accuracy, f1, precision, recall, auc = model.evaluate(test_dataset)

    print("Loss: ", loss)
    print("Accuracy: ", accuracy)
    print("F1: ", f1)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("AUC: ", auc)

    return history


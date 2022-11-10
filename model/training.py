import os

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


def train_and_evaluate(inputs, model, params):
    """Evaluate the model

    Args:
        inputs: (dict) contains the inputs of the graph (features, labels...)
        model: (keras.Sequential) keras model with pre-defined layers
        params: (Params) contains hyperparameters of the model.
                Must define: num_epochs, train_size, batch_size, eval_size, save_summary_steps
    """
    features_train, labels_train = inputs['train']
    features_val, labels_val = inputs['val']
    features_test, labels_test = inputs['test']

    logdir = os.path.join("logs")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=params.early_stopping_patience),
        ModelCheckpoint(filepath='best_model.h5', monitor='val_loss',
                        save_best_only=True),
        TensorBoard(logdir, histogram_freq=1)
    ]

    history = model.fit(
        features_train,
        labels_train,
        validation_data=(features_val, labels_val),
        callbacks=callbacks,
        epochs=params.num_epochs)

    loss, accuracy, f1, precision, recall, auc = model.evaluate(features_test, labels_test)

    print("Loss: ", loss)
    print("Accuracy: ", accuracy)
    print("F1: ", f1)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("AUC: ", auc)

    return history


import os
import tensorflow as tf

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

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
        TensorBoard(logdir, histogram_freq=0)  # https://github.com/keras-team/keras/issues/15163
    ]

    if params.embeddings == 'BERT':
        train_ds = inputs['train'][2]
        val_ds = inputs['val'][2]
        test_ds = inputs['test'][2]

        with tf.device('/cpu:0'):
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                callbacks=callbacks,
                epochs=params.num_epochs)
        loss, accuracy, f1, precision, recall, auc = model.evaluate(test_ds)
    else:
        features_train, labels_train = inputs['train'][0], inputs['train'][3]
        features_val, labels_val = inputs['val'][0], inputs['val'][3]
        features_test, labels_test = inputs['test'][0], inputs['test'][3]

        print(f"features_train shape: {features_train.shape}")
        print(f"labels_train shape: {labels_train.shape}")
        print(f"features_val shape: {features_val.shape}")
        print(f"labels_val shape: {labels_val.shape}")
        print(f"features_test shape: {features_test.shape}")
        print(f"labels_test shape: {labels_test.shape}")
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

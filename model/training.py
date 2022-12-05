import json
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
    features_train, labels_train = inputs['train'][0], inputs['train'][3]
    features_val, labels_val = inputs['val'][0], inputs['val'][3]
    features_test, labels_test = inputs['test'][0], inputs['test'][3]

    logdir = os.path.join("logs")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=params.early_stopping_patience),
        ModelCheckpoint(filepath=f"best_model_{params.model_version}_{params.l2_reg_lambda}_{params.learning_rate}_{params.batch_size}_{params.dropout_rate}", monitor='val_loss',
                        save_best_only=True),
        TensorBoard(logdir, histogram_freq=0)  # https://github.com/keras-team/keras/issues/15163
    ]
    print(f"features_train shape: {features_train.shape}")
    print(f"labels_train shape: {labels_train.shape}")
    print(f"features_val shape: {features_val.shape}")
    print(f"labels_val shape: {labels_val.shape}")
    print(f"features_test shape: {features_test.shape}")
    print(f"labels_test shape: {labels_test.shape}")

    history = model.fit(
        features_train,
        labels_train,
        batch_size=params.batch_size,
        validation_data=(features_val, labels_val),
        callbacks=callbacks,
        epochs=params.num_epochs)

    loss, accuracy, f1, precision, recall = model.evaluate(features_test, labels_test)
    
    test_history = {"loss": loss, "binary_accuracy": accuracy, "f1_m": f1, "precision_m": precision, "recall_m": recall}
    json.dump(test_history, open(f"test_history_model:{params.model_version}_h1units:{params.h1_units}_h2units:{params.h2_units}_l2reglambda:{params.l2_reg_lambda}_lr:{params.learning_rate}_batchsize:{params.batch_size}_dropout:{params.dropout_rate}.json", 'w'))

    print("Loss: ", loss)
    print("Accuracy: ", accuracy)
    print("F1: ", f1)
    print("Precision: ", precision)
    print("Recall: ", recall)
#     print("AUC: ", auc)

    return history

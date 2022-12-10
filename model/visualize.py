import json
import math

import matplotlib.pyplot as plt


def metrics_to_plot(history, params):
    # grab evaluation metrics on dev set over course of training
    history_dict = history.history
    json.dump(history_dict, open(f"history_model:{params.model_version}_"
                                 f"embeddings:{params.embeddings}_"
                                 f"h1units:{params.h1_units}_"
                                 f"h2units:{params.h2_units}_"
                                 f"l2reglambda:{params.l2_reg_lambda}_"
                                 f"lr:{params.learning_rate}_"
                                 f"batchsize:{params.batch_size}_"
                                 f"dropout:{params.dropout_rate}.json", 'w'))
    epochs = range(1, len(history_dict['loss']) + 1)

    metrics = list(history_dict.keys())[:len(list(history_dict.keys())) // 2]

    # Plot 2 columns of subplots
    x_len = math.ceil(len(metrics)/2)
    figure, axis = plt.subplots(x_len, 2, layout="constrained")
    figure.set_size_inches(9, 9)

    x = 0
    y = 0
    for metric in metrics:
        axis[x, y].plot(epochs, history_dict[metric], 'bo', label=f'Training {metric}')
        axis[x, y].plot(epochs, history_dict[f"val_{metric}"], 'b', label=f'Validation {metric}')
        axis[x, y].set_title(f'Training and validation {metric}')
        axis[x, y].set_xlabel('Epochs')
        axis[x, y].set_ylabel(metric)
        axis[x, y].legend()
        if x < x_len - 1:
            x += 1
        else:
            y += 1
            x = 0

    plt.savefig(f"plt_model:{params.model_version}_"
                f"embeddings:{params.embeddings}_"
                f"h1units:{params.h1_units}_"
                f"h2units:{params.h2_units}_"
                f"l2reglambda:{params.l2_reg_lambda}_"
                f"lr:{params.learning_rate}_"
                f"batchsize:{params.batch_size}_"
                f"dropout:{params.dropout_rate}.png")
    plt.show()

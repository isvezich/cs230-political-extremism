"""Train the model"""

import argparse
import logging
import os

from model.utils import Params
from model.utils import set_logger
from model.training import train_and_evaluate
from model.input_fn import input_fn
from model.model_fn import model_fn
from model.visualize import metrics_to_plot

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--data_dir', default='data/QAnon', help="Directory containing the dataset")
parser.add_argument('--embeddings_dir', default='data/embeddings', help="Directory containing pre-trained embeddings")
parser.add_argument('--which_embeddings', default='GloVe', help="Which pre-trained embeddings to use")


if __name__ == '__main__':
    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Check that the dataset exists
    labels_dataset = os.path.join(args.data_dir, 'Hashed_allAuthorStatus.csv')
    features_dataset = os.path.join(args.data_dir, 'Hashed_Q_Submissions_Raw_Combined.csv')
    glove_dataset = os.path.join(args.embeddings_dir, 'glove.6B.50d.txt')
    msg = "{} file not found. Make sure you have the right dataset"
    assert os.path.isfile(labels_dataset), msg.format(labels_dataset)
    assert os.path.isfile(features_dataset), msg.format(labels_dataset)
    assert os.path.isfile(glove_dataset), msg.format(glove_dataset)

    logging.info("Creating the datasets...")
    # Create the two iterators over the two datasets
    print(args.which_embeddings)
    if args.which_embeddings == "GloVe":
        embeddings_path = glove_dataset
        params.embeddings = "GloVe"
    elif args.which_embeddings == "None":
        params.embeddings = None
    else:
        raise NotImplementedError("Unknown embeddings option: {}".format(args.which_embeddings))

    inputs = input_fn(labels_dataset, features_dataset)
    logging.info("- done.")

    # Define the models (2 different set of nodes that share weights for train and eval)
    logging.info("Creating the model...")
    if args.which_embeddings == 'None':
        print('which embeddings == None (train - 58')
        train_model, inputs = model_fn(inputs, params)
    else:
        train_model, inputs = model_fn(inputs, params, embeddings_path)

    logging.info("- done.")

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    history = train_and_evaluate(inputs, train_model, params)
    metrics_to_plot(history)

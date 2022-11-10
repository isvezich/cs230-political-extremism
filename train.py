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
    msg = "{} file not found. Make sure you have the right dataset"
    assert os.path.isfile(labels_dataset), msg.format(labels_dataset)
    assert os.path.isfile(features_dataset), msg.format(labels_dataset)

    logging.info("Creating the datasets...")
    # Create the two iterators over the two datasets
    inputs = input_fn(labels_dataset, features_dataset, params)
    logging.info("- done.")

    # Define the models (2 different set of nodes that share weights for train and eval)
    logging.info("Creating the model...")
    train_model = model_fn(params)
    logging.info("- done.")

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    history = train_and_evaluate(inputs, train_model, params)
    metrics_to_plot(history)
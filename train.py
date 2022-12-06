"""Train the model"""

import argparse
import logging
import os

from model.utils import Params
from model.utils import set_logger
from model.training import train_and_evaluate
from model.input_fn import input_fn, input_fn_bert_lstm, input_fn_bert_rnn
from model.model_fn import model_fn
from model.visualize import metrics_to_plot

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")
parser.add_argument('--embeddings_dir', default='data/embeddings', help="Directory containing pre-trained embeddings")
parser.add_argument('--which_embeddings', default='GloVe', help="Which pre-trained embeddings to use")
parser.add_argument('--h1_units', default=256, help="Units in hidden layer 1")
parser.add_argument('--h2_units', default=128, help="Units in hidden layer 2")
parser.add_argument('--l2_reg_lambda', default=1e-2, help="Weight on l2 penalty")
parser.add_argument('--learning_rate', default=0.001, help="Learning rate")
parser.add_argument('--batch_size', default=32, help="Batch size")
parser.add_argument('--num_epochs', default=2, help="Num epochs")
parser.add_argument('--dropout_rate', default=0.1, help="Dropout rate")
parser.add_argument('--early_stopping_patience', default=10, help="Early stopping patience")
parser.add_argument('--sample_rate', default=1., help="Percent of data to use")

if __name__ == '__main__':
    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    if args.h1_units:
        params.h1_units = int(args.h1_units)
    if args.h2_units:
        params.h2_units = int(args.h2_units)
    if args.l2_reg_lambda:
        params.l2_reg_lambda = float(args.l2_reg_lambda)
    if args.learning_rate:
        params.learning_rate = float(args.learning_rate)
    if args.batch_size:
        params.batch_size = int(args.batch_size)
    if args.num_epochs:
        params.num_epochs = int(args.num_epochs)
    if args.dropout_rate:
        params.dropout_rate = float(args.dropout_rate)
    if args.early_stopping_patience:
        params.early_stopping_patience = int(args.early_stopping_patience)
    if args.sample_rate:
        params.sample_rate = float(args.sample_rate)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    pos_dataset = os.path.join(args.data_dir, 'q-posts-v2.csv.gz')
    neg_dataset = os.path.join(args.data_dir, 'non-q-posts-v2.csv.gz')
    bert_dataset = os.path.join(args.data_dir, 'bert.csv.gz')
    glove_dataset = os.path.join(args.embeddings_dir, 'glove.6B.50d.txt')
    msg = "{} file not found. Make sure you have the right dataset"
    assert os.path.isfile(pos_dataset), msg.format(pos_dataset)
    assert os.path.isfile(neg_dataset), msg.format(neg_dataset)
    assert os.path.isfile(bert_dataset), msg.format(bert_dataset)
    assert os.path.isfile(glove_dataset), msg.format(glove_dataset)

    logging.info("Creating the datasets...")
    # Create the two iterators over the two datasets
    print(args.which_embeddings)
    if args.which_embeddings == "GloVe":
        embeddings_path = glove_dataset
        params.embeddings = "GloVe"
    elif args.which_embeddings == "SBERT":
        params.embeddings = "SBERT"
    elif args.which_embeddings == "None":
        params.embeddings = None
    else:
        raise NotImplementedError("Unknown embeddings option: {}".format(args.which_embeddings))

    if params.model_version == 'BERT':
        logging.info('Making bert dataset')
        inputs = input_fn_bert_lstm(bert_dataset, params)
    if params.model_version == 'BERT_RNN':
        logging.info('Making bert dataset')
        inputs = input_fn_bert_rnn(bert_dataset, params)
    else:
        inputs = input_fn(pos_dataset, neg_dataset, params)
    logging.info("- done.")

    # Define the models (2 different set of nodes that share weights for train and eval)
    logging.info("Creating the model...")
    if args.which_embeddings == 'None':
        print('which embeddings == None, BERT or BERT_RNN: train - 58')
        train_model, inputs = model_fn(inputs, params)
    elif args.which_embeddings == 'SBERT':
        print('which embeddings == SBERT: train - 62')
        train_model, inputs = model_fn(inputs, params)
    else:
        print('which embeddings == GloVe: train - 65')
        train_model, inputs = model_fn(inputs, params, embeddings_path)

    logging.info("- done.")

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    history = train_and_evaluate(inputs, train_model, params)
    metrics_to_plot(history, params)

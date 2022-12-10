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
parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")
parser.add_argument('--embeddings_dir', default='data/embeddings', help="Directory containing pre-trained embeddings")

# Arguments that can override params
parser.add_argument('--h1_units', help="Units in hidden layer 1")
parser.add_argument('--h2_units', help="Units in hidden layer 2")
parser.add_argument('--l2_reg_lambda', help="Weight on l2 penalty")
parser.add_argument('--learning_rate', help="Learning rate")
parser.add_argument('--batch_size', help="Batch size")
parser.add_argument('--num_epochs', help="Num epochs")
parser.add_argument('--dropout_rate', help="Dropout rate")
parser.add_argument('--early_stopping_patience', help="Early stopping patience")
parser.add_argument('--sample_rate', help="Percent of data to use")
parser.add_argument('--sentences_length', help="Max sentence length per author")
parser.add_argument('--max_features', help="Max length of features")
parser.add_argument('--embedding_size', help="Size of the embedding")
parser.add_argument('--word_embeddings', help='Which pre-trained word embeddings to use, choices - are None and GloVe')

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
    if args.sentences_length:
        params.sentences_length = int(args.sentences_length)
    if args.max_features:
        params.max_features = int(args.max_features)
    if args.embedding_size:
        params.embedding_size = int(args.embedding_size)
    if args.word_embeddings:
        params.embeddings = args.word_embeddings

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
    print(f'Embeddings == {params.embeddings}: train')
    embeddings_path = None
    if params.embeddings == "GloVe":
        embeddings_path = glove_dataset

    # Create the input tensors from the datasets
    inputs = input_fn(pos_dataset, neg_dataset, bert_dataset, params, embeddings_path)
    logging.info("- done.")

    # Define the models (2 different set of nodes that share weights for train and eval)
    logging.info("Creating the model...")
    train_model, inputs = model_fn(inputs, params)
    logging.info("- done.")

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    history = train_and_evaluate(inputs, train_model, params)
    metrics_to_plot(history, params)

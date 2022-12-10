"""General utility functions"""
import json
import logging
import numpy as np
import tensorflow as tf
from nltk import word_tokenize


# from https://github.com/cs230-stanford/cs230-code-examples/blob/master/tensorflow/nlp/model/utils.py
class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


# from https://github.com/cs230-stanford/cs230-code-examples/blob/master/tensorflow/nlp/model/utils.py
def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


# from https://github.com/cs230-stanford/cs230-code-examples/blob/master/tensorflow/nlp/model/utils.py
def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map


def sentence_to_avg(sentence, word_to_vec_map):
    """
    Converts a sentence (string) into a list of words (strings). Extracts the GloVe representation of each word
    and averages its value into a single vector encoding the meaning of the sentence.

    Arguments:
    sentence -- string, one training example from X
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation

    Returns:
    avg -- average vector encoding information about the sentence, numpy-array of shape (J,), where J can be any number
    """
    # Split sentence into list of lower case words
    words = word_tokenize(sentence.numpy().decode("utf-8").lower())

    vecs = []
    # Average the word vectors
    for w in words:
        v = word_to_vec_map.get(w)
        if v is not None:
            vecs.append(v)
    if vecs:
        avg = tf.math.reduce_mean(vecs, axis=0)
    else:
        any_word = next(iter(word_to_vec_map))
        avg = tf.zeros(word_to_vec_map[any_word].shape)

    return tf.cast(avg, dtype='float32')


def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4).

    Arguments:
    X -- array of sentences (strings), of shape (m,)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this.

    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    m = X.shape[0]  # number of training examples
    X_indices = np.zeros((m, max_len))

    for i in range(m):  # loop over training examples

        # Convert the ith training sentence in lower case and split is into words
        sentence_words = word_tokenize(X[i].numpy().decode("utf-8").lower())
        del sentence_words[max_len:]

        j = 0
        # Loop over the words of sentence_words
        for w in sentence_words:
            v = word_to_index.get(w)
            if v is not None:
                X_indices[i, j] = v
                j +=1

    return X_indices


def sentence_to_sbert_avg(inp, model):
    sentences = json.loads(inp.numpy().decode("utf-8"))
    sentences = [s for s in sentences if isinstance(s, str)]
    sentence_embeddings = model.encode(sentences)
    sentence_embeddings = np.mean(sentence_embeddings, axis=0)
    return tf.convert_to_tensor(sentence_embeddings)


def sentence_to_sbert_seq(inp, model):
    sentences = json.loads(inp.numpy().decode("utf-8"))
    sentences = [s for s in sentences if isinstance(s, str)]
    sentence_embeddings = model.encode(sentences)
    return sentence_embeddings


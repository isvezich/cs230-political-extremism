import pandas as pd
import numpy as np
import tensorflow as tf

from model.utils import read_glove_vecs, sentence_to_avg, sentences_to_indices


def load_data_to_df(path):
    df = pd.read_csv(path, compression='gzip')
    # convert types (originally all strings) & filter features to date range before first q drop
    df = df \
        .groupby("hashed_author") \
        .agg(
        {
            "title": lambda x: list(x),
            "selftext": lambda x: list(x),
            "q_level": "mean",
        }
    )

    # concatenate title & body text into 1 string to create embedding from all the words that
    # author ever wrote
    # words is title + " " + selftext
    def do_join(xs):
        return " ".join([s for s in xs if type(s) == str])

    df["words"] = (df["title"] + df["selftext"]).apply(do_join)
    df.dropna(subset=['words', 'q_level'], inplace=True)
    print(df['q_level'].value_counts())
    print(df["words"].head())

    return df

def load_bert_data_to_df(path, params):
    df = pd.read_csv(path, compression='gzip')
    if params.model_version == 'BERT_LSTM':
        df = df.groupby('post_id').agg({"text": lambda x: list(x), "author": "first", "q_level": "first"})
        df = df.groupby('author').agg({"text": lambda x: list(x), "q_level": "first"})
        df = df.sample(frac=params.sample_rate).reset_index()
    else:
        df = df.groupby('author').agg({"text": lambda x: list(x), "q_level": "first"})
        df = df.sample(frac=params.sample_rate).reset_index()
    print(f"length of features: {len(df)}")

    return df

def convert_bert_lstm_df_to_tensor(df, params):
    authors = []
    for e in df['text'].items():
        posts = []
        for i, post in enumerate(e[1]):
            if i < params.posts_length:
                del post[params.sentences_length:]
                try:
                    posts.append(tf.strings.lower(tf.ragged.constant(post)))
                except ValueError:
                    posts.append(tf.strings.lower(tf.ragged.constant([w for w in post if isinstance(w, str)])))
        authors.append(tf.ragged.stack(posts, axis=0))

    return tf.ragged.stack(authors, axis=0)


def convert_bert_rnn_mlp_df_to_tensor(df, params):
    authors = []
    for i, sentences in enumerate(df['text']):
        del sentences[params.sentences_length:]
        try:
            authors.append(tf.strings.lower(tf.ragged.constant(sentences)))
        except ValueError:
            authors.append(tf.strings.lower(tf.ragged.constant([w for w in sentences if isinstance(w, str)])))

    return tf.ragged.stack(authors, axis=0)

def load_all_data_to_df(pos_path, neg_path, params):
    pos = load_data_to_df(pos_path)
    neg = load_data_to_df(neg_path)
    features = pd.concat([pos, neg])
    features = features.sample(frac=params.sample_rate).reset_index()
    print(f"length of features: {len(features)}")

    return features

def prepare_average_word_embeddings(inputs, params, embeddings_path):
    words_to_index, index_to_words, word_to_vec_map = read_glove_vecs(embeddings_path)
    # Get a valid word contained in the word_to_vec_map.
    str_feat_train = []
    str_feat_val = []
    str_feat_test = []
    # inputs['train'][0] is words_train, a 1D string tensor, 1 string per author / label
    # inputs['train'][0].shape[0] is (n,) for dataset with n examples
    for i in range(inputs['train'][0].shape[0]):  # for each example:
        author_text = inputs['train'][0][i]
        str_feat_train.append(sentence_to_avg(author_text, word_to_vec_map))
    print('finished sentence_to_avg for train')
    for i in range(inputs['val'][0].shape[0]):
        str_feat_val.append(sentence_to_avg(inputs['val'][0][i], word_to_vec_map))
    print('finished sentence_to_avg for val')
    for i in range(inputs['test'][0].shape[0]):
        str_feat_test.append(sentence_to_avg(inputs['test'][0][i], word_to_vec_map))
    print('finished sentence_to_avg for test')
    inputs['train'][0] = tf.cast(tf.stack(str_feat_train), 'float64')
    inputs['val'][0] = tf.cast(tf.stack(str_feat_val), 'float64')
    inputs['test'][0] = tf.cast(tf.stack(str_feat_test), 'float64')
    inputs['train'].append(tf.data.Dataset.from_tensor_slices((inputs['train'][0], inputs['train'][1])) \
        .shuffle(params.batch_size, reshuffle_each_iteration=True).batch(params.batch_size))
    inputs['val'].append(tf.data.Dataset.from_tensor_slices((inputs['val'][0], inputs['val'][1])) \
        .shuffle(params.batch_size, reshuffle_each_iteration=True).batch(params.batch_size))
    inputs['test'].append(tf.data.Dataset.from_tensor_slices((inputs['test'][0], inputs['test'][1])) \
        .shuffle(params.batch_size, reshuffle_each_iteration=True).batch(params.batch_size))
    inputs['word_to_vec_map'] = word_to_vec_map
    inputs['words_to_index'] = words_to_index
    return inputs


def prepare_sequence_word_embeddings(inputs, params, embeddings_path):
    maxLen = params.max_word_length
    words_to_index, index_to_words, word_to_vec_map = read_glove_vecs(embeddings_path)
    inputs['train'][0] = sentences_to_indices(inputs['train'][0], words_to_index, maxLen)
    print('finished sentences_to_indices for train')
    inputs['val'][0] = sentences_to_indices(inputs['val'][0], words_to_index, maxLen)
    print('finished sentences_to_indices for val')
    inputs['test'][0] = sentences_to_indices(inputs['test'][0], words_to_index, maxLen)
    print('finished sentences_to_indices for test')
    inputs['train'].append(tf.data.Dataset.from_tensor_slices((inputs['train'][0], inputs['train'][1])) \
        .shuffle(params.batch_size, reshuffle_each_iteration=True).batch(params.batch_size))
    inputs['val'].append(tf.data.Dataset.from_tensor_slices((inputs['val'][0], inputs['val'][1])) \
        .shuffle(params.batch_size, reshuffle_each_iteration=True).batch(params.batch_size))
    inputs['test'].append(tf.data.Dataset.from_tensor_slices((inputs['test'][0], inputs['test'][1])) \
        .shuffle(params.batch_size, reshuffle_each_iteration=True).batch(params.batch_size))
    inputs['word_to_vec_map'] = word_to_vec_map
    inputs['words_to_index'] = words_to_index
    return inputs

def input_fn(pos_path, neg_path, bert_path, params, embeddings_path=None):
    """Input function for NER
    Args:
        pos_path: (string) relative path to positive dataset csv
        neg_path: (string) relative path to negative dataset csv
        bert_path: (string) relative path to bert csv
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        embeddings_path: (string) relative path to pre-trained embeddings if they are to be used, else None
    """
    # Load the dataset into memory
    print("Loading QAnon dataset and creating df...")
    if params.model_version.startswith('BERT'):
        print('Making bert dataset')
        df = load_bert_data_to_df(bert_path, params)
    else:
        print('Making word embedding dataset')
        df = load_all_data_to_df(pos_path, neg_path, params)

    # split into train/dev/test
    np.random.seed(0)
    indices = np.random.choice(a=[0, 1, 2], size=len(df), p=[.6, .2, .2])

    print("Splitting data into train dev test")
    train_df = df[indices == 0]
    val_df = df[indices == 1]
    test_df = df[indices == 2]

    # convert to tensor to input to model
    # words is 1D string tensor, 1 element per author: all of that author's text
    if params.model_version == 'BERT_LSTM':
        words_train = convert_bert_lstm_df_to_tensor(train_df, params)
        words_val = convert_bert_lstm_df_to_tensor(val_df, params)
        words_test = convert_bert_lstm_df_to_tensor(test_df, params)
    elif params.model_version == 'BERT_RNN' or params.model_version == 'BERT_MLP':
        words_train = convert_bert_rnn_mlp_df_to_tensor(train_df, params)
        words_val = convert_bert_rnn_mlp_df_to_tensor(val_df, params)
        words_test = convert_bert_rnn_mlp_df_to_tensor(test_df, params)
    else:
        words_train = tf.convert_to_tensor(train_df["words"], dtype=tf.string)
        words_val = tf.convert_to_tensor(val_df["words"], dtype=tf.string)
        words_test = tf.convert_to_tensor(test_df["words"], dtype=tf.string)

    labels_train = tf.convert_to_tensor(train_df["q_level"])
    labels_val = tf.convert_to_tensor(val_df["q_level"])
    labels_test = tf.convert_to_tensor(test_df["q_level"])

    inputs = {
        'train': [words_train, labels_train],
        'val': [words_val, labels_val],
        'test': [words_test, labels_test],
    }

    if params.embeddings == 'GloVe':
        print("Preparing word embeddings")
        if params.model_version == 'mlp':
            inputs = prepare_average_word_embeddings(inputs, params, embeddings_path)
        else:
            inputs = prepare_sequence_word_embeddings(inputs, params, embeddings_path)
    else:
        inputs['train'].append(tf.data.Dataset.from_tensor_slices((words_train, labels_train)) \
            .shuffle(params.batch_size, reshuffle_each_iteration=True).batch(params.batch_size))
        inputs['val'].append(tf.data.Dataset.from_tensor_slices((words_val, labels_val)) \
            .shuffle(params.batch_size, reshuffle_each_iteration=True).batch(params.batch_size))
        inputs['test'].append(tf.data.Dataset.from_tensor_slices((words_test, labels_test)) \
            .shuffle(params.batch_size, reshuffle_each_iteration=True).batch(params.batch_size))

    print("Done data processing")

    return inputs

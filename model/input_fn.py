import pandas as pd
import numpy as np
import tensorflow as tf

from model.model_fn import custom_standardization


def load_data_to_df(path):
    df = pd.read_csv(path, compression='gzip')

    # convert types (originally all strings) & filter features to date range before first q drop
    df["score"] = df["score"].astype(int)
    df["num_comments"] = df["num_comments"].astype(int)

    df = df \
        .groupby("hashed_author") \
        .agg(
        {
            "score": "mean",
            "num_comments": "mean",
            "title": lambda x: list(x),
            "selftext": lambda x: list(x),
            "q_level": "mean",
        }
    )

    # concatenate title & body text into 1 string to create embedding from all the words that
    # author ever wrote--we should consider better ways to do this
    df["words"] = (df["title"] + df["selftext"]).astype(str)
    print(df['q_level'].value_counts())
    print(df.head())

    return df

def load_bert_data_to_df(path):
    df = pd.read_csv(path, compression='gzip')
    df = df.groupby('post_id').agg({"text": lambda x: list(x), "author": "first", "q_level": "first"})
    df = df.groupby('author').agg({"text": lambda x: list(x), "q_level": "first"})

    return df

def convert_bert_df_to_tensor(df):
    authors = []
    for e in df['text'].items():
        posts = []
        for i, post in enumerate(e[1]):
            if i < 40:
                del post[5:]
                try:
                    posts.append(tf.strings.lower(tf.ragged.constant(post)))
                except ValueError:
                    post.pop()
                    posts.append(tf.strings.lower(tf.ragged.constant(post)))
        authors.append(tf.ragged.stack(posts, axis=0))

        # authors.append(tf.ragged.stack([
        #     tf.strings.lower(tf.ragged.constant(post)) for post in e[1]
        # ], axis=0))

    return tf.ragged.stack(authors, axis=0)

    # return \
    #     tf.ragged.stack([
    #         tf.ragged.stack([
    #             tf.strings.lower(tf.ragged.constant(post)) for post in e[1]
    #         ], axis=0) for e in df['text'].items()
    #     ], axis=0)


def load_all_data_to_df(pos_path, neg_path):
    pos = load_data_to_df(pos_path)
    neg = load_data_to_df(neg_path)
    features = pd.concat([pos, neg])
    features = features.sample(frac=1).reset_index()

    return features


def input_fn_bert(bert_path):
    """Input function for NER
    Args:
        bert_path: (string) relative path to labels and features csv
    """
    # Load the dataset into memory
    print("Loading QAnon dataset and creating df...")
    df = load_bert_data_to_df(bert_path)

    # split into train/dev/test
    np.random.seed(0)
    # indices = np.random.choice(a=[0, 1, 2, 3], size=len(df), p=[.03, .01, .01, 0.95])
    indices = np.random.choice(a=[0, 1, 2], size=len(df), p=[.6, .2, .2])

    print("Splitting data into train dev test")
    train_df = df[indices == 0]
    words_train = convert_bert_df_to_tensor(train_df)
    labels_train = tf.convert_to_tensor(train_df["q_level"])
    train_ds = tf.data.Dataset.from_tensor_slices((words_train, labels_train))\
        .shuffle(3, reshuffle_each_iteration=True).batch(3)

    val_df = df[indices == 1]
    words_val = convert_bert_df_to_tensor(val_df)
    labels_val = tf.convert_to_tensor(val_df["q_level"])
    val_ds = tf.data.Dataset.from_tensor_slices((words_val, labels_val))\
        .shuffle(3, reshuffle_each_iteration=True).batch(3)

    test_df = df[indices == 2]
    words_test = convert_bert_df_to_tensor(test_df)
    labels_test = tf.convert_to_tensor(test_df["q_level"])
    test_ds = tf.data.Dataset.from_tensor_slices((words_test, labels_test))\
        .shuffle(3, reshuffle_each_iteration=True).batch(3)

    print("Done data processing")
    inputs = {
        'train': (words_train, labels_train, train_ds),
        'val': (words_val, labels_val, val_ds),
        'test': (words_test, labels_test, test_ds),
    }

    return inputs

def input_fn(pos_path, neg_path):
    """Input function for NER
    Args:
        labels_path: (string) relative path to labels csv
        features_path: (string) relative path to features csv
    """
    # Load the dataset into memory
    print("Loading QAnon dataset and creating df...")
    df = load_all_data_to_df(pos_path, neg_path)

    # convert to tensor to input to model
    words = tf.convert_to_tensor(df["words"], dtype=tf.string)
    score = tf.convert_to_tensor(df["score"])
    num_replies = tf.convert_to_tensor(df["num_comments"])
    labels = tf.convert_to_tensor(df["q_level"])

    # split into train/dev/test
    np.random.seed(0)
    indices = np.random.choice(a=[0, 1, 2], size=len(labels), p=[.6, .2, .2])

    words_train = words[indices == 0]
    score_train = score[indices == 0]
    num_replies_train = num_replies[indices == 0]
    labels_train = labels[indices == 0]
    words_val = words[indices == 1]
    score_val = score[indices == 1]
    num_replies_val = num_replies[indices == 1]
    labels_val = labels[indices == 1]
    words_test = words[indices == 2]
    score_test = score[indices == 2]
    num_replies_test = num_replies[indices == 2]
    labels_test = labels[indices == 2]

    inputs = {
        'train': [words_train, score_train, num_replies_train, labels_train],
        'val': [words_val, score_val, num_replies_val, labels_val],
        'test': [words_test, score_test, num_replies_test, labels_test],
    }

    return inputs

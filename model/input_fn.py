import datetime
import re
import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords

import tensorflow as tf


def load_data_to_df(labels_path, features_path):
    labels = pd.read_csv(labels_path)
    features = pd.read_csv(features_path)

    # convert types (originally all strings) & filter features to date range before first q drop
    features["date_created"] = pd.to_datetime(features["date_created"])
    features_pre = features.loc[features["date_created"] < datetime.datetime.strptime("2017-10-01", "%Y-%m-%d"), :]
    features_pre["score"] = features_pre["score"].astype(int)
    features_pre["numReplies"] = features_pre["numReplies"].astype(int)
    features_pre["upvote_ratio"] = features_pre["upvote_ratio"].astype(float)
    features_pre["is_self"] = features_pre["is_self"].astype(int)

    features_pre = features_pre \
        .groupby("author") \
        .agg(
        {
            "subreddit": lambda x: list(x),
            "id": lambda x: list(x),
            "score": "mean",
            "numReplies": "mean",
            "title": lambda x: list(x),
            "text": lambda x: list(x),
            "is_self": "mean",
            "domain": lambda x: list(x),
            "url": lambda x: list(x),
            "permalink": lambda x: list(x),
            "upvote_ratio": "mean",
            "date_created": lambda x: list(x)
        }
    )

    features_pre.reset_index(inplace=True)

    df = features_pre.merge(labels, left_on="author", right_on="QAuthor")

    # concatenate title & body text into 1 string to create embedding from all the words that
    # author ever wrote--we should consider better ways to do this
    df["words"] = (df["subreddit"] + df["domain"] + df["title"] + df["text"]).astype(str)

    return df


# clean up junk from string
def custom_standardization(input_data):
    cachedStopWords = stopwords.words("english")

    lowercase = tf.strings.lower(input_data)
    for word in cachedStopWords:
        lowercase = tf.strings.regex_replace(lowercase, word, '')
    lowercase = tf.strings.regex_replace(lowercase, 'nan', '')
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')


def input_fn(labels_path, features_path):
    """Input function for NER

    Args:
        labels_path: (string) relative path to labels csv
        features_path: (string) relative path to features csv

    """
    # Load the dataset into memory
    print("Loading QAnon dataset and creating df...")
    df = load_data_to_df(labels_path=labels_path, features_path=features_path)

    # convert to tensor to input to model
    words = tf.convert_to_tensor(df["words"])
    score = tf.convert_to_tensor(df["score"])
    num_replies = tf.convert_to_tensor(df["numReplies"])
    labels = tf.convert_to_tensor(df["isUQ"])

    # split into train/dev/test
    np.random.seed(0)
    indices = np.random.choice(a=[0, 1, 2], size=len(labels), p=[.6, .2, .2])

    features_train = {
        "words": words[indices == 0],
        "score": score[indices == 0],
        "num_replies": num_replies[indices == 0]
    }

    features_val = {
        "words": words[indices == 1],
        "score": score[indices == 1],
        "num_replies": num_replies[indices == 1]
    }

    features_test = {
        "words": words[indices == 2],
        "score": score[indices == 2],
        "num_replies": num_replies[indices == 2]
    }

    labels_train = labels[indices == 0]
    labels_val = labels[indices == 1]
    labels_test = labels[indices == 2]

    inputs = {
        'train': {
            'features': features_train,
            'labels': labels_train
        },
        'val': {
            'features': features_val,
            'labels': labels_val
        },
        'test': {
            'features': features_test,
            'labels': labels_test
        }
    }

    return inputs

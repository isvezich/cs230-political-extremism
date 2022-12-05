# From https://www.philschmid.de/tensorflow-sentence-transformers
import tensorflow as tf
from transformers import TFBertTokenizer

from model.sentence_transformer import TFSentenceTransformer


class E2ESentenceTransformer(tf.keras.Model):
    def __init__(self, model_name_or_path, **kwargs):
        super().__init__()
        # loads the in-graph tokenizer
        self.tokenizer = TFBertTokenizer.from_pretrained(model_name_or_path, **kwargs)
        # loads our TFSentenceTransformer
        self.model = TFSentenceTransformer(model_name_or_path, **kwargs)


    def call(self, inputs):
        # runs tokenization and creates embedding
        tokenized = self.tokenizer(inputs)
        print(tokenized)
        return self.model(tokenized)


model = E2ESentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model(tf.constant(["hi there. bye.", "where is the cow today why! is it not here now there?"])).shape
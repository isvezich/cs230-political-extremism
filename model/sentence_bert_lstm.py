import tensorflow as tf

from model.e2e_sentence_transformer import E2ESentenceTransformer


# Input should be author, post, sentences e.g. (0, 0, 0) is the first author's first post's first sentence
class SentenceBertLSTM(tf.keras.Model):
    def __init__(self, model_name_or_path, params, **kwargs):
        super().__init__()
        self.model = E2ESentenceTransformer(model_name_or_path, **kwargs)
        self.params = params

    def build(self, inputs):
        self.lstm_posts = tf.keras.layers.LSTM(self.params.lstm_units_post,
                                               recurrent_regularizer=tf.keras.regularizers.L2(self.params.l2_reg_lambda),
                                               recurrent_dropout=self.params.dropout_rate)
        self.lstm_authors = tf.keras.layers.LSTM(self.params.lstm_units_author,
                                                 recurrent_regularizer=tf.keras.regularizers.L2(self.params.l2_reg_lambda),
                                                 recurrent_dropout=self.params.dropout_rate)

    def embed_post(self, post):
        post_model = self.model(post)
        return post_model

    def embed_author(self, posts):
        # sentence_encodings_per_author = tf.ragged.map_flat_values(self.embed_post, posts)
        sentence_encodings_per_author = tf.map_fn(fn=self.embed_post,
                         elems=posts,
                         fn_output_signature=tf.RaggedTensorSpec(shape=[None, 384], ragged_rank=0,
                                                                 dtype=tf.float32)
                         )

        return self.lstm_posts(sentence_encodings_per_author)


    def embed_authors(self, authors):
        author_encodings = tf.map_fn(fn=self.embed_author,
                         elems=authors,
                         fn_output_signature=tf.RaggedTensorSpec(shape=[None, self.params.lstm_units_post],
                                                                 ragged_rank=0,
                                                                 dtype=tf.float32)
                         )

        return self.lstm_authors(author_encodings)

    # Expect input of dimension 4, where the 4th dimension contains str tensors of sentences
    # @tf.function
    def call(self, inputs):
        return self.embed_authors(inputs)
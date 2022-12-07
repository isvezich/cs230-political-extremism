import tensorflow as tf

from model.e2e_sentence_transformer import E2ESentenceTransformer


# Input should be author, sentences e.g. (0, 0) is the first author's first post's first sentence
class SentenceBertMLP(tf.keras.Model):
    def __init__(self, model_name_or_path, params, **kwargs):
        super().__init__()
        self.model = E2ESentenceTransformer(model_name_or_path, **kwargs)
        self.params = params

    def build(self, inputs):
        self.mlp = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_last', keepdims=False)

    def embed_author(self, posts):
        post_model = self.model(posts)
        return post_model

    def embed_authors(self, authors):
        author_encodings = tf.map_fn(fn=self.embed_author,
                         elems=authors,
                         fn_output_signature=tf.RaggedTensorSpec(shape=[None, 384],
                                                                 ragged_rank=0,
                                                                 dtype=tf.float32)
                         )

        return self.mlp(author_encodings)

    # Expect input of dimension 4, where the 4th dimension contains str tensors of sentences
    # @tf.function
    def call(self, inputs):
        return self.embed_authors(inputs)

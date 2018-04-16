
class Mode:
    TRAIN = 'train'
    EVAL = 'eval'
    PRED = 'pred'


def build_model(sentence, mode, **config):
    """"Build the model graph.

    Arguments:
        sentence: A `Tensor` of shape `[B, N]` (where B is the batch size
            and N the sentence length and can change dynamically) containing the indices
            of the words (type `tf.int32`).
        mode: The graph mode (type `Mode`).
        config: A configuration dictionary.

    Returns:
        Training mode:
            A tuple containing the loss (type `tf.float32`, shape `[]`) and the embedding
            tensor (type `tf./float32`, shape `[K, M]` where K is the word-embedding
            dimensionality and M is the vocabulary size).
        Evaluation mode:
            A `Tensor` (type `tf.float32`, shape `[B]`) containing the perplexity for
            each batch element.
        Prediction mode:
            A `Tensor` (type `tf.int32, shape `[B, N]`) containing the maximum-likelihood
            prediction of each word.
    """

    raise NotImplementedError

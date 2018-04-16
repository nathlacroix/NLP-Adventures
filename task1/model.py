import tensorflow as tf


class Mode:
    TRAIN = 'train'
    EVAL = 'eval'
    PRED = 'pred'


def build_model(sentences, pad_ind, mode, **config):
    """"Build the model graph.

    Arguments:
        sentences: A `Tensor` of shape `[B, N]` (where B is the batch size
            and N the sentence length and can change dynamically) containing the indices
            of the words (type `tf.int32`).
        pad_ind: int corresponding to the index of the padding token.
        mode: The graph mode (type `Mode`).
        config: A configuration dictionary.

    Returns:
        Training mode:
            A tuple containing the loss (type `tf.float32`, shape `[]`) and the embedding
            tensor (type `tf.float32`, shape `[K, M]` where K is the word-embedding
            dimensionality and M is the vocabulary size).
            Note: the loss is normalized with respect to sentence length and batch_size
        Evaluation mode:
            A `Tensor` (type `tf.float32`, shape `[B]`) containing the perplexity for
            each batch element.
            Note: the perplexity is normalized with respect to the sentence length,
            but NOT with respect to batch_size
        Prediction mode:
            A `Tensor` (type `tf.int32, shape `[B, N]`) containing the maximum-likelihood
            prediction of each word.
    """

    # Note: sentence_size corresponds to the length of the output sentence.
    # Do not mix up with sentences_length which
    # correspond to the true lengths of the sentences (without padding).
    required = ['vocab_size', 'embedding_size', 'state_size', 'sentence_size']
    for r in required:
        assert r in config

    vocab_size = config['vocab_size']
    embedding_size = config['embedding_size']
    state_size = config['state_size']
    sentence_size = config['sentence_size']
    seed = config.get('seed', None)

    batch_size = tf.shape(sentences)[0]

    with tf.variable_scope('net', reuse=tf.AUTO_REUSE):
        # Create embeddings
        embeddings = tf.get_variable(
                'Embeddings', shape=[vocab_size, embedding_size],
                initializer=tf.contrib.layers.xavier_initializer(seed=seed))
        word_embeddings = tf.nn.embedding_lookup(embeddings, sentences)

        lstm = tf.contrib.rnn.LSTMCell(
                state_size,
                initializer=tf.contrib.layers.xavier_initializer(seed=seed),
                num_proj=vocab_size)
        h_0 = tf.zeros((batch_size, vocab_size))
        c_0 = tf.zeros((batch_size, state_size))
        state = tf.contrib.rnn.LSTMStateTuple(c_0, h_0)
        logits = tf.zeros(tf.stack([tf.shape(sentences)[0], tf.constant(0),
                                    tf.constant(vocab_size)], axis=0))
        i = 0

        def step(i, h_prev, s, x, logits):
            if mode == Mode.PRED:
                def cond_true():
                    xi = tf.gather(x, i, axis=1)
                    return xi

                def cond_false():
                    max_likelihood_prev_ind = tf.argmax(h_prev, axis=1)
                    xprev = tf.gather(embeddings, max_likelihood_prev_ind)
                    return xprev

                xnext = tf.cond(tf.less(i, tf.shape(sentences)[1]),
                                cond_true, cond_false)
            else:
                xnext = tf.gather(x, i, axis=1)

            h_new, state_new = lstm(xnext, s)

            logits = tf.concat([logits, tf.expand_dims(h_new, axis=1)], axis=1)
            return i + 1, h_new, state_new, x, logits

        # Dynamic while loop. Note: the logits are already downprojected
        _, _, state, _, logits = tf.while_loop(
                lambda i, h_prev, s, x, logits: tf.less(i, sentence_size),
                step,
                (i, h_0, state, word_embeddings, logits),
                shape_invariants=(
                    tf.TensorShape([]),
                    tf.TensorShape([None, vocab_size]),
                    tf.contrib.rnn.LSTMStateTuple(
                        tf.TensorShape([None, state_size]),
                        tf.TensorShape([None, vocab_size])),
                    tf.TensorShape([None, None, embedding_size]),
                    tf.TensorShape([None, None, vocab_size])))

    if mode == Mode.TRAIN:
        train_loss = compute_loss(sentences, logits, pad_ind)
        return tf.reduce_mean(train_loss), embeddings

    if mode == Mode.EVAL:
        eval_loss = compute_loss(sentences, logits, pad_ind)
        perplexity = tf.exp(eval_loss)
        return perplexity

    if mode == Mode.PRED:
        max_likelihood_pred = tf.argmax(logits, axis=2)
        return max_likelihood_pred


def mask_padding(pad_ind, input_tensor, labels):
    mask = tf.not_equal(labels, pad_ind)
    mask = tf.cast(mask, tf.float32)
    sentences_length = tf.reduce_sum(mask, axis=1)
    return tf.multiply(input_tensor, mask), sentences_length


def compute_loss(labels, output, pad_ind):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=output)
    loss, sentences_length = mask_padding(pad_ind, loss, labels)
    loss = tf.divide(tf.reduce_sum(loss, axis=1), sentences_length)
    return loss

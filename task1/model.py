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
            of the words (type `tf.int32`). In the predictive mode, the batch size must
            be one.
        pad_ind: int corresponding to the index of the padding token.
        mode: The graph mode (type `Mode`).
        config: A configuration dictionary. Its entry `"sentence_size"` corresponds to
            the length of the output sentence. In the case of predictive mode
            (generation), it can be different from the length of the input sentence,
            which is not padded in that mode.

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

    # Required configuration entries
    required = ['vocab_size', 'embedding_size', 'state_size', 'sentence_size']
    for r in required:
        assert r in config

    # Copy parameters for convenience
    vocab_size = config['vocab_size']
    embedding_size = config['embedding_size']
    state_size = config['state_size']
    sentence_size = config['sentence_size']
    down_proj_size = config.get('down_proj_size', None)
    seed = config.get('random_seed', None)

    batch_size = tf.shape(sentences)[0]

    # Create network
    with tf.variable_scope('net', reuse=tf.AUTO_REUSE):
        # Create embeddings
        embeddings = tf.get_variable(
                'Embeddings', shape=[vocab_size, embedding_size],
                initializer=tf.contrib.layers.xavier_initializer(seed=seed))
        word_embeddings = tf.nn.embedding_lookup(embeddings, sentences)

        # Create LSTM celle
        lstm = tf.contrib.rnn.LSTMCell(
                state_size,
                initializer=tf.contrib.layers.xavier_initializer(seed=seed))

        # Create intermediate projection
        if down_proj_size is not None:
            W_down_proj = tf.get_variable(
                    'Down_proj_weights', shape=[down_proj_size, state_size],
                    initializer=tf.contrib.layers.xavier_initializer(seed=seed))
        else:
            down_proj_size = state_size

        # Create softmax projection
        W_softmax = tf.get_variable(
                'Softmax_weights', shape=[vocab_size, down_proj_size],
                initializer=tf.contrib.layers.xavier_initializer(seed=seed))

        # Initialize loop variables
        h_0 = tf.zeros((batch_size, state_size))
        c_0 = tf.zeros((batch_size, state_size))
        state = tf.contrib.rnn.LSTMStateTuple(c_0, h_0)
        logits = tf.zeros(tf.stack([tf.shape(sentences)[0], tf.constant(0),
                                    tf.constant(vocab_size)], axis=0))
        i = 0

        # Body of the dynamic while loop
        def step(i, h_prev, s, x, logits):
            # Input (last word) of the cell is different in prediction mode
            if mode == Mode.PRED:
                def cond_true():
                    return tf.gather(x, i, axis=1)

                def cond_false():
                    if down_proj_size is not None:
                        down_projection = tf.matmul(h_prev, tf.transpose(W_down_proj))
                    else:
                        down_projection = h_prev
                    max_likelihood_prev_ind = tf.argmax(
                            tf.matmul(down_projection, tf.transpose(W_softmax)), axis=1)
                    return tf.gather(embeddings, max_likelihood_prev_ind)

                # Take ground truth word in the first part and prediction in the second
                xnext = tf.cond(tf.less(i, tf.shape(sentences)[1]),
                                cond_true, cond_false)
            else:
                xnext = tf.gather(x, i, axis=1)  # take the i-th ground truth word

            h_new, state_new = lstm(xnext, s)

            if down_proj_size is not None:
                down_projection = tf.matmul(h_new, tf.transpose(W_down_proj))
            else:
                down_projection = h_new

            logit = tf.matmul(down_projection, tf.transpose(W_softmax))
            logits = tf.concat([logits, tf.expand_dims(logit, axis=1)], axis=1)
            return i + 1, h_new, state_new, x, logits

        # Dynamic while loop.
        # We do not predict <bos> and do not feed <eos> as input.
        _, _, state, _, logits = tf.while_loop(
                lambda i, h_prev, s, x, logits: tf.less(i, sentence_size-1),
                step,
                (i, h_0, state, word_embeddings, logits),
                shape_invariants=(
                    tf.TensorShape([]),
                    tf.TensorShape([None, state_size]),
                    tf.contrib.rnn.LSTMStateTuple(
                        tf.TensorShape([None, state_size]),
                        tf.TensorShape([None, state_size])),
                    tf.TensorShape([None, None, embedding_size]),
                    tf.TensorShape([None, None, vocab_size])))

    if mode == Mode.TRAIN:
        train_loss = compute_loss(sentences[:, 1:], logits, pad_ind)
        return tf.reduce_mean(train_loss), embeddings

    if mode == Mode.EVAL:
        eval_loss = compute_loss(sentences[:, 1:], logits, pad_ind)
        perplexity = tf.exp(eval_loss)
        return perplexity

    if mode == Mode.PRED:
        max_likelihood_pred = tf.argmax(logits, axis=2)
        return tf.concat(
                [sentences[:],
                 tf.cast(max_likelihood_pred[:, tf.shape(sentences)[1]-1:], tf.int32)],
                axis=1)


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

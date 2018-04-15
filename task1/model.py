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


    # initial variables
    ## config & dimension vars
    vocab_size = None
    embedding_size = None
    state_size = None
    sentence_size = None
    seed = None
    batch_size = tf.shape(sentences)[0]

    if 'vocab_size' in config:
        vocab_size = config['vocab_size']
    else:
        raise ValueError('vocab_size parameter not found in config.')

    if 'embedding_size' in config:
        embedding_size = config['embedding_size']
    else:
        raise ValueError('embedding_size parameter not found in config.')

    if 'state_size' in config:
        state_size = config['state_size']
    else:
        raise ValueError('state_size parameter not found in config.')

    if 'sentence_size' in config:
        sentence_size = config['sentence_size']
    else:
        raise ValueError('sentence_size parameter not found in config.')

    if 'seed' in config:
        seed = config['seed']

    with tf.variable_scope('net', reuse=tf.AUTO_REUSE):

        # Create embeddings
        embeddings = tf.get_variable('Embeddings',shape=[vocab_size, embedding_size],
                                     initializer=tf.contrib.layers.xavier_initializer(seed=seed))
        word_embeddings = tf.nn.embedding_lookup(embeddings, sentences)


        #Create LSTM
        lstm =  tf.contrib.rnn.LSTMCell(state_size,
                                        initializer=tf.contrib.layers.xavier_initializer(seed=seed),
                                        num_proj=vocab_size)
        h_t = tf.zeros((batch_size, vocab_size))
        c_t = tf.zeros((batch_size, state_size))
        state =  tf.contrib.rnn.LSTMStateTuple(c_t, h_t)
        logits = tf.zeros(tf.stack([tf.shape(sentences)[0], tf.constant(0), tf.constant(vocab_size)], axis=0))
        i = 0

        def step(i, h_prev, s, x, logits):
            if mode == Mode.PRED:
                def cond_true(s):
                    xi = tf.gather(x, i, axis=1)
                    h_new, state_new = lstm(xi, s)
                    return h_new, state_new

                def cond_false(h_prev, embeddings, s):
                    max_likelihood_prev_ind = tf.argmax(tf.nn.softmax(h_prev), axis=1)
                    h_new, state_new = lstm(tf.gather(embeddings, max_likelihood_prev_ind), s)
                    return h_new, state_new

                h_new, state_new = tf.cond(tf.less(i, tf.shape(sentences)[1]),
                                           lambda: cond_true(s),
                                           lambda: cond_false(h_prev, embeddings, s))
                logits = tf.concat([logits, tf.expand_dims(h_new, axis=1)], axis=1)
            else:
                xi = tf.gather(x, i, axis=1)
                h_new, state_new = lstm(xi, s)
                logits = tf.concat([logits, tf.expand_dims(h_new, axis=1)], axis=1)


            return i + 1, h_new, state_new, x, logits

        # Dynamic while loop. Note: the logits are already downprojected
        _, _, state, _, logits = tf.while_loop(lambda i, h_prev, s, x, logits: tf.less(i, sentence_size),
                                               step,
                                               (i, h_t, state, word_embeddings, logits),
                                               shape_invariants=(tf.TensorShape([]),
                                                                 tf.TensorShape([None, vocab_size]),
                                                                 tf.contrib.rnn.LSTMStateTuple(tf.TensorShape([None, state_size]),
                                                                                               tf.TensorShape([None, vocab_size])),
                                                                 tf.TensorShape([None, None, embedding_size]),
                                                                 tf.TensorShape([None, None, vocab_size])))

    if mode == Mode.TRAIN:
        train_loss = compute_loss(sentences, logits, pad_ind, batch_normalize=True)
        return tf.reduce_sum(train_loss), embeddings

    if mode == Mode.EVAL:
        eval_loss = compute_loss(sentences, logits, pad_ind, batch_normalize=False)
        perplexity = tf.exp(eval_loss)
        return perplexity

    if mode == Mode.PRED:
        prob = tf.nn.softmax(logits) # is axis -1 ok ?
        max_likelihood = tf.reduce_max(prob, axis=2)
        return max_likelihood


def mask_padding(pad_ind, input_tensor, labels):
    """"Replaces cell content of an input Tensor with zeros for cells which the value
       corresponds to pad_ind in a label tensor.

    Arguments:
        pad_ind: int of the padding index in the lookup table.
        input_tensor: Tensor (shape '[B, N]') one wishes to remove the padding from.
        labels: Tensor (shape '[B, N]') of the same shape as input_tensor with the corresponding labels.
    Returns:
        masked_tensor:
            input_tensor where the values of the cells corresponding to pad_ind in the label tensor
            are replaced by zeros.
        sentences_length:
            A `Tensor` (type `tf.float32`, shape `[B]`) containing the lengths of the non-padded sentences.
    """

    mask = tf.not_equal(labels, pad_ind)
    mask = tf.cast(mask, tf.float32)
    sentences_length = tf.reduce_sum(mask, axis=1)

    return tf.multiply(input_tensor, mask), sentences_length

def compute_loss(labels, output, pad_ind, batch_normalize=False, loss_function=None):
    if loss_function is not None:
        loss = loss_function(labels, output)
    else:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=output)

    loss, sentences_length = mask_padding(pad_ind, loss, labels)

    if batch_normalize:
        # loss = reduced on batch dim loss / (batch size * sentence length)
        loss = tf.divide(tf.reduce_sum(loss, axis=1),
                         tf.multiply(tf.cast(tf.shape(labels)[0], tf.float32), sentences_length))
    else:
        loss = tf.divide(tf.reduce_sum(loss, axis=1), sentences_length)

    return loss

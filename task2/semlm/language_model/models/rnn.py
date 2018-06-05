import tensorflow as tf
import numpy as np

from .base_model import BaseModel, Mode


class RNN(BaseModel):
    def build_model(self, data, pad_ind, mode, ending_size=1, **config):
        """"Build the model graph.

        Arguments:
            data: A `Tensor` of shape `[B, N]` (where B is the batch size
                and N the sentence length and can change dynamically)
                containing the indices of the words (type `tf.int32`).
            pad_ind: int corresponding to the index of the padding token.
            mode: The graph mode (type `Mode`).
            ending_size: in eval mode, number of components in the ending
            config: A configuration dictionary.

        Returns:
            Training mode:
                A tuple containing the loss (type `tf.float32`, shape `[]`) and
                the embedding tensor (type `tf.float32`, shape `[K, M]` where K is
                the word-embedding dimensionality and M is the vocabulary size).
                Note: the loss is normalized with respect to
                sentence length and batch_size
            Evaluation mode:
                A `Tensor` (type `tf.float32`, shape `[1]`) containing the
                probability of the ending given the context (in log scale)
        """

        # Required configuration entries
        required = ['vocab_size', 'embedding_size', 'state_size']
        for r in required:
            assert r in config

        # Copy parameters for convenience
        vocab_size = config['vocab_size']
        embedding_size = config['embedding_size']
        state_size = config['state_size']
        down_proj_size = config.get('down_proj_size', None)
        seed = config.get('random_seed', None)

        batch_size = tf.shape(data)[0]

        # Create network
        with tf.variable_scope('net', reuse=tf.AUTO_REUSE):
            # Create embeddings
            embeddings = tf.get_variable(
                    'Embeddings', shape=[vocab_size, embedding_size],
                    initializer=tf.contrib.layers.xavier_initializer(seed=seed))
            # Set the embedding of the pad token to zero
            mask = np.ones(vocab_size, dtype=np.float32)
            mask[pad_ind] = 0
            embeddings = tf.multiply(tf.expand_dims(mask, 1), embeddings)
            word_embeddings = tf.nn.embedding_lookup(embeddings, data)

            # Create LSTM cell
            lstm = tf.contrib.rnn.LSTMCell(
                    state_size,
                    initializer=tf.contrib.layers.xavier_initializer(seed=seed))

            # Create intermediate projection
            if down_proj_size is not None:
                W_down_proj = tf.get_variable(
                        'Down_proj_weights', shape=[state_size, down_proj_size],
                        initializer=tf.contrib.layers.xavier_initializer(seed=seed))
                W_down_proj = tf.tile(tf.expand_dims(W_down_proj, 0), [batch_size, 1, 1])
            else:
                down_proj_size = state_size

            # Create softmax projection
            W_softmax = tf.get_variable(
                    'Softmax_weights', shape=[down_proj_size, vocab_size],
                    initializer=tf.contrib.layers.xavier_initializer(seed=seed))
            W_softmax = tf.tile(tf.expand_dims(W_softmax, 0), [batch_size, 1, 1])

            def length(sequence):
                """ Compute the actual length of each sequence (without padding)
                sequence should be of size [batch_size, max_length, embedding_size] """
                used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
                # used is now a mask with 1 for actual words and 0 for padding
                length = tf.reduce_sum(used, 1)
                length = tf.cast(length, tf.int32)
                return length

            # Create the rnn
            output, state = tf.nn.dynamic_rnn(lstm,
                                              word_embeddings,
                                              dtype=tf.float32,
                                              sequence_length=length(word_embeddings))

            # Downproject if necessary
            if down_proj_size != state_size:
                down_projection = tf.matmul(output, W_down_proj)
            else:
                down_projection = output

            # Get the pre-softmax logits
            logits = tf.matmul(down_projection, W_softmax)

        if mode == Mode.TRAIN:
            train_loss = self.compute_loss(data, logits, pad_ind)
            return tf.reduce_mean(train_loss), embeddings

        if mode == Mode.VAL:
            eval_loss = self.compute_loss(data, logits, pad_ind)
            perplexity = tf.exp(eval_loss)
            return perplexity

        if mode == Mode.TEST:
            probs = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=data,
                                                                   logits=logits)
            probs = tf.reshape(probs, [-1])[-ending_size:]
            return tf.reduce_sum(probs)

    def mask_padding(self, pad_ind, input_tensor, labels):
        mask = tf.not_equal(labels, pad_ind)
        mask = tf.cast(mask, tf.float32)
        sentences_length = tf.reduce_sum(mask, axis=1)
        return tf.multiply(input_tensor, mask), sentences_length

    def compute_loss(self, labels, output, pad_ind):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                              logits=output)
        loss, sentences_length = self.mask_padding(pad_ind, loss, labels)
        loss = tf.divide(tf.reduce_sum(loss, axis=1), sentences_length)
        return loss

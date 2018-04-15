import tensorflow as tf
import numpy as np

from model import build_model, Mode


if __name__ == '__main__':
    config = {
            'vocab_size': 20000,
            'state_size': 512,
            'embedding_size': 100,
            'sentence_size': 30,
    }
    batch_size = 12
    pad_ind = 178 # random

    sentences = tf.placeholder(tf.int32, [None, None])

    loss, embedding_tensor = build_model(sentences, pad_ind, Mode.TRAIN, **config)
    perplexity = build_model(sentences, pad_ind, Mode.EVAL, **config)
    predictions = build_model(sentences, pad_ind, Mode.PRED, **config)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        # Check training
        sample = np.random.choice(
                np.arange(config['vocab_size']),
                (batch_size, config['sentence_size']))
        _loss, _embedding = sess.run(
                [loss, embedding_tensor], feed_dict={sentences: sample})
        assert np.isscalar(_loss)
        assert _embedding.shape == (config['vocab_size'],
                                    config['embedding_size'])

        # Check evaluation
        sample = np.random.choice(
                np.arange(config['vocab_size']),
                (batch_size, config['sentence_size']))
        _perplexity = sess.run(perplexity, feed_dict={sentences: sample})
        assert _perplexity.shape == (batch_size,)

        # Check prediction
        ground_truth_size = 5
        #sample = np.random.choice(
        #        np.arange(config['vocab_size']), (1, ground_truth_size))
        sample = np.array([[10,24,178,178,178],[1, 2,3,4,5]])
        _predictions = sess.run(predictions, feed_dict={sentences: sample})

        assert _predictions.shape == (2, config['sentence_size'])
        d = 0

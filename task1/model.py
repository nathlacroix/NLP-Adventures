import tensorflow as tf
import numpy as np
import csv
import os
from dataset import get_dataset

default_config = {
    'sentence_size': 30,
    'batch_size': 64,
    'random_seed': 0,
    'num_epochs': 100,
    'learning_rate': 0.001,
    'validation': True,
    'validation_interval': 100,
    'saved_models_dir': "./logs/models",
    'output_dir': "./logs/outputs",
    'summaries_dir': "./logs/summaries",
}

class Mode:
    TRAIN = 'train'
    EVAL = 'eval'
    PRED = 'pred'


def build_model(sentence, mode, experiment, **config):
    """"Build the model graph.

    Arguments:
        sentence: A `Tensor` of shape `[B, N]` (where B is the batch size
            and N the sentence length and can change dynamically) containing the indices
            of the words (type `tf.int32`).
        mode: The graph mode (type `Mode`).
        experiment: 'A', 'B' or 'C'
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


def train(data, experiment, **config):
    """ Train the model on the data.

    Arguments:
        data: A tuple of size 1 or 2 (if a validation set is available)
            where each element is an array of shape (M, N) (where M is the dataset size
            and N the sentence length) containing the indices of the words.
        experiment: 'A', 'B' or 'C'
        config: A configuration dictionary.
    """
    # Prepare batches
    (M, N) = data[0].shape
    x = tf.placeholder(tf.int32, [None, N], 'sentences')
    x = tf.random_shuffle(x, seed=config['random_seed'])
    batches = [data[0][beg:(beg+config['batch_size']), :]
               for beg in range(0, M, config['batch_size'])]
    # List of np.array of size 'batch_size' x N
    # (except for the last batch, which can be smaller)
    n_batches = len(batches)

    (loss, embedding) = build_model(x, 'train', experiment, **config)
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(config['learning_rate'], name='optimizer')
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)

    summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(config['summaries_dir'],
                                                      'experiment_' + experiment,
                                                      'train'))
    test_writer = tf.summary.FileWriter(os.path.join(config['summaries_dir'],
                                                     'experiment_' + experiment,
                                                     'test'))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print("Start training...")
        for epoch in range(config['num_epochs']):
            for x_batch in batches:
                _, batch_loss, summary = sess.run([train_op, loss, summaries], feed_dict={x: x_batch})
                step = sess.run(global_step) - 1
                train_writer.add_summary(summary, step)
                print("Epoch " + str(epoch) + " ; batch " +
                      str(step % n_batches) + " ; loss = " + str(batch_loss))
                if config['validation'] and step % config['validation_interval'] == 0:
                    validation_loss, summary = sess.run([loss, summaries], feed_dict={x: data[1]})
                    test_writer.add_summary(summary, step)
                    print("Validation loss = " + str(validation_loss))

        save_path = saver.save(sess, os.path.join(config['saved_models_dir'],
                                                  'experiment_' + experiment,
                                                  "experiment_" + experiment + ".ckpt"))
        print("Training is over!")
        print("Trained model saved in " + save_path)


def eval(data, experiment, **config):
    """ Evaluate the model on the data and store the result in a file.

    Arguments:
        data: An array of shape [M, N] (where M is the dataset size
            and N the sentence length) containing the indices
            of the words.
        filepath: path where the output file will be stored
        experiment: 'A', 'B' or 'C'
        config: A configuration dictionary.

    Returns:
        A numpy array of size M containing the perplexity of every sentences.
    """
    x = tf.placeholder(tf.int32, [None, data.shape[1]], 'sentences')
    perplexities_op = build_model(x, 'eval', experiment, **config)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, os.path.join(config['saved_models_dir'],
                                         'experiment_' + experiment,
                                         "experiment_" + experiment + ".ckpt"))
        perplexities = sess.run(perplexities_op, feed_dict={x: data})

    path = os.path.join(config['output_dir'], "group27.perplexity" + experiment)
    with open(path, 'w') as writer:
        for p in perplexities:
            writer.write(str(p) + '\n')
    print("Evaluation is over!")
    print("Evaluation saved in " + path)


def pred(data, dictionary, experiment='C', **config):
    """ Evaluate the model on the data and store the result in a file.

    Arguments:
        data: An array of shape [M, N] (where M is the dataset size
            and N the sentence length) containing the indices
            of the words.
        filepath: path where the output file will be stored
        experiment: 'A', 'B' or 'C'
        config: A configuration dictionary.

    Returns:
        A numpy array of size M containing the perplexity of every sentences.
    """
    x = tf.placeholder(tf.int32, [None, data.shape[1]], 'sentences')
    prediction_op = build_model(x, 'pred', experiment, **config)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, os.path.join(config['saved_models_dir'],
                                         'experiment_' + experiment,
                                         "experiment_" + experiment + ".ckpt"))
        prediction = sess.run(prediction_op, feed_dict={x: data})

    # Write the predictions to a file
    path = os.path.join(config['output_dir'], "group27.continuation")
    with open(path, 'w') as writer:
        for s in range(prediction.shape[0]):
            for w in range(prediction.shape[1]):
                word = dictionary[prediction[s, w]]
                if word == '<eos>' or w == prediction.shape[1]-1:
                    writer.write(word)
                    break
                else:
                    writer.write(word + ' ')
            if s != prediction.shape[0] - 1:
                writer.write('\n')
    print("Prediction is over!")
    print("Prediction saved in " + path)

import logging
import numpy as np
import os
from models.base_model import Mode
from models.rnn import RNN

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
import tensorflow as tf


def set_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)


def train(train_data, val_data, pad_ind, exp_dir, **config):
    """ Train the model on the data.

    Arguments:
        train_data: An array of shape (M, N) used for training (where M is the dataset
                    size and N the sentence length) containing the indices of the words.
        val_data: idem but for validation.
        pad_ind: index of the token <pad>
        exp_dir: path of the log directory for this experiment.
        config: A configuration dictionary.
    """
    set_seed(config['random_seed'])
    # Prepare batches
    (M, N) = train_data.shape
    np.random.shuffle(train_data)
    x = tf.placeholder(tf.int32, [None, N], 'sentences')
    batches = [train_data[beg:(beg+config['batch_size']), :]
               for beg in range(0, M, config['batch_size'])]
    val_batches = [val_data[beg:(beg+config['batch_size']), :]
                   for beg in range(0, val_data.shape[0], config['batch_size'])]
    # List of np.array of size 'batch_size' x N
    # (except for the last batch, which can be smaller)
    n_batches = len(batches)

    if config['model'] == 'rnn':
        model = RNN()
    with tf.name_scope("training"):
        (loss, embedding) = model.build_model(x, pad_ind, Mode.TRAIN, **config)
        summary_train = tf.summary.scalar('training_loss', loss)

    optimizer = tf.train.AdamOptimizer(config['learning_rate'], name='optimizer')
    global_step = tf.Variable(0, name='global_step', trainable=False)
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    train_op = optimizer.apply_gradients(zip(gradients, variables),
                                         global_step=global_step)

    with tf.name_scope("validation"):
        perplexities_op = model.build_model(x, pad_ind, Mode.VAL, **config)

    train_writer = tf.summary.FileWriter(os.path.join(exp_dir,
                                                      "summaries/train"))
    test_writer = tf.summary.FileWriter(os.path.join(exp_dir,
                                                     "summaries/test"))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        logging.info("Start training...")
        try:
            for it in range(config['num_iter']):
                x_batch = batches[it % n_batches]
                _, batch_loss, summary = sess.run([train_op, loss, summary_train],
                                                  feed_dict={x: x_batch})
                if it % config['validation_interval'] == 0:
                    train_writer.add_summary(summary, it)
                    # Perform batch validation
                    val_perplexities = []
                    for val_batch in val_batches:
                        val_perplexity = sess.run(perplexities_op,
                                                  feed_dict={x: val_batch})
                        val_perplexities += val_perplexity.tolist()
                    val_perplexity = np.mean(val_perplexities)

                    # Create a summary averaging the whole validation dataset
                    val_summary = tf.Summary()
                    val_summary.value.add(tag="validation_perplexity",
                                          simple_value=val_perplexity)
                    test_writer.add_summary(val_summary, it)
                    tf.logging.info(
                            'Iter {:4d}: Loss {:.4f}: '
                            'Validation perplexity {:.4f}'.format(
                                it, batch_loss, val_perplexity))
            logging.info("Training is over!")
        except KeyboardInterrupt:
            logging.info('Got Keyboard Interrupt, saving model and closing.')

        save_path = saver.save(sess, os.path.join(exp_dir, "models/model.ckpt"))
        logging.info("Model saved in " + save_path)


def eval(data, pad_ind, exp_dir, **config):
    """ Evaluate the model on the data and store the result in a .npz file.
        This function operates with a batch size of 1 for now.

    Arguments:
        data: A tuple of the form (context, ending1, ending2) where each element
            is a list of list of components
        pad_ind: index of the token <pad>
        exp_dir: path of the log directory for this experiment
        config: A configuration dictionary.
    """
    set_seed(config['random_seed'])
    # Prepare batches
    M = len(data[0])
    x = tf.placeholder(tf.int32, [None, None], 'sentences')
    ending_size = tf.placeholder(tf.int32, [], 'ending_size')

    if config['model'] == 'rnn':
        model = RNN()
    proba_ending = model.build_model(x, pad_ind, Mode.TEST, ending_size, **config)
    saver = tf.train.Saver()
    features = []
    with tf.Session() as sess:
        saver.restore(sess, os.path.join(exp_dir, "models/model.ckpt"))
        for i in range(M):
            feature = []
            for j in range(1, config['context_size'] + 1):
                feature_endings = []
                for ending in [1, 2]:
                    ending_length = len(data[ending][i])
                    sentence = data[0][i][-j:] + data[ending][i]
                    sentence = np.array(sentence).reshape((1, -1))  # batch 1
                    f = sess.run(proba_ending, feed_dict={x: sentence,
                                                          ending_size: ending_length})
                    feature_endings.append(float(f))
                if config['use_binary_features']:
                    feature_endings.append(float(feature_endings[0] <
                                                 feature_endings[1]))
                feature += feature_endings
            features.append(feature)

    features = np.array(features)

    path = os.path.join(exp_dir, "outputs/events.npz")
    arrays = {}
    for j in range(features.shape[1]):
        arrays['feature' + str(j)] = features[:, j]
    np.savez_compressed(path, **arrays)
    logging.info("Evaluation is over!")
    logging.info("Features saved in " + path)

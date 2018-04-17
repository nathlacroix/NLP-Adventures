import logging
import numpy as np
import os

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
import tensorflow as tf  # noqa: E402
from model import build_model, Mode  # noqa: E402


def train(train_data, val_data, external_embedding, pad_ind, exp_dir, **config):
    """ Train the model on the data.

    Arguments:
        train_data: An array of shape (M, N) used for training (where M is the dataset
                    size and N the sentence length) containing the indices of the words.
        val_data: idem but for validation.
        external_embedding: a np array of shape (embedding_size, vocab_size).
        pad_ind: index of the token <pad>
        exp_dir: path of the log directory for this experiment.
        config: A configuration dictionary.
    """
    # Prepare batches
    (M, N) = train_data.shape
    np.random.shuffle(train_data)
    x = tf.placeholder(tf.int32, [None, N], 'sentences')
    batches = [train_data[beg:(beg+config['batch_size']), :]
               for beg in range(0, M, config['batch_size'])]
    # List of np.array of size 'batch_size' x N
    # (except for the last batch, which can be smaller)
    n_batches = len(batches)

    with tf.name_scope("training"):
        (loss, embedding) = build_model(x, pad_ind, Mode.TRAIN, **config)
        summary_train = tf.summary.scalar('training_loss', loss)

    # Possibly load a pretrained embedding
    if config['use_external_embedding']:
        pretrained_embeddings = tf.placeholder(tf.float32, [None, None])
        assign_op = embedding.assign(pretrained_embeddings)

    optimizer = tf.train.AdamOptimizer(config['learning_rate'], name='optimizer')
    global_step = tf.Variable(0, name='global_step', trainable=False)
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    train_op = optimizer.apply_gradients(zip(gradients, variables),
                                         global_step=global_step)

    with tf.name_scope("validation"):
        perplexities_op = build_model(x, pad_ind, Mode.EVAL, **config)
        summary_val = tf.summary.scalar('validation_perplexity',
                                        tf.reduce_mean(perplexities_op))

    train_writer = tf.summary.FileWriter(os.path.join(exp_dir,
                                                      "summaries/train"))
    test_writer = tf.summary.FileWriter(os.path.join(exp_dir,
                                                     "summaries/test"))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        if config['use_external_embedding']:  # assign a pretrained embedding
            sess.run(assign_op, {pretrained_embeddings: external_embedding})

        logging.info("Start training...")
        try:
            for it in range(config['num_iter']):
                x_batch = batches[it % n_batches]
                _, batch_loss, summary = sess.run([train_op, loss, summary_train],
                                                  feed_dict={x: x_batch})
                if it % config['validation_interval'] == 0:
                    train_writer.add_summary(summary, it)
                    val_perplexity, summary_eval = sess.run([perplexities_op,
                                                             summary_val],
                                                            feed_dict={x: val_data})
                    test_writer.add_summary(summary_eval, it)
                    tf.logging.info(
                            'Iter {:4d}: Loss {:.4f}: '
                            'Validation perplexity {:.4f}'.format(
                                it, batch_loss, np.mean(val_perplexity)))
            logging.info("Training is over!")
        except KeyboardInterrupt:
            logging.info('Got Keyboard Interrupt, saving model and closing.')

        save_path = saver.save(sess, os.path.join(exp_dir, "models/model.ckpt"))
        logging.info("Model saved in " + save_path)


def eval(data, pad_ind, exp_dir, **config):
    """ Evaluate the model on the data and store the result in a file
        (perplexity of every sentence).

    Arguments:
        data: An array of shape [M, N] (where M is the dataset size
              and N the sentence length) containing the indices of the words.
        pad_ind: index of the token <pad>
        exp_dir: path of the log directory for this experiment
        config: A configuration dictionary.
    """
    # Prepare batches
    (M, N) = data.shape
    x = tf.placeholder(tf.int32, [None, N], 'sentences')
    batches = [data[beg:(beg+config['batch_size']), :]
               for beg in range(0, M, config['batch_size'])]
    # List of np.array of size 'batch_size' x N
    # (except for the last batch, which can be smaller)

    perplexities_op = build_model(x, pad_ind, Mode.EVAL, **config)
    saver = tf.train.Saver()
    perplexities = []
    with tf.Session() as sess:
        saver.restore(sess, os.path.join(exp_dir, "models/model.ckpt"))
        for x_batches in batches:
            perplexity_batch = sess.run(perplexities_op, feed_dict={x: x_batches})
            perplexities += perplexity_batch.tolist()

    path = os.path.join(exp_dir, "outputs/group27.perplexity")
    with open(path, 'w') as writer:
        for p in perplexities:
            writer.write(str(p) + '\n')
    logging.info("Evaluation is over!")
    logging.info("Evaluation saved in " + path)


def pred(data, dictionary, pad_ind, exp_dir, **config):
    """ Make a prediction with the model from the data and store the result in a file
        (sentences completed by the network)

    Arguments:
        data: A list of lists (of various sizes) where each sublist contains the indices
              of the words of the beginning of the sentence to predict.
        dictionary: list of words composing the vocabulary. The index of each word
                    determines uniquely the word.
        pad_ind: index of the token <pad>
        exp_dir: path of the log directory for this experiment
        config: A configuration dictionary.
    """
    x = tf.placeholder(tf.int32, [None, None], 'sentences')
    prediction_op = build_model(x, pad_ind, Mode.PRED, **config)
    saver = tf.train.Saver()
    prediction = np.zeros((len(data), config['sentence_size']), dtype=int)
    with tf.Session() as sess:
        saver.restore(sess, os.path.join(exp_dir, "models/model.ckpt"))
        for s in range(len(data)):
            current_sentence = np.array([data[s]])
            prediction[s, :] = sess.run(prediction_op, feed_dict={x: current_sentence})

    # Write the predictions to a file
    path = os.path.join(exp_dir, "outputs/group27.continuation")
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
    logging.info("Prediction is over!")
    logging.info("Prediction saved in " + path)

import tensorflow as tf
import numpy as np
import os

from model import build_model, Mode


def train(train_data, val_data, exp_dir, **config):
    """ Train the model on the data.

    Arguments:
        train_data: An array of shape (M, N) used for training (where M is the dataset
                    size and N the sentence length) containing the indices of the words.
        val_data: idem but for validation
        exp_dir: path of the log directory for this experiment
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
        (loss, embedding) = build_model(x, Mode().TRAIN, **config)
        summary_train = tf.summary.scalar('training_loss', loss)
    optimizer = tf.train.AdamOptimizer(config['learning_rate'], name='optimizer')
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)

    with tf.name_scope("validation"):
        perplexities_op = build_model(x, Mode().EVAL, **config)
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
        # TODO: load pretrained embedding if path is present in config

        print("Start training...")
        for epoch in range(config['num_epochs']):
            for x_batch in batches:
                _, batch_loss, summary = sess.run([train_op, loss, summary_train],
                                                  feed_dict={x: x_batch})
                step = sess.run(global_step) - 1
                print("Epoch " + str(epoch) + " ; batch " +
                      str(step % n_batches) + " ; loss = " + str(batch_loss))
                if step % config['validation_interval'] == 0:
                    train_writer.add_summary(summary, step)
                    val_perplexity, summary_eval = sess.run([perplexities_op,
                                                             summary_val],
                                                            feed_dict={x: val_data})
                    test_writer.add_summary(summary_eval, step)
                    print("Validation perplexity = " + str(val_perplexity))

        save_path = saver.save(sess, os.path.join(exp_dir, "models/model.ckpt"))
        print("Training is over!")
        print("Trained model saved in " + save_path)


def eval(data, exp_dir, **config):
    """ Evaluate the model on the data and store the result in a file
        (perplexity of every sentence).

    Arguments:
        data: An array of shape [M, N] (where M is the dataset size
              and N the sentence length) containing the indices of the words.
        exp_dir: path of the log directory for this experiment
        config: A configuration dictionary.
    """
    x = tf.placeholder(tf.int32, [None, data.shape[1]], 'sentences')
    perplexities_op = build_model(x, Mode().EVAL, **config)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, os.path.join(exp_dir, "models/model.ckpt"))
        perplexities = sess.run(perplexities_op, feed_dict={x: data})

    path = os.path.join(exp_dir, "outputs/group27.perplexity")
    with open(path, 'w') as writer:
        for p in perplexities:
            writer.write(str(p) + '\n')
    print("Evaluation is over!")
    print("Evaluation saved in " + path)


def pred(data, dictionary, exp_dir, **config):
    """ Make a prediction with the model from the data and store the result in a file
        (sentences completed by the network)

    Arguments:
        data: An array of shape [M, N] (where M is the dataset size
              and N the sentence length) containing the indices of the words.
        dictionary: list of words composing the vocabulary. The index of each word
                    determines uniquely the word.
        exp_dir: path of the log directory for this experiment
        config: A configuration dictionary.
    """
    x = tf.placeholder(tf.int32, [None, data.shape[1]], 'sentences')
    prediction_op = build_model(x, Mode().PRED, **config)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, os.path.join(exp_dir, "models/model.ckpt"))
        prediction = sess.run(prediction_op, feed_dict={x: data})

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
    print("Prediction is over!")
    print("Prediction saved in " + path)

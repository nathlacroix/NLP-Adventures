import numpy as np
import spacy
import argparse
import csv
import datetime as tm
from pathlib import Path
import yaml
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import os

dir = str(Path(__file__).parents[0])
train_parsing_instructions = {'beginning': [2],
                        'body': [3, 4],
                        'climax': [5],
                        'ending': [6]}
eval_parsing_instructions = {'beginning': [1],
                        'body': [2, 3],
                        'climax': [4],
                        'ending1': [5],
                        'ending2': [6]}
test_parsing_instructions = {'beginning': [0],
                             'body' : [1,2],
                             'climax': [3],
                             'ending1': [4],
                             'ending2': [5]
                             }
story_struct={'beginning': 0,
              'body': 1,
              'climax': 2,
              'ending': 3,
              'context': 'sum'}
default_probas = [{'posterior': 'ending', 'prior': ['climax', 'body', 'beginning']},
                  {'posterior': 'ending', 'prior': ['climax', 'body']},
                  {'posterior': 'ending', 'prior': 'climax'},
                  {'posterior': 'ending', 'prior': 'context'}]
vader_pos = .05
vader_neg = -.05
blob_pos = .1
blob_neg = -.1

default_sent = {'method': 'average'}

def load_stories(filename, parsing_instructions, header=True):
    """
    Load the stories from filename and sort them in three
    dimensions: the 4 first concatenated sentences, the
    first proposition and the second option.
    """
    if(not Path(filename).exists()):
        raise Exception(filename + " cannot be found. Aborting.")
    stories = []
    with open(filename, 'r') as csvfile:
        if header:
            csvfile.readline()

        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            segmented_story = []
            for _, sentences in parsing_instructions.items():
                seg = ''
                for sentence in sentences:
                    seg = seg + row[sentence] + ' '

                segmented_story = segmented_story + [seg]
            stories.append(segmented_story)

    return stories

class SentimentAnalyzer:
    '''
    Class to perform sentiment analysis on a given dataset.
    Required arguments for init:
    -sentiment_files_path_dict: this dict lists the paths to the different lexicons to load
                                as sentiments references. For the moment this is hard coded into
                                positive, negative and mpqa.
    Attributes
    -probas_wanted:
        list of probabilities to compute. Default is hard coded list from paper.
        Has to be coded as list of dicts, where each dict is contains at least the key 'posterior'.
        If it has also a 'prior', the latter should be either a list if the prior is a joint prior,
        or simply a value.
        Values are 'strings' that can be found in story_struct such as "ending", "beginning", etc.
    -sent_traj_counts_array:
        array containing the counts of each sentiment trajectory encountered in training.
        The first columns correspond to the corresponding sentiment of the part of the story,
        last column indicates the number of occurences of this trajectory.
    -sent_condensed_traj_counts_array:
        same as above but with condensed stories (= all context is evaluated at once).
    -save_traj_path:
        path to save the sent_counts
    '''

    def __init__(self, sentiment_files_path_dict=None,
                 probas_wanted=default_probas,
                 sent_traj_counts_path=' ',
                 save_traj=True,
                 save_traj_path=None,
                 force_retrain=False,
                 sent_method=default_sent,
                 vader_pos_threshold=.05,
                 vader_neg_threshold=-.05, **kwargs):
        ''' Note: positive_words & negative words are lists of strings, mpqa_dicts is a list of dict'''
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print('ERROR: did you install the spacy language model on your computer?' \
                  ' If not do it using eg. : python -m spacy download en_core_web_sm')
        print('INFO: Loading sentiment lists ...')
        self.positive_words, self.negative_words, self.mpqa_dicts \
            = self.load_sentiment_lexica(sentiment_files_path_dict)
        print('INFO: Done.')

        self.sent_traj_counts_dict = {}
        self.sent_condensed_traj_counts_dict = {}
        self.sent_traj_counts_array = None
        self.sent_condensed_traj_counts_array = None
        self.sent_traj_counts_path = dir + sent_traj_counts_path
        self.save_traj = save_traj
        self.save_traj_path = dir + save_traj_path
        self.probas_wanted = probas_wanted
        self.combination_of_methods = sent_method['method']
        self.force_retrain = force_retrain

        self.sent_method = sent_method['method']

        self.pos_threshold = sent_method.get('pos_threshold',
                                             vader_pos if sent_method['method'] == 'vader'
                                                       else blob_pos if sent_method['method'] == 'blobtext'
                                                       else 0.001)
        self.neg_threshold = sent_method.get('neg_threshold',
                                             vader_neg if sent_method['method'] == 'vader'
                                                       else blob_neg if sent_method['method'] == 'blobtext'
                                                       else -0.001)

        if save_traj and save_traj_path is None:
            print('ERROR: did not specify saving directory for sentiment traj. They will not be saved')
            self.save_traj = False

    def load_sentiment_lexica(self, path_dict):
        positives = self.read_wordlist(path_dict['positive'])
        negatives = self.read_wordlist(path_dict['negative'])
        mpqas = self.read_mpqa(path_dict['mpqa'])
        return positives, negatives, mpqas

    def read_wordlist(self, path):
        with open(dir + path, mode='r') as wordlist_file:
            wordlist = []
            for line in wordlist_file:
                if line.startswith(';') or line.startswith('\n'):
                    pass
                else:
                    wordlist.append(line[0:-1])

            if wordlist == []:
                print('WARNING : could not parse {}. No words retrieved.'.format(path))

        return wordlist

    def read_mpqa(self, path):
        with open(dir + path, mode='r') as file:
            wordlist = []
            for line in file:
                if line.startswith(';') or line.startswith('\n'):
                    pass
                else:
                    splited = line.split(' ')
                    dict = {}
                    for arg in splited:
                        key, value = arg.split('=')
                        if value.endswith('\n'):
                            value = value[:-1]
                        dict[key] = value
                    wordlist.append(dict)

        return wordlist

    def story2sent(self, story, combination_of_methods=None, return_normalized=True, **kwargs):
        story_sent = []
        vader_sent = []
        blobtext_sent = []
        if combination_of_methods == None:
            combination_of_methods = self.combination_of_methods

        if self.sent_method == 'vader':
            analyzer = SentimentIntensityAnalyzer()
            for seg in story:
                vader_sent.append(analyzer.polarity_scores(seg)['compound'])
            return self.categorize(vader_sent,
                                   self.pos_threshold,
                                   self.neg_threshold) if return_normalized else vader_sent
        elif self.sent_method == 'blobtext':
            for seg in story:
                blobtext_sent.append(TextBlob(seg).sentiment.polarity)
            return self.categorize(blobtext_sent,
                                   self.pos_threshold,
                                   self.neg_threshold) if return_normalized else blobtext_sent

        else:
            for seg in story:
                pos, neg, pos_mpqa, neg_mpqa = 0, 0, 0, 0
                seg_parsed = self.nlp(seg)
                for sentence in seg_parsed.sents:
                    for token in sentence:
                        if token.pos_ in ['ADP', 'AUX', 'DET', 'NUM', 'PRON',
                                          'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'SPACE']:
                            continue
                        negated = self.is_negated(token)
                        if token.lemma_ in self.positive_words:
                            if negated:
                                neg = neg + 1
                            else:
                                pos = pos + 1
                        if token.lemma_ in self.negative_words:
                            if negated:
                                pos = pos + 1
                            else:
                                neg = neg + 1

                        # search if in mpqa list
                        if kwargs.get('lock_pos', False):
                            mpqa_dict = next((dict for dict in self.mpqa_dicts
                                              if (dict['word1'] == token.lemma_
                                                  and (dict['pos1'] == 'anypos' or
                                                       dict['pos1'] == self.transl(token.pos_)))), None)
                        else:
                            mpqa_dict = next((dict for dict in self.mpqa_dicts if dict['word1'] == token.lemma_), \
                                         None)
                        if mpqa_dict is not None:
                            if mpqa_dict['priorpolarity'] == 'positive':
                                if negated:
                                    neg_mpqa = neg_mpqa + 1
                                else:
                                    pos_mpqa = pos_mpqa + 1
                            if mpqa_dict['priorpolarity'] == 'negative':
                                if negated:
                                    pos_mpqa = pos_mpqa + 1
                                else:
                                    neg_mpqa = neg_mpqa + 1

                seg_sent = pos - neg
                seg_sent_mpqa = pos_mpqa - neg_mpqa
                story_sent.append([seg_sent, seg_sent_mpqa])
            story_sent = self.combine_sentiment_methods(story_sent, combination_of_methods,
                                                        return_normalized=return_normalized)
        #print('story sent: {}' .format(story_sent))
        print('vader: {}'.format(vader_sent))
        return story_sent

    def categorize(self, array, pos_threshold, neg_threshold):
        return list(map(int, [1 if sent > pos_threshold else -1 if sent < neg_threshold else 0 for sent in array]))


    def is_negated(self, token):
        negated_token = False
        if 'neg' in [child.dep_ for child in token.children]:
                negated_token = True
        return negated_token


    def transl(self, pos):
        pos_name = {
            'ADV': 'adverb',
            'VERB': 'verb',
            'NOUN': 'noun',
            'ADJ': 'adj'
        }
        return  pos_name.get(pos, 'else')


    def combine_sentiment_methods(self, story_sent, combination='average', return_normalized=True):
        story_sent = np.asarray(story_sent)
        if combination == 'average':
            story_sent = np.sum(story_sent, axis=1)
        if combination == 'binglui':
            story_sent = np.sign(story_sent[:,0])
        if combination == 'mpqa':
            story_sent = np.sign(story_sent[:, 1])
        if return_normalized:
            return np.sign(story_sent)
        else:
            return story_sent

    def sent_traj_to_str(self, sent):
        return " ".join(str(x) for x in sent)

    def train(self, train_stories_list, **kwargs):
        '''
        This function trains a sentiment_analyzer either by loading precomputed counts arrays
        or on given training story list.
        If a saving path has been specified, it will automatically save the counts arrays for next time.

        :param train_stories_list:
                    list of training stories
        :return:
        '''
        if Path(self.sent_traj_counts_path).exists() and not self.force_retrain:
            print("INFO: found file with sentiment trajectories counts in {}." \
                  " Loading array from file instead of training." \
                  .format(self.sent_traj_counts_path))
            with np.load(self.sent_traj_counts_path) as data:
                self.sent_traj_counts_array = data['sent_traj_counts_array']
                self.sent_condensed_traj_counts_array = data['sent_condensed_traj_counts_array']
        else:
            print("INFO: Did not find file with pretrained sentiment trajectories." \
                  "Training ngram model on {} stories" .format(len(train_stories_list)))
            i = 0
            for train_story in train_stories_list:
                i = i + 1
                if i % 50 == 0:
                    print("INFO: Processing story {}".format(i))
                sentiment = self.story2sent(train_story, **kwargs)
                #print(train_story)
                sentiment_condensed = self.story2sent(train_story, return_normalized=False, **kwargs)
                sentiment_condensed = np.sign([np.sum(sentiment_condensed[0:story_struct['ending']]),
                                               sentiment[story_struct['ending']]])
                #not condensed
                if self.sent_traj_to_str(sentiment) in self.sent_traj_counts_dict:
                    self.sent_traj_counts_dict[self.sent_traj_to_str(sentiment)] = \
                        self.sent_traj_counts_dict[self.sent_traj_to_str(sentiment)] + 1
                else:
                    self.sent_traj_counts_dict[self.sent_traj_to_str(sentiment)] = 1

                #condensed
                if self.sent_traj_to_str(sentiment_condensed) in self.sent_condensed_traj_counts_dict:
                    self.sent_condensed_traj_counts_dict[self.sent_traj_to_str(sentiment_condensed)] = \
                        self.sent_condensed_traj_counts_dict[self.sent_traj_to_str(sentiment_condensed)] + 1
                else:
                    self.sent_condensed_traj_counts_dict[self.sent_traj_to_str(sentiment_condensed)] = 1
            i = 0
            #not condensed
            for traj, val in self.sent_traj_counts_dict.items():
                row = np.concatenate((np.fromstring(traj, dtype=np.int, sep=' '),
                                      np.array([val])))
                if i == 0:
                    self.sent_traj_counts_array = row
                else:
                    self.sent_traj_counts_array = np.vstack((self.sent_traj_counts_array,
                                                             row))
                i = i + 1

            #condensed
            i = 0
            for traj, val in self.sent_condensed_traj_counts_dict.items():
                row = np.concatenate((np.fromstring(traj, dtype=np.float, sep=' '),
                                      np.array([val])))
                if i == 0:
                    self.sent_condensed_traj_counts_array = row.astype(np.int)
                else:
                    self.sent_condensed_traj_counts_array= np.vstack((self.sent_condensed_traj_counts_array,
                                                             row.astype(np.int)))
                i = i + 1

            if self.save_traj:
                try:
                    np.savez_compressed(self.save_traj_path ,
                                        sent_traj_counts_array=self.sent_traj_counts_array,
                                        sent_condensed_traj_counts_array=self.sent_condensed_traj_counts_array)
                except FileNotFoundError:
                    f = open(self.save_traj_path , 'w')
                    f.close()

    def predict_proba(self, eval_stories_list, probas_wanted=None, predict_neutral=False, **kwargs):
        '''
        This function predicts probabilities
        :param eval_stories_list:
        :param probas_wanted:
        :return:
        '''
        #note: two last columns have to be "ending 1 and ending2"
        if probas_wanted == None:
            probas_wanted = self.probas_wanted
        print("INFO: Model predicting the following probabilities: {}".format(probas_wanted))

        if self.sent_traj_counts_array is None:
            raise Exception("Model not trained. Please train model first.")
        proba_features = []

        if not predict_neutral:
            print("Removing neutral endings..." , end='')
            self.sent_condensed_traj_counts_array = \
                self.sent_condensed_traj_counts_array[self.sent_condensed_traj_counts_array[:, 1] != 0]
            self.sent_traj_counts_array = \
                self.sent_traj_counts_array[self.sent_traj_counts_array[:, story_struct['ending']] != 0]
            print("Done.")

        i = 0
        for story in eval_stories_list:
            if i % 50 == 0:
                print("Predicting story {}/{}".format(i, len(eval_stories_list)))
            i += 1

            story_sent = self.story2sent(story, return_normalized=False)
            print(story)
            print(story_sent)
            #make sure the two endings are in story_sent: note: normally 4 dims in array but + counts = 5
            assert len(story_sent) == self.sent_traj_counts_array.shape[1]

            for ending in [len(story_sent) - 1, len(story_sent) - 2]:
                story_proba_features = []
                # if ending is neutral, send 0 proba back (?!)

                masked_sent_story = np.asarray([x for i, x in enumerate(story_sent) if i != ending])
                for proba in probas_wanted:
                    if 'prior' in proba:
                        if isinstance(proba['prior'], list):
                            proba_val = self.calc_proba_prior(np.sign(masked_sent_story),
                                                          story_struct[proba['posterior']],
                                                          [story_struct[prior] for prior in proba['prior']])
                        elif proba['prior'] == 'context':
                            context_sent = np.sign(np.sum(masked_sent_story[0:story_struct['ending']]))
                            condensed_sent = np.array([context_sent,
                                                       np.sign(masked_sent_story[story_struct['ending']])])
                            # 0 because since size of array changed,
                            # just let know that should take ending as posterior last element (= ending)
                            proba_val = self.calc_proba_prior(condensed_sent, 1, 0,
                                                              self.sent_condensed_traj_counts_array)
                        else:
                            proba_val = self.calc_proba_prior(np.sign(masked_sent_story),
                                                              story_struct[proba['posterior']],
                                                              story_struct[proba['prior']])

                    else:
                        proba_val = self.calc_proba_no_prior(np.sign(masked_sent_story),
                                                             story_struct[proba['posterior']])
#                    print("Proba {}: {}" .format(proba, proba_val))
                    story_proba_features.append(proba_val)
                proba_features.append(story_proba_features)

        proba_features = np.asarray(proba_features).reshape(((-1,
                                                              2 * (self.sent_traj_counts_array.shape[1]-1))))
        return proba_features[:, :self.sent_traj_counts_array.shape[1]-1], \
               proba_features[:, self.sent_traj_counts_array.shape[1]-1:]


    def calc_proba_no_prior(self, sent_story, idx, array=None):
        if array is None:
            array = self.sent_traj_counts_array

        masked_sent_array = self.mask_sent_array(array, sent_story[idx], idx)
        if masked_sent_array.size == 0:
            #print('WARNING: could not find the probability. Will return 0')
            return 0
        else:
            return np.sum(masked_sent_array[:,-1]) / np.sum(array[:, -1])


    def calc_proba_prior(self, sent_story, sent_idx, prior_idx, array=None):
        if array is None:
            array = self.sent_traj_counts_array
        masked_sent_array = self.mask_sent_array(array,
                                                 sent_story[prior_idx],
                                                 prior_idx)
        return self.calc_proba_no_prior(sent_story,
                                        sent_idx,
                                        masked_sent_array)


    def mask_sent_array(self, array, value, idx):
        if isinstance(idx, int):
            mask = array[:, idx] \
                          == np.multiply(np.ones(array.shape[0],
                                                 dtype=np.int), value)
        else:
            mask = np.all(array[:, idx] == np.multiply(np.ones((array.shape[0], len(idx)),
                                                               dtype=np.int), value), axis=1)
        return array[mask]


    def get_sent_endings(self, stories_list):
        sent_endings = []
        for story in stories_list:
            sent_endings.append(self.story2sent(story,
                                                return_normalized=False)[-2:])
        return np.asarray(sent_endings)


    def generate_bin_features(self, probas_ending1, probas_ending2):
        comparison = np.ones(probas_ending1.shape)
        comparison[probas_ending1 < probas_ending2] = -1
        return comparison


    def generate_neutral_features(self, probas_ending1, probas_ending2):
        n1 = np.all(probas_ending1[..., :] == 0, axis=1).astype(np.int)
        n2 = np.all(probas_ending2[..., :] == 0, axis=1).astype(np.int)
        return np.expand_dims(n1, 1), np.expand_dims(n2,1)


    def generate_diff_sent_features(self, probas_ending1, probas_ending2,
                                     exclude_neutral=False):
        diff = np.expand_dims(np.any(probas_ending1[:, ...] != probas_ending2[:, ...], axis=1), 1)

        if not exclude_neutral:
            return diff.astype(np.int)
        else:
            n1, n2 = self.generate_neutral_features(probas_ending1,
                                                    probas_ending2)
            if not np.any(n1) or not np.any(n2):
                print("WARNING: you requested to exclude neutral in diff_sent extra "
                      "features, but there are no neutral endings. The result will be "
                      "the same as if you didn't ask to exclude neutrals.")
            #remove "true values" of diff array if ending1 or 2 is neutral
            temp = np.logical_and(np.logical_not(n1.astype(np.bool)), diff)
            diff_excl_neutr = np.logical_and(np.logical_not(n2.astype(np.bool)),
                                             temp)
            return diff_excl_neutr.astype(np.int)


    def generate_extra_features(self, probas_ending1, probas_ending2,
                                features, stories_list=None, indices=[0, 4]):
        extra = []
        print("Adding requested extra features: {}" .format(features))
        if 'bin' in features:
            b = self.generate_bin_features(probas_ending1,
                                           probas_ending2)
            extra.append(b)

        if 'neutral' in features:
            # Note: if predict_neutral was activated during predict_proba,
            # this feature will be useless
            n1, n2 = self.generate_neutral_features(probas_ending1,
                                                    probas_ending2)
            if not np.any(n1) or not np.any(n2):
                print("WARNING: did not find any neutral ending. "
                      "You asked for adding neutral_features while probably keeping"
                      " predict_neutral as True. I will not add any neutral feature"
                      " as it would be useless.")
            else:
                extra.append(n1)
                extra.append(n2)

        if 'diff_sent_endings' in features:
            d = self.generate_diff_sent_features(probas_ending1,
                                                 probas_ending2)
            extra.append(d)
        if 'diff_sent_endings_exclude_neutral' in features:
            dnn = self.generate_diff_sent_features(probas_ending1,
                                                   probas_ending2,
                                                   exclude_neutral=True)
            extra.append(dnn)
        if 'sent_endings' in features:
            extra.append(self.get_sent_endings(stories_list))

        return np.hstack(extra)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str,
                        help="path to the stories")
    parser.add_argument('output_path', type=str, help="path to the output file")
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('test_file_name', type=str, help='choose between val_stories.csv, ' 
                                                         'test_stories.csv, or test_nlu18.csv')
#    parser.add_argument('--pretrained_traj_path', type=str, help='Path to file containing array' \
#                        'of "counts" of sentiment trajectories')
#    parser.add_argument('--save_traj_path', type=str, help="path to store sentiment_trajectories of model." \
#                        " By default is the same as pretrained_traj_path")
#    parser.add_argument('--force_retrain', type=bool)
    args = parser.parse_args()

    # Load the stories, process them and compute their word embedding
    print('Loading stories according to parsing instructions: {}'.format(train_parsing_instructions))
    train_stories = load_stories(args.data_path + '/train_stories.csv',
                                 train_parsing_instructions)
    print("Train Stories loaded.")
    if args.test_file_name == 'val_stories.csv':
        print('Loading stories according to parsing instructions: {}'.format(eval_parsing_instructions))
        test_stories = load_stories(args.data_path + '/val_stories.csv',
                                    eval_parsing_instructions)
        print("Eval Stories loaded.")
    elif args.test_file_name == 'test_nlu18.csv':
        print('Loading stories according to parsing instructions: {}'.format(test_parsing_instructions))
        test_stories = load_stories(args.data_path + '/test_nlu18.csv',
                                    test_parsing_instructions, header=False)
        print("Test NLU 18 Stories loaded.")
    elif args.test_file_name == 'test_stories.csv':
        print('Loading stories according to parsing instructions: {}'.format(eval_parsing_instructions))
        test_stories = load_stories(args.data_path + '/test_stories.csv',
                                    eval_parsing_instructions)
        print("STC Test Stories loaded.")

    with open(args.config, 'r') as f:
        config = yaml.load(f)
    print("Config: {}".format(config))
    sentiment_analyzer = SentimentAnalyzer(**config)
    start = tm.datetime.now()
    sentiment_analyzer.train(train_stories[0:config.get('n_train_max', None)], **config)
    print('Training time: \n{}' .format(tm.datetime.now() - start))
    print('Traj counts: {} \n Traj condensed counts: \n{}'\
            .format(sentiment_analyzer.sent_traj_counts_array,
                    sentiment_analyzer.sent_condensed_traj_counts_array))

    start = tm.datetime.now()
    print('Computing probabilities ...')
    proba_ending1, \
        proba_ending2 = sentiment_analyzer.predict_proba(test_stories[0:config.get('n_test_max',
                                                                                   None)], **config)
    extra_features = sentiment_analyzer.generate_extra_features(proba_ending1,
                                                                proba_ending2,
                                                                config.get('extra_features',
                                                                           ['bin']))
    print(extra_features)
    # Compute the topic similarity between the endings and the context
    print(proba_ending1, proba_ending2)
    print("Done. Time for prediction: {}" .format(tm.datetime.now() - start))

    # Write the features to a .npz file
    np.savez_compressed(args.output_path,
                        sentiment_ending1=proba_ending1,
                        sentiment_ending2=proba_ending2,
                        extra_features=extra_features)


    print("Sentiment features stored in " + args.output_path)

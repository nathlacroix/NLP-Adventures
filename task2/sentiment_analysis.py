import numpy as np
import spacy
import argparse
import csv
import datetime as tm
from pathlib import Path

sentiment_files_path_dict = {'negative': '/home/nathan/NLP-Adventures/task2' \
                                         '/ressources/sentiment_lexica/negative_words.txt',
                             'positive': '/home/nathan/NLP-Adventures/task2' \
                                         '/ressources/sentiment_lexica/positive_words.txt',
                             'mpqa': '/home/nathan/NLP-Adventures/task2' \
                                     '/ressources/sentiment_lexica/subjective_clues.txt'}
sentiment_files_path_dict_cluster = {'negative': '/cluster/home/lna/NLP-Adventures/task2' \
                                         '/ressources/sentiment_lexica/negative_words.txt',
                             'positive': '/cluster/home/lna/NLP-Adventures/task2' \
                                         '/ressources/sentiment_lexica/positive_words.txt',
                             'mpqa': '/cluster/home/lna/NLP-Adventures/task2' \
                                     '/ressources/sentiment_lexica/subjective_clues.txt'}
train_parsing_instructions = {'beginning': [1],
                        'body': [2, 3],
                        'climax': [4],
                        'ending': [5]}
eval_parsing_instructions = {'beginning': [1],
                        'body': [2, 3],
                        'climax': [4],
                        'ending1': [5],
                        'ending2': [6]}
story_struct={'beginning': 0,
              'body': 1,
              'climax': 2,
              'ending': 3,
              'context': 'sum'}
default_probas = [{'posterior': 'ending', 'prior': ['climax', 'body', 'beginning']},
                  {'posterior': 'ending', 'prior': ['climax', 'body']},
                  {'posterior': 'ending', 'prior': 'climax'},
                  {'posterior': 'ending', 'prior': 'context'}]


def load_stories(filename, parsing_instructions):
    """
    Load the stories from filename and sort them in three
    dimensions: the 4 first concatenated sentences, the
    first proposition and the second option.
    """
    if(not Path(filename).exists()):
        raise Exception(filename + " cannot be found. Aborting.")
    stories = []
    with open(filename, 'r') as csvfile:
        csvfile.readline()  # get rid of the header
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

    def __init__(self, sentiment_files_path_dict,
                 probas_wanted=default_probas,
                 combination_of_methods='average',
                 sent_traj_counts_arrays_filepath='',
                 save_traj=True,
                 save_traj_path=None,
                 force_retrain=False):
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
        self.sent_traj_counts_arrays_filepath = sent_traj_counts_arrays_filepath
        self.save_traj = save_traj
        self.save_traj_path = save_traj_path
        self.probas_wanted = probas_wanted
        self.combination_of_methods = combination_of_methods
        self.force_retrain=force_retrain
        if save_traj and save_traj_path is None:
            print('ERROR: did not specify saving directory for sentiment traj. They will not be saved')
            self.save_traj = False

    def load_sentiment_lexica(self, path_dict):
        positives = self.read_wordlist(path_dict['positive'])
        negatives = self.read_wordlist(path_dict['negative'])
        mpqas = self.read_mpqa(path_dict['mpqa'])
        return positives, negatives, mpqas

    def read_wordlist(self, path):
        with open(path, mode='r') as wordlist_file:
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
        with open(path, mode='r') as file:
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

    def story2sent(self, story, combination_of_methods=None, return_normalized=True):
        story_sent = []
        if combination_of_methods == None:
            combination_of_methods = self.combination_of_methods
        for seg in story:
            pos, neg, pos_mpqa, neg_mpqa = 0, 0, 0, 0
            seg_parsed = self.nlp(seg)

            for sentence in seg_parsed.sents:

                for token in sentence:

                    if token.pos_ in ['ADV', 'ADP', 'AUX', 'DET', 'NUM', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'SPACE']:
                        continue

                    negated = False #self.is_negated(token)
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
        story_sent = self.combine_sentiment_methods(story_sent, combination_of_methods, return_normalized)
        return story_sent


    def is_negated(self, token):
        negated_token = False
        if 'neg' in [child.dep_ for child in token.children]:
                negated_token = True
        return negated_token

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

    def train(self, train_stories_list):
        '''
        This function trains a sentiment_analyzer either by loading precomputed counts arrays
        or on given training story list.
        If a saving path has been specified, it will automatically save the counts arrays for next time.

        :param train_stories_list:
                    list of training stories
        :return:
        '''
        if Path(self.sent_traj_counts_arrays_filepath).exists() and not self.force_retrain:
            print("INFO: found file with sentiment trajectories counts in {}." \
                  " Loading array from file instead of training." \
                  .format(self.sent_traj_counts_arrays_filepath))
            with np.load(self.sent_traj_counts_arrays_filepath) as data:
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
                sentiment = self.story2sent(train_story)
                sentiment_condensed = self.story2sent(train_story, return_normalized=False)
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
                row = np.concatenate((np.fromstring(traj, dtype=np.int, sep=' '),
                                      np.array([val])))
                if i == 0:
                    self.sent_condensed_traj_counts_array = row
                else:
                    self.sent_condensed_traj_counts_array= np.vstack((self.sent_condensed_traj_counts_array,
                                                             row))
                i = i + 1

            if self.save_traj:
                try:
                    np.savez_compressed(self.save_traj_path ,
                                        sent_traj_counts_array=self.sent_traj_counts_array,
                                        sent_condensed_traj_counts_array=self.sent_condensed_traj_counts_array)
                except FileNotFoundError:
                    f = open(self.save_traj_path , 'w')
                    f.close()

    def predict_proba(self, eval_stories_list, probas_wanted=None, predict_neutral=False):
        '''
        This function predicts probabilities
        :param eval_stories_list:
        :param probas_wanted:
        :return:
        '''
        #note: two last columns have to be "ending 1 and ending2"
        if probas_wanted == None:
            probas_wanted = self.probas_wanted
        print(probas_wanted)

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

        for story in eval_stories_list:
            story_sent = self.story2sent(story, return_normalized=False)
            print(story)
            print(story_sent)
            assert len(story_sent) == self.sent_traj_counts_array.shape[1] #make sure the two endings are in story_sent: note: normally 4 dims in array but + counts = 5
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
                            condensed_sent = np.array([context_sent, np.sign(masked_sent_story[story_struct['ending']])])
                            proba_val = self.calc_proba_prior(condensed_sent, 1, 0, self.sent_condensed_traj_counts_array)  # 0 because since size of array changed, just let know that should take ending as posterior last element (= ending)
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
        proba_features = np.asarray(proba_features).reshape(((-1, 2* (self.sent_traj_counts_array.shape[1]-1))))
        return proba_features[:, :self.sent_traj_counts_array.shape[1]-1], \
               proba_features[:, self.sent_traj_counts_array.shape[1]-1:]


    def generate_bin_features(self, probas_ending1, probas_ending2):
        comparison = np.ones(proba_ending1.shape)
        comparison[proba_ending1 < proba_ending2] = -1
        return comparison


    def calc_proba_no_prior(self, sent_story, idx, array=None):
        if array is None:
            array = self.sent_traj_counts_array

        masked_sent_array = self.mask_sent_array(array, sent_story[idx], idx)
        if masked_sent_array.size == 0:
            print('WARNING: could not find the probability. Will return 0')
            return 0
        else:
            return np.sum(masked_sent_array[:,-1]) / np.sum(array[:, -1])


    def calc_proba_prior(self, sent_story, sent_idx, prior_idx, array=None):
        if array is None:
            array = self.sent_traj_counts_array
        masked_sent_array = self.mask_sent_array(array, sent_story[prior_idx], prior_idx)
        return self.calc_proba_no_prior(sent_story, sent_idx, masked_sent_array)


    def mask_sent_array(self, array, value, idx):
        if isinstance(idx, int):
            mask = array[:, idx] \
                          == np.multiply(np.ones(array.shape[0],
                                                 dtype=np.int), value)
        else:
            mask = np.all(array[:, idx] == np.multiply(np.ones((array.shape[0], len(idx)),
                                                               dtype=np.int), value), axis=1)
        return array[mask]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str,
                        help="path to the stories")
    parser.add_argument('output_path', type=str, help="path to the output file")
    parser.add_argument('--pretrained_traj_path', type=str, help='Path to file containing array' \
                        'of "counts" of sentiment trajectories')
    parser.add_argument('--save_traj_path', type=str, help="path to store sentiment_trajectories of model." \
                        " By default is the same as pretrained_traj_path")
    parser.add_argument('--force_retrain', type=bool)
    args = parser.parse_args()

    # Load the stories, process them and compute their word embedding
    print('Loading stories according to parsing instructions: {}'.format(train_parsing_instructions))
    train_stories = load_stories(args.data_path + '/train_stories.csv', train_parsing_instructions)
    print("Stories loaded.")
    print('Loading stories according to parsing instructions: {}'.format(eval_parsing_instructions))
    eval_stories = load_stories(args.data_path + '/val_stories.csv', eval_parsing_instructions)
    print("Stories loaded.")

    if args.save_traj_path == None:
        save_traj_path = args.pretrained_traj_path
    else:
        save_traj_path = args.save_traj_path
    if args.pretrained_traj_path == None:
        args.pretrained_traj_path = ' '

    sentiment_analyzer = SentimentAnalyzer(sentiment_files_path_dict_cluster,
                                           sent_traj_counts_arrays_filepath=args.pretrained_traj_path,
                                           force_retrain= args.force_retrain,
                                           save_traj_path=save_traj_path)
    start = tm.datetime.now()
    sentiment_analyzer.train(train_stories[0:20000])
    print('Training time: \n{}' .format(tm.datetime.now() - start))
    print('Traj counts: {} \n Traj condensed counts: \n{}'\
            .format(sentiment_analyzer.sent_traj_counts_array,
                    sentiment_analyzer.sent_condensed_traj_counts_array))

    start = tm.datetime.now()
    print('Computing probabilities ...')
    proba_ending1, proba_ending2 = sentiment_analyzer.predict_proba(eval_stories)
    binary_features = sentiment_analyzer.generate_bin_features(proba_ending1, proba_ending2)
    # Compute the topic similarity between the endings and the context
    print(proba_ending1, proba_ending2)
    print("Done. Time for prediction: {}" .format(tm.datetime.now() - start))

    # Write the features to a .npz file
    np.savez_compressed(args.output_path,
                        sentiment_ending1=proba_ending1,
                        sentiment_ending2=proba_ending2,
                        binary_features=binary_features)
print("Sentiment features stored in " + str(args.output_path))

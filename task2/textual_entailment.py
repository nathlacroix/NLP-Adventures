import csv
import os
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import namedtuple

from allennlp.service.predictors import Predictor
from allennlp.models.archival import load_archive

Story = namedtuple('Story', ['context', 'endings'])

model_path = 'data/decomposable-attention-elmo-2018.02.19.tar.gz'
gpus = [int(d) for d in os.environ.get('CUDA_VISIBLE_DEVICES', '-1').split(',')]
labels = ['entailment', 'contradiction', 'neutral']
num_context_sentences = 4


def load_stories(filename):
    assert Path(filename).exists()
    stories = []
    with open(filename, 'r') as csvfile:
        csvfile.readline()  # get rid of the header
        stories = [Story(context=r[1:1+num_context_sentences], endings=r[5:7])
                   for r in csv.reader(csvfile, delimiter=',')]
    return stories


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help="path to the stories")
    parser.add_argument('output_path', type=str, help="path to the output file")
    parser.add_argument('--comparisons', dest='comparisons', action='store_true')
    parser.add_argument('--conditional', dest='conditional', action='store_true')
    args = parser.parse_args()

    print("Loading model...")
    archive = load_archive(
            model_path, weights_file=None, cuda_device=gpus[0], overrides="")
    predictor = Predictor.from_archive(archive, 'textual-entailment')

    print("Loading stories...")
    stories = load_stories(args.data_path)

    print("Computing textual entailment...")
    # Define the sentence index of the begining of the context
    context_indices = list(range(num_context_sentences)) if args.conditional else [0]
    probs = {i: {l: [] for l in labels} for i in context_indices}
    for s in tqdm(stories):
        # Merge endings and contexts in a single batch
        context = [' '.join(s.context[i:]) for i in context_indices]
        json_input = [{'premise': c, 'hypothesis': e}
                      for c in context for e in s.endings]
        pred = predictor.predict_batch_json(json_input)
        # Split into a dict according to the context index
        pred = dict(zip(context_indices, zip(pred[::2], pred[1::2])))
        for idx in context_indices:
            for i, l in enumerate(labels):
                probs[idx][l].append([p['label_probs'][i] for p in pred[idx]])
    probs = {l+str(i): np.array(probs[i][l]) for l in labels for i in context_indices}

    print("Computing features...")
    features = {k+'_'+str(i+1): v[:, i] for k, v in probs.items() for i in [0, 1]}
    if args.comparisons:
        comp = {k+'_comp': np.where(p[:, 0] > p[:, 1], 1, -1) for k, p in probs.items()}
        features.update(comp)

    # Write the features to a .npz file
    np.savez_compressed(args.output_path, **features)
    print("Text entailment features stored in " + str(args.output_path))

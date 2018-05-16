import numpy as np
import spacy
import argparse
import csv
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


def load_stories(filename):
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
            context = row[1] + " " + row[2] + " " + row[3] + " " + row[4]
            ending1 = row[5]
            ending2 = row[6]
            stories.append([context, ending1, ending2])
    return stories


def is_topic_word(word):
    return (word.pos_ == 'NOUN') or (word.pos_ == 'VERB')


def get_topic_words(doc):
    return [word for word in doc if is_topic_word(word)]


def compute_embedding(stories, nlp):
    """
    Argument: stories: a list of stories, each divided in three strings (context,
    ending1 and ending2)
              nlp: lang object of spacy

    Returns: a list of word embeddings of the topic words for each one of the
    three strings.
    """
    embeddings = []
    for story in stories:
        docs = [nlp(doc) for doc in story]
        embedding = [[w.vector for w in get_topic_words(doc)] for doc in docs]
        embeddings.append(embedding)
    return np.array(embeddings)


def compute_similarity(embeddings):
    def similarity_score(context, ending):
        sim_matrix = cosine_similarity(context, ending)
        sim_words = np.amax(sim_matrix, axis=0)
        return np.mean(sim_words)

    similarities = []
    for story in embeddings:
        sim1 = similarity_score(story[0], story[1])
        sim2 = similarity_score(story[0], story[2])
        binary_feature = int(sim1 <= sim2)
        similarities.append([sim1, sim2, binary_feature])
    return np.array(similarities)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str,
                        help="path to the stories")
    parser.add_argument('output_path', type=str, help="path to the output file")
    args = parser.parse_args()

    # Set up spacy
    nlp = spacy.load('en', disable=['parser', 'ner'])  # 'en_core_web_lg' is better

    # Load the stories, process them and compute their word embedding
    print("Loading stories...")
    stories = load_stories(args.data_path)
    print("Stories loaded.")
    print("Computing embeddings and topic similarity...")
    embeddings = compute_embedding(stories, nlp)

    # Compute the topic similarity between the endings and the context
    similarities = compute_similarity(embeddings)
    print("Similarities computed.")

    # Write the features to a .npz file
    np.savez_compressed(args.output_path,
                        topic_ending1=similarities[:, 0],
                        topic_ending2=similarities[:, 1],
                        binary_feature=similarities[:, 2])
    print("Topic consistency features stored in " + str(args.output_path))

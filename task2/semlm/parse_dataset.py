import csv
import argparse
from tqdm import tqdm
from pathlib import Path

from srl import Srl

num_endings = 2


def load_stories(filename, has_titles=True):
    assert Path(filename).exists()
    with open(filename, 'r') as csvfile:
        csvfile.readline()  # get rid of the header
        stories = [r[2 if has_titles else 1:]
                   for r in csv.reader(csvfile, delimiter=',')]
    return stories


def is_ascii(s):
    return len(s) == len(s.encode())


def remove_unicode(s):
    s_ascii = ''
    for c in s:
        if ord(c) > 127:
            if c == 'é':
                s_ascii += 'e'
        else:
            s_ascii += c
    return s_ascii


def parse_string(srl_pipeline, text,
                 add_period_tag=True, add_discourse_marker=True):
    index_predicate_token, index_pos_token = srl_pipeline.parse(text)
    components = [(i, {'type': 'frame', 'name': p, 'token': t})
                  for i, p, t in index_predicate_token]

    for i, tag, token in index_pos_token:
        # if add_period_tag and tag == '.':
            # components.append((i, {'type': 'punctuation', 'name': 'period'}))
        if add_discourse_marker and tag == 'CC':
            components.append((i, {'type': 'discourse', 'name': token}))

    # need_pos = add_period_tag or add_discourse_marker
    # if need_pos:
        # doc = spacy_pipeline(text)
        # print(doc)
        # for i, token in enumerate(doc):
            # print(token.tag_, token.lemma_)
            # if add_period_tag and token.tag_ == '.':
                # components.append((i, {'type': 'punctuation', 'name': 'period'}))
            # if add_discourse_marker and token.tag_ == 'CC':
                # components.append((i, {'type': 'discourse', 'name': token.lemma_}))

    components = sorted(components, key=lambda c: c[0])
    components = [c[1] for c in components]  # strip token index
    return components


def append_to_file(path, parsed):
    text = ' '.join([p['name'] for p in parsed])
    with open(path, 'a') as f:
        print(text, file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help="path to the stories")
    parser.add_argument('output_path', type=str, help="path to the output file")
    parser.add_argument('--has_endings', dest='has_endings', action='store_true')
    parser.add_argument('--has_titles', dest='has_titles', action='store_true')
    parser.add_argument('--srl_annotation', default='framenet', type=str)
    args = parser.parse_args()

    output_path = Path(args.output_path)
    if args.has_endings:
        endings_output_paths = [Path(output_path.parents[0], '{}_ending{}{}'.format(
            output_path.stem, i, output_path.suffix)) for i in range(num_endings)]

    print("Loading parsing pipeline...")
    srl_pipeline = Srl(args.srl_annotation)
    def parse(s):
        return parse_string(srl_pipeline, s)

    print("Loading stories...")
    stories = load_stories(args.data_path, args.has_titles)

    failures = 0
    for s in tqdm(stories):
        # jnius crashes with unicode characters
        if not is_ascii(' '.join(s)):
            s = [remove_unicode(sentence) for sentence in s]

        context_end = -num_endings-1 if args.has_endings else len(s)
        parsed_context = []
        try:
            for c in s[:context_end]:
                parsed_context.extend(parse(c))
                parsed_context.append({'type': 'punctuation', 'name': 'period'})
        except KeyError as e:
            print('SRL parsing failed: {}'.format(e))
            failures += 1
            continue
        append_to_file(output_path, parsed_context)

        if args.has_endings:
            for e, p in zip(s[context_end:-1], endings_output_paths):
                parsed_ending = parse(e)
                parsed_ending.append({'type': 'punctuation', 'name': 'period'})
                append_to_file(p, parsed_ending)

    print('Num failures: {}'.format(failures))

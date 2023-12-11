import argparse
import json
from collections import Counter
import itertools


def extract_vocab(iterable, top_k=None, start=0):
    """ Turns an iterable of list of tokens into a vocabulary.
        These tokens could be single answers or word tokens in questions.
    """
    all_tokens = itertools.chain.from_iterable(iterable)
    counter = Counter(all_tokens)
    if top_k:
        most_common = counter.most_common(top_k)
        most_common = (t for t, c in most_common)
    else:
        most_common = counter.keys()
    # descending in count, then lexicographical order
    tokens = sorted(most_common, key=lambda x: (counter[x], x), reverse=True)
    vocab = {t: i for i, t in enumerate(tokens, start=start)}
    return vocab


def main():
    problems = '/mnt/seagate12t/VQA/scienceqa/problems.json'
    problems = json.load(open(problems, 'r'))
    questions = []
    hints = []
    choices = []
    for p in problems.values():
        questions.append(p['question'])
        hints.append(p['hint'])
        choices.append(p['choices'])
    # process to split the question and hint
    questions = [q.split() for q in questions]
    hints = [h.split() for h in hints]
    # remove the ', . ?'
    questions = [[w.replace(',', '').replace('.', '').replace('?', '').lower() for w in q] for q in questions]
    hints = [[w.replace(',', '').replace('.', '').replace('?', '').lower() for w in h] for h in hints]

    question_vocab = extract_vocab(questions,)
    hints_vocab = extract_vocab(hints,)
    choice_vocab = extract_vocab(choices,)

    vocabs = {
        'question': question_vocab,
        'hint': hints_vocab,
        'choice': choice_vocab,
    }
    # with open('/mnt/seagate12t/VQA/scienceqa/vocab.json', 'w') as fd:
    #     json.dump(vocabs, fd)
    # with open('./datasets/assets/scienceqa_vocab.json', 'w') as fd:
    #     json.dump(vocabs, fd)

if __name__ == '__main__':
    main()

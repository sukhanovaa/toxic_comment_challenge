#replace random words with <unk> so as to help learn to generalize
import random
from tqdm import tqdm
from math import ceil
from nltk.corpus import wordnet as wn
from itertools import chain


def word_dropout(tokens, unk_token='<unk>', ratio=0.25):
    """
    Replaces random tokens with <unk>. Known to somewhat improve
        generalization of a model, left unused
    :param tokens: list of tokens
    :param ratio: approx. ratio of tokens to be replaced
    :return: string with unks
    """
    to_replace = set(random.choices(range(len(tokens)),
                                    k=int(len(tokens) * ratio)))
    return [unk_token if e in to_replace else tokens[e]
            for e, x in enumerate(tokens)]


def word_dropout_file(file, unk_token='<unk>', ratio=0.25):
    '''
    Performs replacing random tokens with <unk> on every line of a file.
    :param line: list of tokens
    :param ratio: approx. ratio of tokens to be replaced
    :return: string
    '''
    with open(file, 'r') as inp, open(file+'.unk', 'w') as out:
        for line in inp:
            replacement_result = word_dropout(line)
            out.write(' '.join(replacement_result) + '\n')


def add_synonyms(line, hash, ratio=0.5):
    '''

    :param line: line to process
    :param hash: dictionary for remembering synonyms sets
    :param ratio: a proportion of tokens to be replaced
    :return: string, hash
    '''
    def synonym_set(word):
        s = [x for x in set(chain.from_iterable([w.lemma_names()
                                                 for w in wn.synsets(word)]))
             if '_' not in x]
        if not len(s):
            s.append(word) #is this memoization?
        return s

    line = line.strip().split()
    long_tokens = [e for e,x in enumerate(line) if len(x) > 3]
    to_replace = set(random.choices(long_tokens,
                                    k=ceil(len(long_tokens)*ratio)))
    for e, t in enumerate(line):
        if e in to_replace:
            if t not in hash:
                hash[t] = synonym_set(t)
            line[e] = random.choice(hash[t])
    return ' '.join(line), hash


def synonym_augmentation(file):
    '''
    Try and replace tokens with their corresponding one-word synonyms independently
    (WordNet enables synonymous expressions as well)
    :param file: plain text, tokenized BUT not split by BPE
    :return: None, only writes a file
    '''
    synonyms = dict({})
    with open(file, 'r') as text, open(file+'.syn.txt', 'w') as out:
        for line in tqdm(text):
            new, synonyms = add_synonyms(line, synonyms)
            out.write(new + '\n')
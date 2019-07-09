#from string import punctuation

from tqdm import tqdm
from itertools import islice
import os


#id,"comment_text","toxic","severe_toxic","obscene","threat","insult","identity_hate"
# avg length of a toxic comment ~ 391 char


def take(n, generator, field = None):
    if field: #for csv
        return [x[field] for x in islice(generator, n)]
    else:
        return list(islice(generator, n))


def import_file(path, file_type = 'text', delimiter = None):
    inp = open(path, 'r')
    if file_type != 'text':
        from csv import DictReader
        if file_type == 'csv':
            handler = DictReader(inp)
            return handler
        elif file_type == 'tsv':
            handler = DictReader(inp, delimiter=delimiter)
            return handler
        else:
            print('Invalid type of input file')
    else:
        return inp


def merge_dataset(X_file, Y_file):
    '''
    During training on colaboratory, I used the file format which is common
    in NMT frameworks: input|||corresponding output
    '''
    with open(X_file, 'r') as x, open(Y_file, 'r') as y, open(X_file+'.merged', 'w') as out:
        for pair in zip(x, y):
            out.write(' |||'.join(map(str.strip, pair)) + '\n')
merge_dataset('/Users/alenasuhanova/PycharmProjects/toxic_comments/train/plain_add_syn/2009.shuffled.500000.txt',
              '/Users/alenasuhanova/PycharmProjects/toxic_comments/train/plain_add_syn/2009.shuffled.500000.txt.bert_predictions.txt')
merge_dataset('/Users/alenasuhanova/PycharmProjects/toxic_comments/train/plain_add_syn/train.plain_add_syn.txt',
              '/Users/alenasuhanova/PycharmProjects/toxic_comments/train/plain_add_syn/train.bert_predictions.txt')


def shuffle_data(dataset, descending=False):
    '''
    :param dataset: string
    :param descending: bool, whether to sort by length in descending order
    :return: generator of strings
    '''
    from subprocess import Popen, PIPE
    shuf = Popen('shuf', stdin=PIPE, stdout=PIPE)
    res, err = shuf.communicate(dataset.encode('utf8'))
    res = sorted(res.decode('utf8').split('\n'),
                 key=lambda x: len(x.split()) // 8, reverse=descending)
    return (x for x in res if len(x.strip()))


def shuffle_file(file, output_length=7, descending=False):
    with open(file, 'r') as inp:
        shuffled = shuffle_data(inp.read())
    with open(file+'.shuf', 'w') as out:
        out.write(shuffled)


def trycopy(path, destination_dir):
    #Colaboratory cannot handle big files, so sending them straight to drive was a bad idea
    #instead, I had to compress files and send them like this because all standard utils failed
    import gzip, os
    from tqdm import tqdm
    with tqdm(gzip.open(path, 'rb')) as infile:
        with gzip.open(os.path.join(destination_dir, os.path.split(path)[-1]), 'wb') as outfile:
            for line in infile:
                outfile.write(line)
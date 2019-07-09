import faiss 
import numpy as np
import json, csv
from source.utils import take

a = np.random.choice(10)


def construct_base_index(path, save_path):
    '''
    Construct index to perform k-nn search
    :param path: data to process
    :return:
    '''
    with open(path, 'r') as vecs:
        X = np.asarray([json.loads(line) for line in vecs], dtype='float32')

    nlist = 100
    d = 512  # USE embedding size
    quantizer = faiss.IndexFlatL2(d)  # the other index
    index = faiss.IndexIVFFlat(quantizer, d, nlist) #inner-product search by default

    index.train(X)
    index.add(X)
    faiss.write_index(index, save_path) #'USE2index.pkl'


def mode(array):
    '''
    Find the most frequent value among candidates
    :param array: candidates
    :return: most frequent value
    '''
    most = max(map(array.count, array))
    modes = set(filter(lambda x: array.count(x) == most, array))
    if set(array) - modes:
        return modes 
    else:
        return None #no true mode among candidates


def label_comments(index, batch, nn = 1): #nn is number of nearest neighbours
    '''
    Deprecated, used to label data on-the-fly
    '''
    from source.universal_encoder import encode_batch
    index.nprobe = 10
    embedded = encode_batch(batch)
    D, I = index.search(embedded, nn)
    return I


def search_and_label_txt(index_path, raw_path, Y_path, batch_size = 10000, from_line = None):
    with open(Y_path, 'r') as targets:#, open(index_path, 'r') as ind:
        labels = targets.readlines()
        print(len(labels))
        index = faiss.read_index(index_path)
        assert index.is_trained
        print('index loaded')
    with open(raw_path, 'r') as inputs, open(raw_path+'.labeled.txt', 'w') as out:
        if from_line:
            for i in range(from_line+1):
                next(inputs)
        batch = take(batch_size, inputs)
        count = 0
        while len(batch) > 0:
            count +=1
            search_res = label_comments(index=index, batch=batch)
            for row in search_res:
                out.write(labels[row[0]])
            if from_line:
                print('{}'.format(from_line + batch_size * count), end='\r')
            else:
                print('{}'.format(batch_size * count), end='\r')
            del search_res
            batch = take(batch_size, inputs)


def search_and_label_gzip(index_path, raw_path, Y_path, save_path, nn = 5,
                          batch_size = 10000, from_line = None):
    '''
    Using compressed vector data from Universal Encoder, propagate labels on
    the unlabeled data using most frequent value among its neighbours
    :param index_path: where the trained index at
    :param raw_path: original text to extract
    :param Y_path: original labels to extract
    :param nn: number of nearest neighbours to consider
    :param batch_size: batch size
    :param from_line: if the search failed previously, skip process up to this line
    :return: None, writes to file
    '''
    import gzip, json
    from itertools import islice

    def take_gzip(n, iterable):
        return np.array([json.loads(x) for x in list(islice(iterable, n))], dtype='float32')

    with open(Y_path, 'r') as targets:#, open(index_path, 'r') as ind:
        labels = targets.readlines()
        print(len(labels))
        index = faiss.read_index(index_path)
        assert index.is_trained
        print('index loaded')
    index.nprobe = 10
    with gzip.open(raw_path, 'rb') as inputs, open(save_path, 'w') as out:
        if from_line:
            for i in range(from_line+1):
                next(inputs)
        batch = take_gzip(batch_size, inputs)
        count = 0
        while len(batch) > 0:
            count +=1
            D, search_res= index.search(batch, nn)
            for row in search_res:
                candidate = mode([labels[row[i]] for i in range(nn)])
                if candidate:
                    out.write(list(candidate)[0])
                else:
                    out.write(labels[row[0]]) #closest match
            if from_line:
                print('{}'.format(from_line + batch_size * count), end='\r')
            else:
                print('{}'.format(batch_size * count), end='\r')
            del search_res
            batch = take_gzip(batch_size, inputs)


def toxic_out(file_data, file_labels):
    '''
    dummy snippet to divide texts into neutral / containing some toxicity
    :param file_data: path to a file with texts
    :param file_labels: path to a files with labels
    :return: none, writes to file
    '''
    with open(file_data) as inp_data, open(file_labels) as inp_labels, \
            open(file_data+'.neutral', 'w') as out_neutral, \
            open(file_data+'.neutral.labels', 'w') as out_neutral_labels, \
            open(file_data+'.toxic', 'w') as out_toxic, \
            open(file_data+'.toxic.labels', 'w') as out_toxic_labels:
        for lines in zip(inp_data, inp_labels):
            if '1' in lines[1]:
                out_toxic.write(lines[0])
                out_toxic_labels.write(lines[1])
            else:
                out_neutral.write(lines[0])
                out_neutral_labels.write(lines[1])
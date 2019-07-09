from collections import Counter
from mosestokenizer import MosesPunctuationNormalizer, MosesTokenizer
from string import punctuation
from source.utils import *
import unicodedata, re


# """
# GENERAL_PUNCTUATION is a dictionary of punctuation constants, extended to Asian
# punctuation marks, replaced by more uniform puntuation set
# (from MosesPunctuationNormalizer)
# """
# GENERAL_PUNCTUATION = {'`':'\'', r"'":'\'', '„':'"', '“':'"', '”':'\"', '–':'-', '—':" - ", '´':"'", '‘':'"', '‚':'"', '’':'"', "''":'"', '´´':'"', '…':'...', " « ":'"', "« ":'"', "«":'"', " » ":'"', " »": '"',"»":'"'," %":'%',"nº ":"nº "," :":":"," ;":";"," ?":"?"," !":"!","¿":"?","¡":"!","？":"?","！":"!","。":".","，":",","、":",","一":" - ","：":":","；":";","《":'"',"》":'"',"〈":'"',"〉":'"',"·":" ",'\u0002':' ', '\u0003':' ', '\u0004':' ', '\u0009':' ', '\u0017':' ', '\u001D':' ', '\u007F':' ', }


def ngrams_generator(tokenized, n=4):
    for x in range(1, n + 1):
        for k in zip(*(tokenized[i:] for i in range(x))):
            yield '_'.join(k)


def ngrams(text, n=4):
    return list(ngrams_generator(text, n))


def tokenizer_moses(text, column='comment_text'): #column for extracting from csv
    '''
    A proper wrapper for moses text preprocessing utilities,
    because they can't handle newlines
        text: string
        out: list
    '''
    result = []
    with MosesPunctuationNormalizer() as punct, MosesTokenizer('en') as tok:
        if column:
            texts = list(filter(None, text[column].lower().split('\n')))
        else:
            texts = text
        for t in texts:
            if len(t.strip()):
                norm = punct(t)
                tokens = tok(norm)
                result.extend(tokens)
    return result


def tokenizer_space(text):
    return text.split()


def count_ngrams(file, file_type='csv', tsv_delimiter=None):
    '''
    Text must be preprocessed beforehand, I used perl versions of moses utils
    count_ngrams_from_line was previously used on raw text but it's inefficient
    '''
    import multiprocessing as mp
    from tqdm import tqdm
    pool = mp.Pool(4)
    global_c = Counter()
    file_h = import_file(file, file_type=file_type, delimiter=tsv_delimiter)

    jobs = pool.imap_unordered(ngrams, file_h)
    for j in tqdm(jobs):
        global_c.update(j)
    pool.close()
    pool.join()
    with open(file+'.count', 'w') as out:
        for i in global_c.most_common():
            out.write(i[0] + '\t' + str(i[1]) + '\n')


def filter_punctuation(line):
    return line.translate(str.maketrans('', '', punctuation))




def clean_punct_after_moses(line):
    UNESCAPED = {'&amp;': ' and ',
                 '&#124;': ' ',
                 '&lt;': ' ',
                 '&qt;': ' ',
                 '&apos;': '’',
                 # this character is normalized to single quote char
                 # by moses, so tokenized version won't have it and
                 # the symbol stays unambiguous
                 '&quot;': ' ',
                 '&#91;': ' ',
                 '&#93;': ' ',
                 }
    PUNCT_NO_APOS = [x for x in punctuation if x != '’']
    PUNCT_TABLE = str.maketrans(''.join(PUNCT_NO_APOS),
                                ' ' * len(PUNCT_NO_APOS))

    line = line.strip().lower()
    for item in UNESCAPED:
        line = line.replace(item, UNESCAPED[item])
    line = line.translate(PUNCT_TABLE).split()
    line = ' '.join(line) #single spaces
    return line


def normalize_line(line):
    '''
    This is the order in which all the text was preprocessed to train/test
    transformers. BPE was used after
    '''
    line = tokenizer_moses(line)
    line = clean_punct_after_moses(line)
    return line


def normalize_to_ascii(line):
    #in the original raws texts, there's 13691 different characters
    return unicodedata.normalize('NFKD', line).encode('ascii', 'ignore')


def tokenize_with_vocabulary(line, vocabulary, pattern):
    '''
    Use regex to perform bpe-like tokenization
    :param line: line to process
    :param vocabulary: mapping from tokens to corresponding IDs
    :param pattern: regex
    :return: a list of IDs from the tokenized line
    '''
    parts = pattern.findall(line)
    ids = [vocabulary[x] for x in parts]
    return ids


def bpe_file(file, vocab, save_to):
    from tqdm.auto import tqdm
    vocab_dict = {}
    vocab_dict['[BOS]'] = 0
    vocab_dict['[EOS]'] = 1
    with open(vocab) as vocabulary_file:
        vocab_dict.update({line.strip().split()[0]: e+2
                           for e, line in enumerate(vocabulary_file)})
    # inside_pattern = '{}(?=[^\s])'
    # vocab_to_pattern = [inside_pattern.format(p[0].strip('@')) if '@' in p[0]
    #                     else p[0].strip('_')
    #                     for p in sorted(vocab_dict.items(),
    #                                     key=lambda l: len(l[0]), reverse=True)]
    # vocab_to_pattern = re.compile('|'.join(vocab_to_pattern))

    vocab_to_pattern = [p[0] for p in sorted(vocab_dict.items(),
                                        key=lambda l: len(l[0]), reverse=True)]
    vocab_to_pattern = re.compile('|'.join(vocab_to_pattern))
    with open(file) as f, open(save_to, 'w') as out:
        for line in tqdm(f):
            line = normalize_to_ascii(line.lower())
            out.write(' '.join(list(map(str, tokenize_with_vocabulary(line.lower(),
                                        vocab_dict, vocab_to_pattern))))
                      + '\n')
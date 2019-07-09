import os, gensim, logging, time

start = time.clock()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        _train_dir = [x for x in os.listdir(self.dirname) if 'bpe.txt' in x]
        print(_train_dir)
        for fname in _train_dir:
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

sentences = MySentences('/Users/alenasuhanova/PycharmProjects/toxic_comments/wiki_data')  # a memory-friendly iterator
model = gensim.models.Word2Vec(sentences, iter=50, size=300, workers=4, negative=15)

with open('wiki.cbow', 'wb') as out:
    model.save(out)

end = time.clock()
print(end-start)
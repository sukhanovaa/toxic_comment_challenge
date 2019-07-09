from torch.nn import functional as F
from torch import nn
from torch import optim
import torch
import math
import numpy as np
import os
import logging
from collections import namedtuple
from sklearn.metrics import roc_auc_score
from subprocess import Popen, PIPE


BOS_TOKEN = 0
EOS_TOKEN = 1


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        # to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(
                                    '%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)
        # to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(
                                                    '%(asctime)s:%(message)s'))
        logger.addHandler(stream_handler)


def take(iterable, start, length):
    return iterable[start:start+length]


def batch_generator(dataset, batch_size, mapping2idx, train: bool = True):
    '''
    Dumb epoch iterator. dataset_fh is a file stream
    Assumes that input and target features are separated by '|||'
    '''
    shuf = Popen('shuf', stdin=PIPE, stdout=PIPE)
    res, err = shuf.communicate(dataset.encode('utf8'))
    res = [x for x in sorted(res.decode('utf8').split('\n'),
                             key=lambda x: len(x.split()) // 7)
           if x.strip()]
    c = 0
    while True:
        if train: #take random position
            batch = take(res, np.random.choice(len(res)), batch_size)
        else:
            batch = take(res, c, batch_size)
            if not len(batch):
                break
            c += batch_size
        X, Y = [], []
        for b in batch:
            x, y = b.strip().split('|||')
            y = [int(v) for v in y.split()]
            Y.append(y)
            x = [mapping2idx[t] for t in x.split() if t in mapping2idx]
            #x.insert(0, mapping2idx[BOS_TOKEN])
            X.append(x)
        max_len = int(8 * np.ceil(max([len(x) for x in X]) / 8))
        for x in X:
            x.extend([EOS_TOKEN] * (max_len - len(x)))
        X = torch.from_numpy(np.asarray(X))
        Y = torch.from_numpy(np.asarray(Y, dtype='float32'))
        yield (X, Y)


def load_vectors(vecs_path: str, header: bool = False, vec_dim: int = 300):
    assert os.path.exists(vecs_path)
    word2idx = {}
    #word2idx[BOS_TOKEN] = 0
    #word2idx[EOS_TOKEN] = 1
    word2idx[EOS_TOKEN] = 0
    vecs = []
    with open(vecs_path, 'r') as raw:
        if header: #word2vec format
            num, dim = next(raw).split()  # header
        else: #glove
            dim = vec_dim
        #vecs.append([0] * int(dim))   # BOS
        vecs.append([0] * int(dim))   # EOS
        for idx, line in enumerate(raw):
            values = line.split()
            word2idx[values[0]] = idx + 2
            vecs.append([float(x) for x in values[1:]])
            assert len(vecs[-1]) == dim
    return word2idx, torch.tensor(vecs)


def load_vocab(vocab):
    assert os.path.exists(vocab)
    word2idx = {}
    word2idx[BOS_TOKEN] = 0
    word2idx[EOS_TOKEN] = 1
    with open(vocab, 'r') as raw:
        for idx, line in enumerate(raw):
            word2idx[line.strip()] = idx + 2
    return word2idx


def init_weights_he(m):
    # This might actually be considered redundant
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


# Most of the model is taken from
# https://github.com/konstantinosKokos/UniversalTransformer
# but with slight changes as to the ordering of layers
EncoderInput = namedtuple('EncoderInput', ['encoder_input', 'mask'])


class gelu(nn.Module): #from BERT
    def __init__(self):
        super(gelu, self).__init__()

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def ScaledDotProduct(queries, keys, values, mask = None):
    dk = keys.shape[-1]
    weights = torch.bmm(queries, keys.transpose(2, 1)) / math.sqrt(dk)
    if mask is not None:
        weights = weights.masked_fill_(mask == 0, value=-1e10)
    weights = F.softmax(weights, dim=-1)
    return torch.bmm(weights, values)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int, d_k: int, d_v: int):
        super(MultiHeadAttention, self).__init__()
        self.q_transformations = nn.ModuleList([nn.Linear(in_features=d_model,
                                                          out_features=d_k,
                                                          bias=False)
                                                for _ in range(num_heads)])
        self.k_transformations = nn.ModuleList([nn.Linear(in_features=d_model,
                                                          out_features=d_k,
                                                          bias=False)
                                                for _ in range(num_heads)])
        self.v_transformations = nn.ModuleList([nn.Linear(in_features=d_model,
                                                          out_features=d_v,
                                                          bias=False)
                                                for _ in range(num_heads)])
        self.Wo = nn.Linear(in_features=num_heads * d_v, out_features=d_model,
                            bias=False)

    def forward(self, queries, keys, values, mask = None):
        qs = [qt(queries) for qt in self.q_transformations]
        ks = [kt(keys) for kt in self.k_transformations]
        vs = [vt(values) for vt in self.v_transformations]
        outputs = [ScaledDotProduct(qs[i], ks[i], vs[i], mask) for i in
                   range(len(qs))]
        outputs = torch.cat(outputs, dim=-1)
        return self.Wo(outputs)


def PE(b: int, n: int, d_inp: int, d_model: int, freq: int = 10000):
    pe = torch.zeros(n, d_model)
    position = torch.arange(0, n, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_inp, 2, dtype=torch.float) *
                         -(math.log(freq) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    # return pe.unsqueeze(0).expand(b, n, d_model)
    return pe.repeat(b, 1, 1)


def PT(b: int, t: int, n: int, d_inp: int, d_model: int, freq: int = 10000):
    pe = torch.zeros(n, d_model)
    position = torch.arange(0, n, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_inp, 2, dtype=torch.float) *
                         - (math.log(freq) / d_model))
    times = torch.arange(0, t, dtype=torch.float).unsqueeze(1)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.repeat(t, 1, 1)

    pe[:, :, 0::2] = pe[:, :, 0::2] + torch.sin(times *
                                div_term).unsqueeze(1).expand(t, n, d_inp//2)
    pe[:, :, 1::2] = pe[:, :, 1::2] + torch.cos(times *
                                div_term).unsqueeze(1).expand(t, n, d_inp//2)
    return pe.unsqueeze(1).expand(t, b, n, d_model)


class FFN(nn.Module):
    def __init__(self, d_intermediate: int, d_model: int) -> None:
        super(FFN, self).__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(normalized_shape=d_model),
            nn.Linear(in_features=d_model, out_features=d_intermediate),
            gelu(),
            nn.Linear(in_features=d_intermediate, out_features=d_model)
        )

    def forward(self, x):
        return self.network(x)


class EncoderLayer(nn.Module):
    def __init__(self, num_heads: int, d_model: int, d_k: int, d_v: int,
                 d_intermediate: int, dropout: float) -> None:
        super(EncoderLayer, self).__init__()
        self.ln_attn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.mha = MultiHeadAttention(num_heads, d_model, d_k, d_v)
        self.ffn = FFN(d_intermediate, d_model)
        self.ln_ffn = nn.LayerNorm(d_model)

    def forward(self, x: EncoderInput) -> EncoderInput:
        n_in = x.encoder_input.shape[1]
        attn = self.ln_attn(x.encoder_input)
        attn = self.mha(attn, attn, attn, x.mask[:, :n_in])
        attn = self.dropout(attn)
        attn = x.encoder_input + attn

        result = self.ln_ffn(attn)
        result = self.ffn(result)
        result = self.dropout(result)
        result = attn + result
        return EncoderInput(encoder_input=result, mask=x.mask)


class RecurrentEncoder(nn.Module):
    def __init__(self, num_steps: int, num_heads: int, d_model: int, d_k: int,
                 d_v: int, d_intermediate: int, dropout=0.1) -> None:
        super(RecurrentEncoder, self).__init__()
        self.layer = EncoderLayer(num_heads, d_model, d_k, d_v, d_intermediate,
                                  dropout)
        self.num_steps = num_steps

    def forward(self, x: EncoderInput) -> EncoderInput:
        b, n, dk = x.encoder_input.shape
        pt = PT(b, self.num_steps, n, dk, dk)
        for i in range(self.num_steps):
            x = self.layer(EncoderInput(encoder_input=x.encoder_input + pt[i],
                                        mask=x.mask[:, :n]))
        return x


class UniversalTransformer(nn.Module):
    def __init__(self, vocab_size: int = 0,
                 pretrained_weights: np.array = None,
                 encoder_layers: int = 3, encoder_heads: int = 8,
                 d_model: int = 300, d_intermediate: int = 1024,
                 dropout: float = 0.5, classes: int = 6,
                 freeze_emb: bool = False):
        self.num_classes = classes
        super(UniversalTransformer, self).__init__()
        if pretrained_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_weights)
            self.d_model = pretrained_weights.shape[1]
        else:  # this is not actually intended to be used
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.d_model = d_model
        if freeze_emb:
            self.embedding.weight.requires_grad = False
        self.encoder = RecurrentEncoder(num_steps=encoder_layers,
                                        num_heads=encoder_heads,
                                        d_model=self.d_model,
                                        d_k=self.d_model // encoder_heads,
                                        d_v=self.d_model // encoder_heads,
                                        dropout=dropout,
                                        d_intermediate=d_intermediate)
        # self.ln = nn.LayerNorm(normalized_shape=self.d_model).to(self.device)
        self.classifier = nn.Sequential(
                            nn.LayerNorm(normalized_shape=self.d_model),
                            nn.Linear(self.d_model, self.num_classes),
                            nn.Sigmoid())
        self.apply(init_weights_he)

    def forward(self, encoder_input):
        embedded = self.embedding(encoder_input)
        encoder_output = self.encoder(EncoderInput(
            encoder_input=embedded,
            mask=(encoder_input != EOS_TOKEN).unsqueeze(-1)))

        # pooled = torch.max(encoder_output.encoder_input, dim=1)[0]
        pooled = torch.sum(encoder_output.encoder_input *
                           encoder_output.mask.to(dtype=torch.float),
                           dim=1) / encoder_output.encoder_input.shape[1]
        logits = self.classifier(pooled)
        return logits


class StackedEncoder(nn.Module):
    def __init__(self, num_layers: int, num_heads: int, d_model: int, d_k: int,
                 d_v: int, d_intermediate: int, dropout=0.1):
        super(StackedEncoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(num_heads, d_model, d_k, d_v,
                                                  d_intermediate, dropout)
                                     for x in range(num_layers)])

    def forward(self, x: EncoderInput) -> EncoderInput:
        b, n, dk = x.encoder_input.shape
        for layer in self.layers:
            x = layer(EncoderInput(encoder_input=x.encoder_input,
                                   mask=x.mask[:, :n]))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size: int = 0, pretrained_weights = None,
                 encoder_layers: int = 6, encoder_heads: int = 8,
                 d_model: int = 300, d_intermediate: int = 1024,
                 dropout: float = 0.5, classes: int = 6,
                 freeze_emb: bool = False):
        self.num_classes = classes
        super(Transformer, self).__init__()
        if pretrained_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_weights)
            self.d_model = pretrained_weights.shape[1]
        else:  # this is not actually intended to be used
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.d_model = d_model
        if freeze_emb:
            self.embedding.weight.requires_grad = False
        self.encoder = StackedEncoder(num_layers=encoder_layers,
                                      num_heads=encoder_heads,
                                      d_model=d_model,
                                      d_k=self.d_model // encoder_heads,
                                      d_v=self.d_model // encoder_heads,
                                      d_intermediate=d_intermediate,
                                      dropout=dropout)
        self.classifier = nn.Sequential(
                            nn.LayerNorm(normalized_shape=self.d_model),
                            nn.Linear(self.d_model, self.num_classes),
                            nn.Sigmoid())
        self.apply(init_weights_he)

    def forward(self, encoder_input):
        embedded = self.embedding(encoder_input)
        b, n, dk = embedded.shape
        pe = PE(b, n, dk, dk)
        encoder_output = self.encoder(
            EncoderInput(encoder_input=embedded + pe,
                         mask=(encoder_input != EOS_TOKEN).unsqueeze(-1)))

        # pooled = torch.max(encoder_output.encoder_input, dim=1)[0]
        pooled = torch.sum(encoder_output.encoder_input *
                           encoder_output.mask.to(dtype=torch.float),
                           dim=1) / encoder_output.encoder_input.shape[1]
        logits = self.classifier(pooled)
        return logits


def load_model(model_dir, checkpoint_no):
    assert os.path.exists(model_dir)
    saved_checkpoints = [os.path.join(model_dir, x)
                         for x in os.listdir(model_dir)
                         if x.endswith('.pth')]
    if checkpoint_no in ['last', -1, '-1']: #last
        saved_checkpoints.sort(key=os.path.getmtime, reverse=True)
        checkpoint = torch.load(saved_checkpoints[0])
        print('checkpoint {} loaded'.format(saved_checkpoints[0]))
    else:
        to_load = [x for x in saved_checkpoints if str(checkpoint_no) in x]
        to_load.sort()
        to_load = to_load[0]
        assert os.path.exists(to_load)
        print(to_load)
        checkpoint = torch.load(to_load)
        print('checkpoint {} loaded'.format(to_load))
    return checkpoint['model_state_dict'], checkpoint['optimizer_state_dict'], \
           checkpoint['scheduler_state_dict'], checkpoint['epoch']


def run_val(model, test_set, batch_size, word2idx_mapping, criterion, device,
            output_path: str=None):
    model.eval()
    with torch.no_grad():
        Y_computed = []
        Y_true = []
        Y_pred = []
        val_loss = 0.
        batch_count = 0
        with open(test_set, 'r') as inp:
            test = inp.read()
        batcher = batch_generator(test, batch_size, word2idx_mapping,
                                  train=False)
        try:
            for data in batcher:
                x_test_batch, y_test_batch = data
                y_pred_batch = model(x_test_batch.to(device))
                loss = criterion(y_pred_batch, y_test_batch.to(device))
                Y_computed.extend(y_pred_batch.detach().cpu().numpy())
                Y_pred.extend(torch.round(y_pred_batch.detach()).cpu().numpy())
                Y_true.extend(y_test_batch.cpu().numpy())
                val_loss += loss.item()
                batch_count +=1
        except StopIteration:
            pass
        Y_computed = np.array(Y_computed)
        Y_true = np.array(Y_true)
        # Y_pred = np.array(Y_pred)
        # weighted_prf = precision_recall_fscore_support(
        #                  y_pred=Y_pred, y_true=Y_true, average='weighted')
        # print(weighted_prf)
        # micro_prf = precision_recall_fscore_support(
        #                 y_pred=Y_pred, y_true=Y_true, average='micro')
        # print(micro_prf)
        # macro_prf = precision_recall_fscore_support(
        #                 y_pred=Y_pred, y_true=Y_true, average='macro')
        # print(macro_prf)
        val_loss = val_loss / batch_count
        roc_auc = np.sum([roc_auc_score(Y_true[:, i], Y_computed[:, i])
                          for i in range(6)])/6
        print(roc_auc)
    if output_path:
        with open(output_path, 'wb') as out:
            np.save(output_path, np.asarray(Y_computed))
    #return (weighted_prf, micro_prf, macro_prf), val_loss, roc_auc
    return val_loss, roc_auc


def train(model_dir: str, train_set: str, test_set: str, device: str = 'cpu',
          continue_from_last: str = None, batch_size: int = 64,
          num_epochs: int = 10, pretrained_vectors: str = None,
          vocab_file: str = None, report_every=5000, validate_every=500):
    # I assume that there was BPE performed on the dataset beforehand
    # (also embeddings were pretrained),
    # so either a vocabulary file (from BPE) or the whole vector model is available
    assert pretrained_vectors or vocab_file
    if pretrained_vectors:
        word2idx_mapping, weights = load_vectors(pretrained_vectors)
    else:
        word2idx_mapping, weights = load_vocab(vocab_file), None

    model = UniversalTransformer(vocab_size=len(word2idx_mapping),
                                 pretrained_weights=weights)
    criterion = nn.BCELoss()  # this or CrossEntropy?
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.01)
    last_epoch = 0
    if continue_from_last:
        assert os.path.exists(continue_from_last)
        saved_checkpoints = [os.path.join(model_dir, x)
                             for x in os.listdir(model_dir)
                             if x.endswith('chkp')]
        checkpoint = torch.load(saved_checkpoints.sort(
                                key=os.path.getmtime, reverse=True)[0])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        last_epoch = checkpoint['epoch']

    model = model.to(device)

    for epoch in range(last_epoch, num_epochs):
        running_loss = 0.0
        with open(train_set, 'r') as inp:
            data = inp.read()
            for i, data in enumerate(batch_generator(data, batch_size,
                                                     word2idx_mapping)):
                optimizer.zero_grad()
                inputs, labels = data
                outputs = model(inputs.to(device))
                loss = criterion(outputs, labels.to(device))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % report_every == 0 and i > 0:  # print every 500 mini-batches
                    logging.log(logging.INFO,
                        'Epoch {}, step {}: loss = {}'.format(epoch + 1, i,
                                                              running_loss / 500))
                    running_loss = 0.0
                if i % validate_every == 0 and i > 0:
                    model.eval()
                    val_loss, roc_auc_score = run_val(model, test_set,
                                               criterion=criterion,
                                               batch_size=batch_size,
                                               word2idx_mapping=word2idx_mapping)
                    logging.log(logging.INFO,
                        'Validation on step {}, scores: {}'.format(i, roc_auc_score))
                    logging.log(logging.INFO,
                        'Validation loss: {}'.format(val_loss))
                    model.train()
                scheduler.step(epoch)

        torch.save({'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    },
                   os.path.join(model_dir, 'epoch{}.chkp'.format(epoch)))
        logging.log(logging.INFO, 'Saved on epoch {}'.format(epoch))

    print('done training')


def validate(model_type, model_dir, checkpoint_no, test_file, vectors,
             batch_size,
             criterion=nn.BCELoss(), device='cuda'):
    mapping = load_vectors(vectors)[0]
    model = model_type(vocab_size=len(mapping)).to(device)
    model.load_state_dict(load_model(model_dir, checkpoint_no)[0])

    loss, roc_auc = run_val(model, test_file, batch_size, mapping, criterion,
                            device=device)
    logging.log(logging.INFO, 'Test ROC-AUC: {}'.format(roc_auc))


def main():
    set_logger('/Users/alenasuhanova/PycharmProjects/toxic_comments/models/basic/log.txt')
    train(model_dir='/Users/alenasuhanova/PycharmProjects/toxic_comments/models/basic',
          train_set='/Users/alenasuhanova/PycharmProjects/toxic_comments/train_classifier/train.merged.shuffled',
          test_set='/Users/alenasuhanova/PycharmProjects/toxic_comments/train_classifier/test.merged',
          pretrained_vectors='/Users/alenasuhanova/PycharmProjects/toxic_comments/wiki.cbow.w2v',
          batch_size=1)

#main()
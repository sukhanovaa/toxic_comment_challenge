import logging, os, re, random
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, RandomSampler, \
    SequentialSampler
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
from unicodedata import normalize

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)


class BPERegexTokenizer:
    def __init__(self, vocab, max_seq_len):
        self.vocab = vocab
        self.pattern = re.compile('|'.join([p[0] for p in
                                            sorted(vocab.items(),
                                                   key=lambda l: len(l[0]),
                                                   reverse=True)]))
        self.max_seq_len = max_seq_len  # BOS, EOS

    def __len__(self):
        return len(self.vocab)

    def tokenize(self, text):
        text = normalize('NFKD', text).encode('ascii', 'ignore').decode('utf8')
        text = text.strip().lower()
        parts = self.pattern.findall(text)
        ids = [self.vocab[x] for x in parts]
        if len(ids) > self.max_seq_len - 2:
            ids = ids[:self.max_seq_len - 2]
        ids = [self.vocab['[BOS]']] + ids
        ids += [self.vocab['[EOS]']] * (self.max_seq_len - len(ids))
        return ids


class BlendDataset(Dataset):
    def __init__(self, data, tokenization_method, labeled=True,
                 predicted=True):
        self.tokenize = tokenization_method
        self.examples = []
        self.labels = []
        self.labeled = labeled
        self.predictions = []
        self.predicted = predicted

        for d in data:
            if predicted:
                x, p = d.strip().rsplit('|||', 1)
                self.predictions.append(list(map(float, p.split())))
                d = x
            if labeled:
                x, y = d.strip().rsplit('|||', 1)
                self.labels.append(list(map(float, y.split())))
                d = x

            x = self.tokenize(d.strip())
            self.examples.append(x)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        items = [torch.tensor(self.examples[index])]
        if self.labeled:
            items.append(torch.tensor(self.labels[index]))
        if self.predicted:
            items.append(torch.tensor(self.predictions[index]))
        return tuple(items)


def get_loader(datapath, tokenization, batch_size, random_order, labeled,
               predicted):
    with open(datapath) as file:
        rows = file.readlines()
    dataset = BlendDataset(rows, tokenization_method=tokenization,
                           labeled=labeled, predicted=predicted)
    if random_order:
        return DataLoader(dataset, batch_size=batch_size,
                          sampler=RandomSampler(dataset))
    else:
        return DataLoader(dataset, batch_size=batch_size,
                          sampler=SequentialSampler(dataset))


def meta_dataloader(loader_labeled, loader_unlabeled):
    """
    Костыли, чтобы переключаться между размеченными и неразмеченными данными
    """
    labeled_iterator = iter(loader_labeled)
    unlabeled_iterator = iter(loader_unlabeled)
    for i in range(int(len(loader_labeled) + len(loader_unlabeled))):
        if random.random() < 0.5:
            try:
                yield next(labeled_iterator)
            except StopIteration:
                labeled_iterator = iter(loader_labeled)
                yield next(labeled_iterator)
        else:
            try:
                yield next(unlabeled_iterator)
            except StopIteration:
                unlabeled_iterator = iter(loader_unlabeled)
                yield next(unlabeled_iterator)


def train_single(model, train_data, test_data, model_dir, tokenizer,
                 batch_size=64, learning_rate=1e-5, num_epochs=5,
                 report_every=1000, device='cuda'):
    set_logger(os.path.join(model_dir, 'log.txt'))

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    loss_history = []
    model.train()

    loader_labeled = get_loader(train_data, tokenization=tokenizer.tokenize,
                                batch_size=batch_size,
                                random_order=True,
                                labeled=True, predicted=True)

    for e in range(num_epochs):
        for i, batch in enumerate(loader_labeled):
            optimizer.zero_grad()
            #train_ex, labels, predictions = batch
            train_ex, labels = batch
            output = model(train_ex.to(device))
            loss = criterion(output, labels.to(device))
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            if i > 0 and i % report_every == 0:
                logging.info('loss {}'.format(loss.item()))

        plt.plot(list(range(len(loss_history))), loss_history)
        plt.show()

        output_model_file = os.path.join(model_dir,
                                         "blendcnn_epoch{}.bin".format(e))
        torch.save({'epoch': e + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'scheduler_state_dict': scheduler.state_dict(),
                    }, output_model_file)
        # validate each epoch
        validate(model, test_data, batch_size, tokenizer.tokenize,
                 device='cuda')


def loss_kd(outputs, teacher_outputs, alpha, T, num_classes, labels=None):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    """
    if labels is not None:
        div_loss = torch.stack([F.kl_div(F.log_softmax(outputs[:, i] / T),
                                         F.softmax(teacher_outputs[:, i] / T))
                                for i in range(num_classes)])
        student_loss = F.binary_cross_entropy_with_logits(outputs,
                                                          labels)
        KD_loss = alpha * T * T * div_loss.mean() + (1. - alpha) * student_loss
    else:
        div_loss = torch.stack([F.kl_div(F.log_softmax(outputs[:, i] / T),
                                         F.softmax(teacher_outputs[:, i] / T))
                                for i in range(num_classes)])
        KD_loss = div_loss.mean()
    return KD_loss


def train_KD(model, train_data, test_data, model_dir, tokenizer,
          alpha, T, unlabeled_data=None, batch_size=64,
          learning_rate=1e-5, num_epochs=5, report_every=1000, device='cuda'):
    set_logger(os.path.join(model_dir, 'log.txt'))

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    loss_history = []
    model.train()

    loader_labeled = get_loader(train_data, tokenization=tokenizer.tokenize,
                                batch_size=batch_size,
                                random_order=True,
                                labeled=True, predicted=True)
    if unlabeled_data:
        loader_unlabeled = get_loader(unlabeled_data,
                                      tokenization=tokenizer.tokenize,
                                      batch_size=batch_size,
                                      random_order=True,
                                      labeled=False, predicted=True)
        data = meta_dataloader(loader_labeled, loader_unlabeled)
    else:
        data = loader_labeled

    for e in range(num_epochs):
        for i, batch in enumerate(data):
            optimizer.zero_grad()
            if len(batch) == 3:
                train_ex, labels, predictions = batch
                labels = labels.float().to(device)
            else:
                train_ex, predictions = batch
                labels = None
            output = model(train_ex.to(device))
            loss = loss_kd(output, predictions.to(device), alpha, T,
                           output.shape[1], labels=labels)
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            if i > 0 and i % report_every == 0:
                logging.info('loss {}'.format(loss.item()))

        plt.plot(list(range(len(loss_history))), loss_history)
        plt.show()

        output_model_file = os.path.join(model_dir,
                                         "blendcnn_{}x{}_epoch{}.bin".format(
                                             alpha, T, e))
        torch.save({'epoch': e + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'scheduler_state_dict': scheduler.state_dict(),
                    }, output_model_file)
        #validate each epoch
        validate(model, test_data, batch_size, tokenizer.tokenize,
                 device='cuda')


def construct_char_dict(vocab):
    # pad is 0, 1
    # from string import printable as CNN_VOCAB
    dict2ids = {x: i + 2 for i, x in enumerate(vocab)}
    dict2ids['[BOS]'] = 0
    dict2ids['[EOS]'] = 1
    return dict2ids


def construct_vocab_from_file(path):
    vocab_dict = {'[BOS]': 0, '[EOS]': 1}
    with open(path) as inp:
        vocab_dict.update({line.strip().split()[0]: e + 2
                           for e, line in enumerate(inp)})
    return vocab_dict


# def load_model(model_dir, checkpoint_no):
#     assert os.path.exists(model_dir)
#     saved_checkpoints = [os.path.join(model_dir, x)
#                          for x in os.listdir(model_dir)
#                          if x.endswith('.pth')]
#     if checkpoint_no in ['last', -1, '-1']:  # last
#         saved_checkpoints.sort(key=os.path.getmtime, reverse=True)
#         checkpoint = torch.load(saved_checkpoints[0])
#         print('checkpoint {} loaded'.format(saved_checkpoints[0]))
#     else:
#         to_load = [x for x in saved_checkpoints if str(checkpoint_no) in x]
#         to_load.sort()
#         to_load = to_load[0]
#         assert os.path.exists(to_load)
#         print(to_load)
#         checkpoint = torch.load(to_load)
#         print('checkpoint {} loaded'.format(to_load))
#     return checkpoint['model_state_dict'], checkpoint['optimizer_state_dict'], \
#            checkpoint['scheduler_state_dict'], checkpoint['epoch']
def load_model(model_path):
    assert os.path.exists(model_path)
    checkpoint = torch.load(model_path)
    return checkpoint


def validate_checkpoint(model, checkpoint_path, test_path, vocab_path,
                        batch_size,
                        max_seq_len=200, device='cuda'):
    vocab = construct_vocab_from_file(vocab_path)
    bpe = BPERegexTokenizer(vocab, max_seq_len=max_seq_len)

    checkpoint = load_model(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    validate(model, test_path, batch_size, bpe.tokenize, device=device)


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


def validate(current_model, test_data, batch_size, tokenization,
             device='cuda'):
    current_model.eval()

    criterion = nn.BCEWithLogitsLoss()
    all_logits, all_labels = None, None
    overall_loss = 0.
    for batch in get_loader(test_data, batch_size=batch_size,
                            tokenization=tokenization, random_order=False,
                            labeled=True, predicted=False):
        test_ex, labels = batch
        with torch.no_grad():
            output = current_model(test_ex.to(device))
            loss = criterion(output, labels.to(device))
            overall_loss += loss.item()

        if all_logits is None:
            all_logits = output.detach().cpu().numpy()
        else:
            all_logits = np.concatenate(
                (all_logits, output.detach().cpu().numpy()), axis=0)

        if all_labels is None:
            all_labels = labels.detach().cpu().numpy()
        else:
            all_labels = np.concatenate(
                (all_labels, labels.detach().cpu().numpy()), axis=0)

    eval_loss = overall_loss / all_labels.shape[0]
    roc_auc = np.mean([roc_auc_score(all_labels[:, i], all_logits[:, i])
                       for i in range(6)])
    result = {'eval_loss': eval_loss,
              'roc_auc': roc_auc}
    logging.info('eval loss {}, roc_auc {}'.format(result['eval_loss'],
                                                   result['roc_auc']))
    return result

if __name__ == '__main__':
    from .modeling import BlendCNN
    vocab = construct_vocab_from_file(
        'drive/My Drive/toxic/sets/kd/2010-2015.simplified_punct.vocab')
    bpe = BPERegexTokenizer(vocab, max_seq_len=200)

    labeled_data = 'drive/My Drive/toxic/sets/kd/train.plain_add_syn.pred.txt'
    unlabeled_data = 'drive/My Drive/toxic/sets/kd/2009.shuffled.500000.pred.txt'
    test_data = 'drive/My Drive/toxic/sets/kd/dev.plain_add_syn.txt'

    model = BlendCNN(num_labels=6,
                     channel_size=100,  # [100]*8,
                     num_channels=10,
                     vocab_length=len(bpe),
                     emb_dim=300,  #
                     kernel_size=5,
                     n_hidden_dense=1024,
                     dropout_prob=0.6)

    save_dir = 'drive/My Drive/toxic/models/kd/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    train_KD(model,
             labeled_data,
             test_data,
             save_dir,
             bpe,
             alpha=0.5,
             T=1,
             unlabeled_data=unlabeled_data,
             learning_rate=1e-6,
             num_epochs=3)
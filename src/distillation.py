import random, os, logging
import re
from string import punctuation
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, \
    SequentialSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLr
from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
import numpy as np
from sklearn.metrics import roc_auc_score

from string import printable as ascii_vocab
from unicodedata import normalize


def construct_char_dict(vocab):
    # pad is 0, 1
    dict2ids = {x: i + 2 for i, x in enumerate(vocab)}
    dict2ids['[BOS]'] = 0
    dict2ids['[EOS]'] = 1
    return dict2ids


# def pretokenize(line):
#     replace_punct = {'`': '\'', r"'": '\'', '„': '"', '“': '"', '”': '\"',
#                      '–': '-', '—': " - ", '´': "'", '‘': '"', '‚': '"',
#                      '’': '"', "''": '"', '´´': '"', '…': '...', " « ": '"',
#                      "« ": '"', "«": '"', " » ": '"', " »": '"', "»": '"',
#                      " %": '%', "nº ": "nº ", " :": ":", " ;": ";", " ?": "?",
#                      " !": "!", "¿": "?", "¡": "!", "？": "?", "！": "!",
#                      "。": ".", "，": ",", "、": ",", "一": " - ", "：": ":",
#                      "；": ";", "《": '"', "》": '"', "〈": '"', "〉": '"',
#                      "·": " ",
#                      '\u0002': ' ', '\u0003': ' ', '\u0004': ' ',
#                      '\u0009': ' ', '\u0017': ' ', '\u001D': ' ',
#                      '\u007F': ' ', }
#     line = line.lower()
#     for r in replace_punct:
#         if r in line:
#             line = line.replace(r, replace_punct[r])
#     clean_punct = str.maketrans(''.join(punctuation), ' ' * len(punctuation))
#     no_unicode = normalize('NFKD', line).encode('ascii', 'ignore').decode(
#         'utf8')
#     if not len(no_unicode.translate(clean_punct).strip()):
#         return None
#     else:
#         return no_unicode.translate(clean_punct)
#
#
# class BPERegexTokenizer:
#     def __init__(self, vocab, max_seq_length=None):
#         word_boundary = r'{}[?:^\s]'
#         self.map = vocab
#         self.pattern = re.compile('|'.join(
#             re.escape(word_boundary.format(p[0])) if '@' in p
#             else re.escape(p[0]) for p in
#             sorted(vocab.items(), key=lambda l: len(l[0]), reverse=True)))
#         self.max_seq_length = max_seq_length
#
#     def tokenize(self, string):
#         string = pretokenize(string)
#         if not string:
#             # raise ValueError('the string makes no sense')
#             return None
#         parts = self.pattern.findall(string)
#         ids = [self.map[x] for x in parts]
#         if self.max_seq_length:
#             if len(ids) > self.max_seq_length - 2:
#                 ids = ids[:(self.max_seq_length - 2)]
#             ids = [self.map['[BOS]']] + ids
#             ids += [self.map['[EOS]']] * (self.max_seq_length - len(ids))
#         return ids

def construct_subword_vocab(path):
    vocab_dict = {}
    vocab_dict['[BOS]'] = 0
    vocab_dict['[EOS]'] = 1
    with open(path) as inp:
        vocab_dict.update({line.strip().split()[0]: e + 2
                           for e, line in enumerate(inp)})
    return vocab_dict


class BPERegexTokenizer:
    def __init__(self, vocab):
        self.map = vocab
        self.pattern = re.compile('|'.join([p[0] for p in sorted(vocab.items(),
                                                                 key=lambda
                                                                     l: len(
                                                                     l[0]),
                                                                 reverse=True)]))

    def tokenize(self, text):
        text = text.strip().lower()
        text = normalize('NFKD', text).encode('ascii', 'ignore').decode('utf8')
        parts = self.pattern.findall(text)
        ids = [self.map[x] for x in parts]
        return ids


class BlendDataset(Dataset):
    def __init__(self, examples, map_to_ids, tokenization_metod,
                 max_seq_length=216, predicted=False, labeled=False):
        '''
        Ugly data structure
        :param examples: iterator containing dataset rows
        :param max_seq_length: maximum length supported by model
        :param map_to_ids: a vocabulary used to fetch padding from
        :param tokenize_metod: means to tokenize the training sequences
        :param predicted: whether it contains teacher model predictions (false for test)
        :param labeled: whether it contains true labels (false for unlabeled)
        '''
        self.examples = []
        self.teacher_preds = []
        self.labels = []
        self.labeled = labeled
        self.predicted = predicted
        for e in examples:
            # original ||| labels ||| predicted
            # x, *y = e.strip().rsplit('|||', 2)
            if labeled and predicted:
                x, y, p = e.strip().rsplit('|||', 2)
                self.labels.append(list(map(float, y.split())))
                self.teacher_preds.append(list(map(float, p.split())))
            elif predicted:
                x, p = e.strip().rsplit('|||', 1)
                self.teacher_preds.append(list(map(float, p.split())))
            elif labeled:
                x, y = e.strip().rsplit('|||', 1)
                self.labels.append(list(map(float, y.split())))
            else:
                x = e.strip()
            ids = tokenization_metod(x)
            if len(ids) > max_seq_length - 2:
                ids = ids[:max_seq_length - 2]
            ids = [map_to_ids['[BOS]']] + ids
            ids += [map_to_ids['[EOS]']] * (max_seq_length - len(ids))
            self.examples.append(ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        if self.labeled and self.predicted:
            return (torch.tensor(self.examples[index]),
                    torch.tensor(self.teacher_preds[index]),
                    torch.tensor(self.labels[index]))
        elif self.labeled:
            return (torch.tensor(self.examples[index]),
                    torch.tensor(self.labels[index]))
        elif self.predicted:
            return (torch.tensor(self.examples[index]),
                    torch.tensor(self.teacher_preds[index]))
        else:
            return torch.tensor(self.examples[index])


def single_dataloader(path, dataset_type, batch_size, sampler, dataset_dict):
    assert os.path.exists(path)
    with open(path) as f:
        rows = f.readlines()
    data = dataset_type(rows, **dataset_dict)
    dataloader = DataLoader(data, sampler=sampler(data), batch_size=batch_size)
    print('loaded {}, {}'.format(path, len(data)))
    return dataloader


def meta_dataloader(loader_labeled, loader_unlabeled):
    """
    Костыли, чтобы переключаться между размеченными и неразмеченными данными
    """
    loader_labeled = iter(loader_labeled)
    loader_unlabeled = iter(loader_unlabeled)
    # print(int(len(loader_labeled)+len(loader_unlabeled))) #number of steps
    for i in range(int(len(loader_labeled) + len(loader_unlabeled))):
        if random.random() < 0.5:
            try:
                yield next(loader_labeled)
            except StopIteration:
                dataloader_iterator = iter(loader_labeled)
                yield next(dataloader_iterator)
        else:
            try:
                yield next(loader_unlabeled)
            except StopIteration:
                dataloader_iterator = iter(loader_unlabeled)
                yield next(dataloader_iterator)


def eval(model, data_path, batch_size, tokenize_method, max_seq_length,
         units_to_ids, device='cuda'):
    model.eval()

    eval_loader = single_dataloader(data_path,
                                    batch_size=batch_size,
                                    sampler=SequentialSampler,
                                    dataset_type=BlendDataset,
                                    dataset_dict={
                                        'max_seq_length': max_seq_length,
                                        'map_to_ids': units_to_ids,
                                        'tokenize_metod': tokenize_method,
                                        'predicted': False,
                                        'labeled': True})

    criterion = nn.BCEWithLogitsLoss()
    all_logits = None
    all_labels = None
    eval_loss = 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in eval_loader:
        input_ids, labels = batch

        with torch.no_grad():
            logits = model(input_ids.to(device))
            tmp_eval_loss = criterion(logits, labels.to(device))

        if all_logits is None:
            all_logits = logits.detach().cpu().numpy()
        else:
            all_logits = np.concatenate(
                (all_logits, logits.detach().cpu().numpy()), axis=0)

        if all_labels is None:
            all_labels = labels.detach().cpu().numpy()
        else:
            all_labels = np.concatenate(
                (all_labels, labels.detach().cpu().numpy()), axis=0)

        eval_loss += tmp_eval_loss.mean().item()

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    roc_auc = np.sum([roc_auc_score(all_labels[:, i], all_logits[:, i])
                      for i in range(6)]) / 6
    result = {'eval_loss': eval_loss,
              'roc_auc': roc_auc}
    return result


def loss_kd(outputs, teacher_outputs, alpha, T, num_classes, labels=None):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    """
    if labels is not None:
        div_loss = torch.stack([F.kl_div(F.logsigmoid(outputs[:, i] / T),
                                         F.sigmoid(teacher_outputs[:, i] / T))
                                for i in range(num_classes)])
        student_loss = F.binary_cross_entropy_with_logits(outputs,
                                                          labels)  # self.lambda_factor * (self.temperature ** 2)
        KD_loss = alpha * div_loss.mean() + (1. - alpha) * student_loss
    else:
        div_loss = torch.stack([F.kl_div(F.logsigmoid(outputs[:, i] / T),
                                         F.sigmoid(teacher_outputs[:, i] / T))
                                for i in range(num_classes)]).mean()
        KD_loss = div_loss.mean()
    return KD_loss


def KD_train(data_labeled, data_unlabeled, test_path, save_dir,
             alpha, T, units2ids, batch_size, learning_rate,
             student_tokenizer=None, max_seq_length=512,
             num_epochs=5, report_every=1000, continue_from=None,
             device='cuda'):
    logger = logging.getLogger()
    logger.info('Alpha = {}, T = {}'.format(alpha, T))
    last_epoch = 0

    #     teacher_model = teacher_model.to(device)
    #     teacher_model.eval()

    student_model = BlendCNN(channels=[100] * 8,
                             num_labels=6, vocab_length=len(student_vocab),
                             use_dropout=True,
                             use_batchnorm=True,
                             emb_dim=512,
                             dropout_prob=0.6)
    student_model = student_model.to(device)
    student_model.train()

    student_criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(student_model.parameters(), lr=learning_rate)
    # scheduler = CyclicLR(optimizer, step_size_up=10000, base_lr=learning_rate, max_lr=learning_rate/4)
    scheduler = CosineAnnealingLr(optimizer, T_max=10000)
    if continue_from:
        assert os.path.exists(continue_from)
        checkpoint = torch.load(continue_from)
        student_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        last_epoch = checkpoint['epoch']

    for e in range(last_epoch, num_epochs):
        running_loss, step, step_counter = 0, 0, 0
        losses = []
        for batch in meta_dataloader(data_labeled, data_unlabeled):
            if len(batch) == 3:  # labeled
                examples, predictions, labels = batch
                labels = labels.float().to(device)
            else:
                examples, predictions = batch
                labels = None
            predictions = predictions.to(device)

            # actual train
            optimizer.zero_grad()
            #             with torch.no_grad():
            #                 teacher_output = teacher_model(teacher_x.to(device),
            #                                                mask.to(device),
            #                                                segments.to(device))
            student_output = student_model(examples.to(device))
            loss = loss_kd(student_output, predictions, alpha=alpha, T=T,
                           num_classes=student_output.shape[1], labels=labels)
            loss.backward()
            running_loss += loss.item()
            losses.append(loss.item())
            #             for param in student_model.parameters():
            #                 clip_grad_norm_(param, 1.)
            optimizer.step()
            step += 1
            if step % report_every == 0:
                step_counter += 1
                logger.info('Loss: {}'.format(running_loss / step))
                running_loss = 0
                step = 0
            # scheduler.step(running_loss)

        plt.plot(list(range(len(losses))), losses)
        plt.show()

    #         output_model_file = os.path.join(save_dir,
    #                                          "blendcnn_{}x{}_epoch{}.bin".format(
    #                                              alpha, T, e))
    #         torch.save({'epoch': e + 1,
    #                     'model_state_dict': student_model.state_dict(),
    #                     'optimizer_state_dict': optimizer.state_dict(),
    #                     'scheduler_state_dict': scheduler.state_dict(),
    #                     }, output_model_file)
    #         for name, param in student_model.named_parameters():
    #             print(name)
    #             if param.requires_grad:
    #                 print(param.data)
    student_model.eval()
    eval_results = eval(student_model,
                        test_path,
                        units_to_ids=units2ids,
                        tokenize_method=student_tokenizer.tokenize,
                        batch_size=batch_size,
                        max_seq_length=max_seq_length)
    logger.info('eval_loss: {}, roc_auc: {}'.format(
        eval_results['eval_loss'], eval_results['roc_auc']))


# class batch_generator():
#     def __init__(self, data_path, batch_size, mapping2idx, max_seq_length,
#                  workers = 8, labeled: bool = True): #device: str = 'cpu'):
#         '''
#         Simple epoch iterator. dataset_fh is a file stream
#         Assumes that input and target features are separated by '|||'
#         '''
#         from multiprocessing import Pool
#         with open(data_path) as f:
#             res = [x for x in f if x.strip()]
#         tokenizer = BPERegexTokenizer(mapping2idx, max_seq_length)

#         examples, labels = [], []
#         for r in res:
#             if labeled:
#                 x, y = r.strip().rsplit('|||', 1)
#                 labels.append(list(map(int, y.split())))
#             else:
#                 x = r.strip()
#             examples.append(x)

#         p = Pool(workers)
#         jobs = p.map(tokenizer.tokenize, examples)
#         for e, j in enumerate(jobs):
#             if j:
#                 examples[e] = j
#         p.close()
#         p.join()
#         self.examples = examples
#         self.labels = labels
#         self.batch_size = batch_size
#         #self.device = device

#     def __len__(self):
#         return len(self.examples)

#     def __next__(self):
#         batch = np.random.choice(len(self.examples[0]), self.batch_size,
#                                  replace=False)
#         x, y = [self.examples[e] for e in batch], \
#                [self.labels[e] for e in batch]
#         return (torch.tensor(x), torch.tensor(y))

#     def __iter__(self):
#         np.random.shuffle(self.examples)
#         return self
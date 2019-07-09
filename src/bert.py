import os, random, time
from pathlib import Path
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, \
    SequentialSampler
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel
from pytorch_pretrained_bert.optimization import BertAdam
import numpy as np
from sklearn.metrics import roc_auc_score


def word_dropout(tokens, unk_token='[UNK]', ratio=0.25):
    """
    Replaces random tokens with <unk>. BERT uses [UNK]
    Turns out it does not have any good influence on finetuning results tho
    :param tokens: list of tokens
    :param ratio: approx. ratio of tokens to be replaced
    :return: string with unks
    """
    to_replace = set(random.choices(range(len(tokens)),
                                    k=np.ceil(len(tokens) * ratio)))
    return [unk_token if e in to_replace else tokens[e]
            for e, x in enumerate(tokens)]


class BertForMultiLabelClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    :param num_labels: number of output classes
    """

    def __init__(self, config, num_labels=2):
        super(BertForMultiLabelClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                     output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True


class CustomDataset(Dataset):
    def __init__(self, texts, tokenizer, max_seq_length=512,
                 labeled=True, pretokenized=False): #use_word_dropout=False):
        """
        Stores tokens from original text mapped to vocab IDs
        as well as corresponding labels and unked versions.
        :param texts: just lines read from file
        :param tokenizer: to tokenize text
        :param max_seq_length: 512 by default
        :param labeled: True for train/test (eval), not for prediction
        :param use_word_dropout: replace random tokens with [UNK] (no need)
        """
        self.examples = []
        self.labels = []
        self.max_seq_length = max_seq_length
        self.labeled = labeled
        for t in texts:
            try:
                if self.labeled:
                    seq, labels = t.strip().split(' |||')
                    self.labels.append([float(x) for x in labels.split()])
                else:
                    seq = t.strip()
                if pretokenized:
                    tokens = seq.split()
                else:
                    tokens = tokenizer.tokenize(seq)

                if len(tokens) > max_seq_length - 2:
                    tokens = tokens[:(max_seq_length - 2)]

                input_ids = tokenizer.convert_tokens_to_ids(
                    ["[CLS]"] + tokens + ["[SEP]"])
                self.examples.append(input_ids)

                # if use_word_dropout:
                #     tokens = word_dropout(tokens)
                #     input_ids = tokenizer.convert_tokens_to_ids(
                #         ["[CLS]"] + tokens + ["[SEP]"])
                #     self.examples.append(input_ids)
                #     if self.labeled:
                #         self.labels.append([float(x) for x in labels.split()])
            except ValueError:
                # print(i)
                pass

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        ids = self.examples[index]
        input_mask = [1] * len(ids)
        # zero-pad
        padding = [0] * (self.max_seq_length - len(ids))
        ids += padding
        input_mask += padding
        segment_ids = [0] * len(ids)  # I never use the second segment anyway
        if self.labeled:
            return (ids, input_mask, segment_ids, self.labels[index])
        else:
            return (ids, input_mask, segment_ids)


def create_dataloader(path, batch_size, tokenizer, max_seq_length, sampler,
                      labeled=True, pretokenized=False):
                      #use_word_dropout=False):
    """
    Basically wraps creation of a Dataset and an associated DataLoader
    :param path: Path() to data
    :param batch_size: batch size
    :param tokenizer: tokenizer (must be BertTokenizer)
    :param max_seq_length: maximum allowed length of a sequence
    :param sampler: Random for train, or Sequential for eval/predict
    :param labeled: whether the data has correct labels
    :param pretokenized: whether the dataset on the path has been tokenized
    :return: DataLoader along with its length
    """
    assert Path.exists(path)
    with path.open() as f:
        rows = f.readlines()
    num_examples = len(rows)
    data = CustomDataset(rows,
                         tokenizer=tokenizer,
                         pretokenized=pretokenized,
                         max_seq_length=max_seq_length,
                         # use_word_dropout=use_word_dropout,
                         labeled=labeled)
    dataloader = DataLoader(data, sampler=sampler(data),
                            batch_size=batch_size)
                            #batch_size=batch_size, collate_fn=custom_collate)
    return dataloader, num_examples


# Prepare model
def get_model(model_path, num_labels, weights=None, device='cuda'):
    if weights:
        assert Path.exists(weights)
        state_dict = torch.load(weights)
        model = BertForMultiLabelClassification.from_pretrained(
            model_path, num_labels=num_labels, state_dict=state_dict)
    else:
        model = BertForMultiLabelClassification.from_pretrained(
            model_path, num_labels=num_labels)
    model.to(device)
    return model


def eval(model, data_path, batch_size, max_seq_length, tokenizer,
         device='cuda'):
    model.to(device)
    model.eval()

    eval_loader, _ = create_dataloader(data_path,
                                       batch_size=batch_size,
                                       tokenizer=tokenizer,
                                       pretokenized=False,
                                       max_seq_length=max_seq_length,
                                       sampler=SequentialSampler,
                                       labeled=True)
    criterion = BCEWithLogitsLoss()
    all_logits = None
    all_labels = None
    eval_loss = 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in eval_loader:
        input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
            logits = model(input_ids.to(device), segment_ids.to(device),
                           input_mask.to(device))
            # loss_fct = BCEWithLogitsLoss(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            tmp_eval_loss = criterion(logits, label_ids.to(device))

        if all_logits is None:
            all_logits = logits.detach().cpu().numpy()
        else:
            all_logits = np.concatenate(
                (all_logits, logits.detach().cpu().numpy()), axis=0)

        if all_labels is None:
            all_labels = label_ids.detach().cpu().numpy()
        else:
            all_labels = np.concatenate(
                (all_labels, label_ids.detach().cpu().numpy()), axis=0)

        eval_loss += tmp_eval_loss.mean().item()

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    roc_auc = [roc_auc_score(all_labels[:, i], all_logits[:, i])
               for i in range(6)]
    roc_auc = np.mean(roc_auc)

    result = {'eval_loss': eval_loss,
              'roc_auc': roc_auc}
    return result


def predict(model, data_path, batch_size, tokenizer, max_seq_length,
            device='cuda'):
    model.to(device)
    model.eval()

    inference_loader, _ = create_dataloader(data_path,
                                            batch_size=batch_size,
                                            tokenizer=tokenizer,
                                            pretokenized=False,
                                            max_seq_length=max_seq_length,
                                            sampler=SequentialSampler,
                                            labeled=False)
    all_logits = None
    for batch in inference_loader:
        with torch.no_grad():
            logits = model([x.to(device) for x in batch])
            logits = logits.sigmoid()

        if all_logits is None:
            all_logits = logits.detach().cpu().numpy()
        else:
            all_logits = np.concatenate(
                (all_logits, logits.detach().cpu().numpy()), axis=0)

    return all_logits


def fit(model, num_epochs, train_path, test_path, batch_size,
        tokenizer, save_to,
        learning_rate=3e-5,
        warmup_proportion=0.1,
        max_seq_length=512,
        gradient_accumulation_steps=1,
        #use_word_dropout=False,
        report_every=5000,
        device='cuda'):
    global_step = 0

    train_loader, num_examples = create_dataloader(train_path,
                                                   batch_size=batch_size,
                                                   tokenizer=tokenizer,
                                                   pretokenized=True,
                                                   # train dataset has been tokenized beforehand
                                                   max_seq_length=max_seq_length,
                                                   sampler=RandomSampler,
                                                   #use_word_dropout=use_word_dropout,
                                                   labeled=True)
    print('loaded train')

    num_train_steps = int(num_examples / batch_size /
                          gradient_accumulation_steps * num_epochs)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if
                    not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if
                    any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=learning_rate,
                         warmup=warmup_proportion,
                         t_total=num_train_steps)
    criterion = BCEWithLogitsLoss()
    model.freeze_bert_encoder()
    model.to(device)
    # scheduler
    for i in range(num_epochs):
        model.train()
        tr_loss = 0
        num_tr_examples, num_tr_steps = 0, 0

        for step, batch in enumerate(train_loader):
            input_ids, input_mask, segment_ids, label_ids = batch

            logits = model(input_ids.to(device), input_mask.to(device),
                           segment_ids.to(device))
            loss = criterion(logits, label_ids.to(device))
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()

            tr_loss += loss.item()
            num_tr_examples += batch_size
            num_tr_steps += 1
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            if (step + 1) % report_every == 0:
                logger.info('Loss: {}'.format(tr_loss / num_tr_steps))

            output_model_file = save_to/"finetuned_epoch{}.bin".format(i)
            torch.save(model.state_dict(), output_model_file)

        model.eval()
        eval_results = eval(model, tokenizer=tokenizer, data_path=test_path,
                            batch_size=batch_size,
                            max_seq_length=max_seq_length)
        logger.info(
            'eval_loss: {}, roc_auc: {}'.format(eval_results['eval_loss'],
                                                eval_results['roc_auc']))


if __name__ == '__main__':
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    import logging

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)
    logger = logging.getLogger(__name__)

    # BERT_PRETRAINED = 'bert-base-uncased'
    BERT_PRETRAINED = 'bert-base-multilingual-cased'
    train_path = Path(
        'drive/My Drive/toxic/sets/bert_add5_syn/plain_add_syn.tok.txt')
    test_path = Path('drive/My Drive/toxic/sets/bert_add5_syn/test.txt')
    BERT_FINETUNED = Path('drive/My Drive/toxic/models/bert-funetuned')
    BERT_FINETUNED.mkdir(exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(BERT_PRETRAINED,
                                              do_lower_case=False)

    model = get_model(BERT_PRETRAINED, num_labels=6)

    model.module.freeze_bert_encoder()
    fit(model=model,
        save_to=BERT_FINETUNED,
        tokenizer=tokenizer,
        train_path=train_path,
        test_path=test_path,
        num_epochs=5,
        batch_size=8)
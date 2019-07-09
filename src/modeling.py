import os
import torch
from torch import nn
from torch.nn import functional as F
from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel
from pytorch_pretrained_bert.tokenization import BertTokenizer


class BertForMultiLabelClassification(BertPreTrainedModel):
    """
    BERT model for classification.
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


def get_bert_model(model_path, num_labels, weights=None, device='cuda'):
    if weights:
        assert os.path.exists(weights)
        state_dict = torch.load(weights)
        model = BertForMultiLabelClassification.from_pretrained(
            model_path, num_labels=num_labels, state_dict=state_dict)
    else:
        model = BertForMultiLabelClassification.from_pretrained(
            model_path, num_labels=num_labels)
    model.to(device)
    return model


class BlendCNN(nn.Module):
    def __init__(self,
                 vocab_length,
                 num_labels=6,
                 channel_size=100,
                 num_channels=10,
                 emb_dim=300,  #
                 kernel_size=5,
                 n_hidden_dense=1024,
                 dropout_prob=0.6):
        super(BlendCNN, self).__init__()

        self.embeddings = nn.Embedding(vocab_length, emb_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.conv1 = nn.Conv1d(in_channels=emb_dim,
                               out_channels=channel_size,
                               kernel_size=kernel_size,
                               padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(in_channels=channel_size,
                               out_channels=channel_size,
                               kernel_size=kernel_size,
                               padding=kernel_size // 2)
        self.batchnorm = nn.BatchNorm1d(channel_size)
        self.activation_fn = nn.ReLU()
        self.num_convs = num_channels - 1
        self.dense = nn.Linear(in_features=channel_size * (num_channels),
                               out_features=n_hidden_dense)
        self.dense_activation = nn.ReLU()
        self.classifier = nn.Linear(n_hidden_dense, num_labels)

    def forward(self, input_ids, padding_idx=1):
        # mask = (input_ids != padding_idx).unsqueeze(1)
        h = self.embeddings(input_ids).transpose(1, 2)
        h = self.dropout(h)
        output = []
        h = self.conv1(h)
        h = self.batchnorm(h)
        h = self.activation_fn(h)
        h = self.dropout(h)
        output.append(F.adaptive_max_pool1d(h, output_size=1).squeeze())

        if self.num_convs > 0:
            for c in range(self.num_convs):
                h = self.conv2(h)
                h = self.batchnorm(h)
                h = self.activation_fn(h)
                h = self.dropout(h)
                h = F.max_pool1d(h, kernel_size=1)
                output.append(
                    F.adaptive_max_pool1d(h, output_size=1).squeeze())

        output = torch.cat(output, -1)
        dense = self.dense(output)
        dense_act = self.activation_fn(dense)
        logits = self.classifier(dense_act)
        return logits


class RCNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_classes=6,
                 dropout_prob=0.6, weights=None):
        super(RCNN, self).__init__()
        if weights is not None:
            self.word_embeddings = nn.Embedding.from_pretrained(weights)
        else:
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2,
                            dropout=dropout_prob, bidirectional=True)
        self.W2 = nn.Linear(2 * hidden_dim + embedding_dim, hidden_dim)
        self.label = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_batch):
        embed = self.word_embeddings(input_batch)
        embed = embed.permute(1, 0, 2)
        lstm_out, (h_n, c_n) = self.lstm(embed)
        final_encoding = torch.cat((lstm_out, embed), 2).permute(1, 0, 2)
        y = self.W2(final_encoding).permute(0, 2, 1)
        y = F.max_pool1d(y, y.size()[2]).squeeze(2)
        logits = self.label(y)
        return logits
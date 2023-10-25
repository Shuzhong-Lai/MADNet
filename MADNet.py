# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.MAI import MAI


class Config(object):
    """config parameters"""

    def __init__(self, dataset):
        self.model_name = 'MADNet'
        self.data_path = dataset + '/data'
        self.train_path = self.data_path + '/multi_train_data.xlsx'
        self.dev_path = self.data_path + '/multi_dev_data.xlsx'
        self.test_path = self.data_path + '/multi_test_data.xlsx'
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]  # classList
        self.f1_save_path = dataset + '/saved_dict/f1_' + self.model_name + '.ckpt'
        self.auc_save_path = dataset + '/saved_dict/auc_' + self.model_name + '.ckpt'
        self.log_path = dataset + '/log/' + self.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = len(self.class_list)

        # for model
        self.learning_rate = 1e-3
        self.num_epoch = 80
        self.require_improvement = 1000
        self.batch_size = 32
        self.dropout = 0.5

        # for nlp
        self.embed = 300  # embedding size
        self.vocab_path = dataset + '/data/vocab.pkl'
        self.min_freq = 1
        self.pad_size = 32
        self.hidden_size = 128
        self.num_layers = 2
        self.hidden_size2 = 64
        self.n_vocab = 0
        self.hidden_size3 = 256
        self.filter_sizes = (2, 3, 4)
        self.num_filters = 256

        # for cv
        self.image_path = self.data_path + '/img'
        self.input_size = 224
        self.conv_arch = ((1, 3, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
        self.fc_features = 512 * 7 * 7
        self.fc_hidden_units = 4096

        # for MAI
        self.beta_shift = 1
        self.dropout_prob = 0.5


class ImageFeatureExtractor(nn.Module):
    def __init__(self, config, init_weights=False):
        super(ImageFeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 96, (11, 11), 4, 1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, (5, 5), 1, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, (3, 3), 1, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 384, (3, 3), 1, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 256, (3, 3), 1, 1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, 2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(config.dropout),
            nn.Linear(6400, 2048),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(2048, 2048),
            nn.ReLU()
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, img):
        feature = self.conv(img)
        out = self.fc(feature)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


class TextFeatureExtractor(nn.Module):
    def __init__(self, config):
        super(TextFeatureExtractor, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=config.num_filters, kernel_size=(k, config.embed)) for k in
             config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        embed = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        embed = embed.unsqueeze(1)  # (batch_size, 1, seq_len, embed_dim)
        conved = [F.relu(conv(embed)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        x_cat = torch.cat(pooled, dim=1)
        out = self.dropout(x_cat)
        return out


# MADNet
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.cv = ImageFeatureExtractor(config)
        self.config = config
        self.nlp = TextFeatureExtractor(config)
        self.first_fc = 768

        self.MAI = MAI(self.first_fc, self.config.dropout_prob)

        # classification layer
        self.fc = nn.Sequential(
            nn.Linear(self.first_fc, 2048),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(256, len(self.config.class_list)),
        )

    def forward(self, x):
        # divide input
        if x.shape[0] != 1:
            x = x.unsqueeze(0)
        nlp_input = x.narrow(1, 0, self.config.pad_size)
        nlp_input = nlp_input.type(torch.IntTensor).to(self.config.device)
        cv_input = x.narrow(1, self.config.pad_size, (3 * self.config.input_size * self.config.input_size)).to(
            self.config.device)
        behavior_input = x.narrow(1, (3 * self.config.input_size * self.config.input_size), 4).to(self.config.device)

        # transform the shape of cv input
        cv_input = cv_input.reshape(self.config.batch_size, 3, self.config.input_size, -1)

        # extract features
        nlp_features = self.nlp(nlp_input)
        cv_features = self.cv(cv_input)
        # due to the high density of information, behavior feature remain to fusion layer
        behavior_features = behavior_input

        # fusion layer using MAI
        nlp_features = nlp_features.reshape(self.config.batch_size, -1)
        cv_features = cv_features.reshape(self.config.batch_size, -1)
        behavior_features = behavior_features.reshape(self.config.batch_size, -1)

        features = self.MAI(nlp_features, cv_features, behavior_features, self.config)

        # predict the anxiety state
        out = self.fc(features)
        return out

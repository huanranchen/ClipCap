import torch
import torch.nn as nn
import torchvision
from torchvision import models
from transformers import BertModel, BertConfig


class ImgEncoder(nn.Module):
    def __init__(self, output_dim=2048):
        super(ImgEncoder, self).__init__()
        self.cnn = models.resnet101(pretrained=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        x = self.cnn.conv1(x)
        x = self.cnn.bn1(x)
        x = self.cnn.relu(x)
        x = self.cnn.maxpool(x)
        x = self.cnn.layer1(x)
        x = self.cnn.layer2(x)
        x = self.cnn.layer3(x)
        x = self.cnn.layer4(x)
        x = self.cnn.avgpool(x)
        x = x.squeeze(2).squeeze(2)
        return x

    def load_model(self, path='./image_encoder.pth'):
        self.load_state_dict(torch.load(path, map_location=self.device))


class TextEncoder(nn.Module):
    def __init__(self,
                 tokenizer,
                 device=torch.device('cpu'),
                 num_hidden_layers=6,
                 output_dim=512):
        super(TextEncoder, self).__init__()
        self.tokenizer = tokenizer
        config = BertConfig(vocab_size=self.tokenizer.vocab_size, num_hidden_layers=num_hidden_layers)
        self.model = BertModel(config).to(device)

    def forward(self, text):
        '''
        :param text: a tuple of test
        :return:
        '''
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input = self.tokenizer.get_bert_tokens(text)
        output = self.model(**input)

        return output[1]


if __name__ == '__main__':
    a = TextEncoder()

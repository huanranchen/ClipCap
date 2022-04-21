import torch
import torch.nn as nn
from transformers import GPT2Model


class Decoder(nn.Module):
    def __init__(self, tokenizer, input_dim=2048, k=10, gpt_dim=768, gpt_out_dim=768, device=torch.device('cpu')):
        super(Decoder, self).__init__()
        self.device = device
        self.k = k  # this part will not be trained
        self.gpt_dim = gpt_dim
        self.tokenizer = tokenizer

        self.gpt = GPT2Model.from_pretrained('distilgpt2').to(device)
        self.mapper = nn.Linear(input_dim, k * gpt_dim)

        self.classifier = nn.Sequential(
            # nn.GELU(),
            nn.Linear(gpt_out_dim, tokenizer.vocab_size),
        )

        self.embedding = nn.Embedding(tokenizer.vocab_size, gpt_dim)

    def _gpt_forward(self, img_feature, text):
        '''
        :param img_feature:
        :param text:  tuple of str
        :return:
        '''
        N = len(text)  # batch_size
        x = self.mapper(img_feature).reshape(-1, self.k, self.gpt_dim)
        dic = self.tokenizer.get_bert_tokens(text)

        # embedding [N, T, D]
        embedded = torch.cat([x, self.embedding(dic['input_ids'])], dim=1)

        # mask,  [N, T]
        mask = torch.cat([torch.ones_like(x[:, :, 0]), dic['attention_mask']], dim=1)  # N, T
        out = self.gpt(inputs_embeds=embedded, attention_mask=mask)
        return out.last_hidden_state, dic

    def forward(self, img_feature, text):
        '''
        :param img_feature:
        :param text:  tuple of str
        :return:
        '''
        out, dic = self._gpt_forward(img_feature, text)

        # print(out.logits)

        pre = self.classifier(out)
        # print(pre.shape)
        # pre = out.logits

        pre = pre[:, self.k:-1, :]  # without last one, because that place is the output of <EOF> or <PAD>

        # generate ground truth
        mask = dic['attention_mask'][:, 1:]
        ground_truth = dic['input_ids'][:, 1:]  # without <START> token

        # reshape
        pre = pre.reshape(-1, pre.shape[-1])  # N*T, D
        ground_truth = ground_truth.reshape(-1)  # N*T
        mask = mask.reshape(-1)  # N*T
        mask = mask.to(torch.bool)
        return pre, mask, ground_truth

    def predict_one(self, img_feature, max_length=20):
        '''
        :param img_feature: (1,D)
        :return: a list contain idx
        '''
        x = self.mapper(img_feature).reshape(-1, self.k, self.gpt_dim)  # (1, K, D)

        now_word_idx = self.tokenizer.word2idx['<START>']
        result = [now_word_idx]
        for now_length in range(max_length):
            now_word_embedding = self.embedding(torch.tensor([now_word_idx])).unsqueeze(0)  # 1, 1, D
            x = torch.cat([x, now_word_embedding], dim=1)
            pre = self.gpt(inputs_embeds=x)
            pre = pre.last_hidden_state[:, -1, :]  #
            pre = self.classifier(pre)  # 1, D
            _, now_word_idx = torch.max(pre, dim=1)
            now_word_idx = now_word_idx.item()
            result.append(now_word_idx)

        return result

    def predict_one_with_ground_truth(self, image_feature, text):
        # pre (1, T, D)
        pre, _, _ = self.forward(image_feature, text)
        pre = pre.squeeze(1)
        result = []
        for i in range(pre.shape[0]):
            _, now_word_idx = torch.max(pre[i,:], dim = 0)
            now_word_idx = now_word_idx.item()
            result.append(now_word_idx)

        return result

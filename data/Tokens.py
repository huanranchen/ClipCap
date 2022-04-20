import torch
from .dataUtils import split_sentence_into_words, read_into_dic
import numpy as np
import os

def invert_dict(d):
    return dict([(v, k) for (k, v) in d.items()])

class MyToken():
    def __init__(self, ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.word2idx = {}
        self.idx2word = {}

    def process_all_words(self, ground_truth):
        '''
        :param ground_truth: a dic from image_name to ground truth
        :return:
        '''
        self.word2idx['<START>'] = 101  # AKA 'CLS'
        self.word2idx['<END>'] = 102
        self.word2idx['<UNK>'] = 99  # unknown
        self.word2idx['token_type_ids'] = 0
        self.word2idx['attention_mask'] = 1
        self.word2idx['<PAD>'] = 2

        all_sentences = list(ground_truth.values())

        # synthesize all_words
        all_words = set()
        for image_ground_truth in all_sentences:
            for sentence in image_ground_truth:
                now_words = set(split_sentence_into_words(sentence))
                all_words = all_words.union(now_words)

        for word in all_words:
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx) + 100

    def get_one_sentence_bert_tokens(self, x):
        '''
        example:
        {'input_ids': [101, 6616, 2017, 102], 'token_type_ids': [0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1]}
        :param x: a sentence, str.
        :return: dic.
        '''
        words = split_sentence_into_words(x)
        input_ids = [self.word2idx[word] if word in self.word2idx else self.word2idx['<UNK>'] for word in words]
        # token_type_ids = [self.word2idx['token_type_ids']] * len(input_ids)
        # attention_mask = [self.word2idx['attention_mask']] * len(input_ids)
        # input_ids = torch.tensor(input_ids)
        # token_type_ids = torch.tensor(token_type_ids)
        # attention_mask = torch.tensor(attention_mask)
        #
        # result = {}
        # result['input_ids'] = input_ids
        # result['token_type_ids'] = token_type_ids
        # result['attention_mask'] = attention_mask
        return input_ids

    def get_bert_tokens(self, sentences):
        '''
        :param sentences: a tuple of str
        :return: a tensor (N, H)
        '''
        attention_mask = []
        N = len(sentences)
        input_ids = [self.get_one_sentence_bert_tokens(sentence) for sentence in sentences]
        lengths = [len(sentence) for sentence in input_ids]
        max_length = max(lengths)
        need_pad_length = [max_length - len for len in lengths]

        # pad
        input_ids = [sentence + [self.word2idx['<PAD>']] * need_pad_length[i] for i, sentence in enumerate(input_ids)]
        attention_mask = [[1] * lengths[i] + [0] * need_pad_length[i] for i, _ in enumerate(lengths)]

        result = {}
        result['input_ids'] = torch.tensor(input_ids, device=self.device)
        result['attention_mask'] = torch.tensor(attention_mask, device=self.device)
        result['token_type_ids'] = torch.zeros((N, max_length), device=self.device, dtype=torch.int32)

        return result

    @property
    def vocab_size(self):
        return len(self.word2idx) + 100

    def save_dic(self, out_path='./'):
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        np.save(out_path + 'word2idx.npy', self.word2idx)

    def load_dic(self, path='./'):
        self.word2idx = np.load(path + 'word2idx.npy', allow_pickle=True).item()

    def sythesize_idx2word(self):
        self.idx2word = invert_dict(self.word2idx)




if __name__ == '__main__':
    ground_truth = read_into_dic()
    token = MyToken()
    token.process_all_words(ground_truth)
    test = ('hello liuhaotian', 'do not worry')
    print(token.get_bert_tokens(test))

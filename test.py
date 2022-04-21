import torch
from models.encoders import ImgEncoder
from models.decoders import Decoder
from data.data import get_test_loader
from data.Tokens import MyToken
from criterions import evaluate


def incorperate_str_list_into_sentence(list):
    '''
    :param list: maybe a lot of end in the end. We will delete start and end.
    for example:
    ['<START>', 'a', 'man', 'in', 'a', 'blue', 'shirt', 'be', 'stand', 'on', 'a', 'sidewalk',
    <END>', '<END>', '<END>', '<END>', '<END>']
    :return: list
    '''
    str = ''
    for word in list:
        if word == '<START>':
            continue
        if word == '<END>':
            break
        str += word
        str += ' '

    return [str]


def test(data_source='test_data', total_num=1, is_forcing=False):
    '''
    :param data_source:
    :param total_num:
    :return: a dic
    '''
    if data_source == 'test_data':
        loader = get_test_loader()

    candidate = {}
    ground_truth = {}
    count = 0

    tokenizer = MyToken()
    tokenizer.load_dic()
    tokenizer.sythesize_idx2word()
    encoder = ImgEncoder()
    decoder = Decoder(tokenizer=tokenizer)
    encoder.load_state_dict(torch.load('encoder.ckpt', map_location=torch.device('cpu')))
    decoder.load_state_dict(torch.load('decoder.ckpt', map_location=torch.device('cpu')))

    for img, text in loader:
        x = encoder(img)
        if is_forcing:
            result = decoder.predict_one_with_ground_truth(x, text)
        else:
            result = decoder.predict_one(x)
        pre = [tokenizer.idx2word[idx] for idx in result]

        # print(incorperate_str_list_into_sentence(pre))
        # print(list(text))
        # assert False

        candidate[count] = incorperate_str_list_into_sentence(pre)
        ground_truth[count] = list(text)
        count += 1
        if count >= total_num:
            return candidate, ground_truth


def get_scores(mode='bleu', is_forcing=False, total_samples=100):
    '''
    :param mode: bleu? cider?
    :param is_forcing: whether or not using teacher forcing
    :param total_samples:use how many data to evaluate
    :return:
    '''
    candidate, ground_truth = test(total_num=total_samples, is_forcing=is_forcing)

    if mode == 'bleu':
        evaluate.estimate_bleu(candidate, ground_truth)

    if mode == 'cider':
        evaluate.estimate_cider(candidate, ground_truth)


if __name__ == '__main__':
    candidate, ground_truth = test()
    print(candidate, ground_truth)
    #get_scores(is_forcing=True,mode = 'cider')

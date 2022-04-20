import torch
from models.encoders import ImgEncoder
from models.decoders import Decoder
from data.data import get_test_loader
from data.Tokens import MyToken

def test(data_source = 'test_data'):
    if data_source == 'test_data':
        loader = get_test_loader()

    tokenizer = MyToken()
    tokenizer.load_dic()
    tokenizer.sythesize_idx2word()
    encoder = ImgEncoder()
    decoder = Decoder(tokenizer=tokenizer)
    # encoder.load_state_dict(torch.load('encoder.ckpt', map_location=torch.device('cpu')))
    # decoder.load_state_dict(torch.load('decoder.ckpt', map_location=torch.device('cpu')))

    for img, text in loader:
        x = encoder(img)
        result = decoder.predict_one(x)
        pre = [tokenizer.idx2word[idx] for idx in result]
        print(pre)

        assert False

if __name__ == '__main__':
    test()
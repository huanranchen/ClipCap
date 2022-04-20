import os.path

import torch
import torch.nn as nn
from models.encoders import ImgEncoder
from models.decoders import Decoder
from data.Tokens import MyToken
from data.data import get_train_loader
from tqdm import tqdm


def train(
        batch_size=32,
        lr=1e-3,
        weight_decay=0,
        label_smoothing=0,
        k=10,
        total_epoch=100,
        image_path='./data/images/',
        lemma_path='./data/Flickr8k.lemma.token.txt',
        if_pretrain=False,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = MyToken()
    tokenizer.load_dic()
    tokenizer.sythesize_idx2word()
    encoder = ImgEncoder().to(device)
    encoder.eval()
    if if_pretrain:
        encoder.load_model()
    decoder = Decoder(tokenizer, k=k).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)
    train_loader, valid_loader = get_train_loader(image_path=image_path, lemma_path=lemma_path, batch_size=batch_size)
    optimizer = torch.optim.AdamW(
        [{'params': decoder.parameters()}], lr=lr, weight_decay=weight_decay
    )

    if os.path.exists('encoder.ckpt'):
        encoder.load_state_dict(torch.load('encoder.ckpt', map_location=torch.device('cpu')))
    if os.path.exists('decoder.ckpt'):
        decoder.load_state_dict(torch.load('decoder.ckpt', map_location=torch.device('cpu')))
    for epoch in range(total_epoch):
        # train
        decoder.train()
        train_loss = 0
        for img, text in tqdm(train_loader):
            img = img.to(device)
            x = encoder(img)
            pre, mask, ground_truth = decoder(x, text)


            # print(pre[mask].shape, ground_truth[mask])
            # for i in range(ground_truth[mask].shape[0]):
            #     print(tokenizer.idx2word[ground_truth[mask][i].item()])
            #
            # pre = pre[mask] # N*T, D
            # _, pre = torch.max(pre, dim = 1)    # N*T
            # for i in range(pre.shape[0]):
            #     print(tokenizer.idx2word[pre[i].item()])
            # assert False

            loss = criterion(pre[mask], ground_truth[mask])

            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)

        # valid
        decoder.eval()
        valid_loss = 0
        for img, text in tqdm(valid_loader):
            with torch.no_grad():
                img = img.to(device)
                x = encoder(img)
                pre, mask, ground_truth = decoder(x, text)
                loss = criterion(pre[mask], ground_truth[mask])
                valid_loss += loss.item()

        valid_loss /= len(valid_loader)

        print(f'epoch {epoch}, train loss = {train_loss}, valid loss = {valid_loss}')
        torch.save(encoder.state_dict(), 'encoder.ckpt')
        torch.save(decoder.state_dict(), 'decoder.ckpt')


def overfit_small_dataset(
        batch_size=32,
        lr=1e-3,
        weight_decay=0,
        label_smoothing=0,
        k=10,
        total_epoch=100,
        image_path='./data/images/',
        lemma_path='./data/Flickr8k.lemma.token.txt',
        if_pretrain=False,
):
    '''
    !!!!!!!!!!!!still not be implemented!!!!!!!!!!!!!!!!!!!!!!!
    :param batch_size:
    :param lr:
    :param weight_decay:
    :param label_smoothing:
    :param k:
    :param total_epoch:
    :param image_path:
    :param lemma_path:
    :param if_pretrain:
    :return:
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = MyToken()
    tokenizer.load_dic()
    tokenizer.sythesize_idx2word()
    encoder = ImgEncoder().to(device)

    if if_pretrain:
        encoder.load_model()
    decoder = Decoder(tokenizer, k=k).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)
    train_loader, valid_loader = get_train_loader(image_path=image_path, lemma_path=lemma_path, batch_size=batch_size)
    optimizer = torch.optim.AdamW(
        [{'params': decoder.parameters()}], lr=lr, weight_decay=weight_decay
    )

    encoder.load_state_dict(torch.load('encoder.ckpt', map_location=torch.device('cpu')))
    decoder.load_state_dict(torch.load('decoder.ckpt', map_location=torch.device('cpu')))
    for epoch in range(total_epoch):
        # train
        train_loss = 0
        for img, text in tqdm(train_loader):
            img = img.to(device)
            x = encoder(img)
            pre, mask, ground_truth = decoder(x, text)

            # print(pre[mask].shape, ground_truth[mask])
            # for i in range(ground_truth[mask].shape[0]):
            #     print(tokenizer.idx2word[ground_truth[mask][i].item()])
            #
            # pre = pre[mask] # N*T, D
            # _, pre = torch.max(pre, dim = 1)    # N*T
            # for i in range(pre.shape[0]):
            #     print(tokenizer.idx2word[pre[i].item()])
            # assert False

            loss = criterion(pre[mask], ground_truth[mask])
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)

        print(f'epoch {epoch}, train loss = {train_loss}')
        torch.save(encoder.state_dict(), 'encoder.ckpt')
        torch.save(decoder.state_dict(), 'decoder.ckpt')


if __name__ == '__main__':
    train(batch_size=1)

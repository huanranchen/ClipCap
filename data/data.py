import torch
from torch.utils.data import Dataset, DataLoader
from .dataUtils import read_into_dic
import os
from PIL import Image
import random
from torchvision import transforms


class MyDataSet(Dataset):
    def __init__(self, image_path='./data/images/', lemma_path='./data/Flickr8k.lemma.token.txt', mode='train'):
        self.image_path = image_path

        self.ground_truth = read_into_dic(lemma_path)
        imgs = os.listdir(image_path)
        imgs.sort()
        self.imgs = imgs

        self.augmentation = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
        ])

        if mode == 'train':
            self.imgs = self.imgs[0:7000]
        if mode == 'valid':
            self.imgs = self.imgs[7000:7500]
        if mode == 'test':
            self.imgs = self.imgs[7500:]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        img_name = self.imgs[item]
        # print(img_name)
        img = Image.open(os.path.join(self.image_path, img_name))
        img = self.augmentation(img)

        text = self.ground_truth[img_name]
        text = text[random.randint(0, len(text) - 1)]

        return img, text


def get_train_loader(image_path='./data/images/', lemma_path='./data/Flickr8k.lemma.token.txt',batch_size=64):
    train_set = MyDataSet(image_path=image_path, lemma_path=lemma_path, mode='train')
    train_loader = DataLoader(train_set, batch_size=batch_size)

    valid_set = MyDataSet(image_path=image_path, lemma_path=lemma_path,mode='valid')
    valid_loader = DataLoader(valid_set, batch_size=batch_size)
    return train_loader, valid_loader

def get_test_loader(image_path='./data/images/', lemma_path='./data/Flickr8k.lemma.token.txt',batch_size=1):
    set = MyDataSet(image_path=image_path, lemma_path=lemma_path, mode='test')
    loader = DataLoader(set, batch_size=1)
    return loader


if __name__ == '__main__':
    loader, _ = get_train_loader()
    for i, t in loader:
        print(i, t)
        assert False

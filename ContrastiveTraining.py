import torch
import torch.nn as nn
from tqdm import tqdm
from models.encoders import ImgEncoder, TextEncoder
from data.data import get_train_loader
import math
from data.Tokens import MyToken
from data.dataUtils import read_into_dic


class TextLinear(nn.Module):
    def __init__(self, input_dim=768, output_dim=512):
        super(TextLinear, self).__init__()
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class ImgLinear(nn.Module):
    def __init__(self, input_dim=2048, output_dim=512):
        super(ImgLinear, self).__init__()
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return self.mlp(x)


def compute_cross_entropy_loss(criterion, logits, device = torch.device('cpu')):
    '''
    :param criterion:
    :param logits:shape N,N
    :return:
    '''
    N, _ = logits.shape
    labels = torch.arange(N, device = device)
    loss_i = criterion(logits.permute(1, 0), labels)
    loss_t = criterion(logits, labels)
    loss = (loss_t + loss_i) / 2
    return loss


def contrastive_training(
        img_encoder,
        text_encoder,
        loader,
        contrastive_dim=384,
        lr=1e-3,
        t=0,
        label_smoothing=0,
        weight_decay=0,
        total_epoch=100,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)
    img_linear = ImgLinear(output_dim=contrastive_dim).to(device)
    text_linear = TextLinear(output_dim=contrastive_dim).to(device)
    optimizer = torch.optim.AdamW(
        [{'params': img_encoder.parameters()}, {'params': text_encoder.parameters()},
         {'params': img_linear.parameters()}, {'params': text_linear.parameters()}], lr=lr,
        weight_decay=weight_decay)
    img_linear.train()
    img_encoder.train()
    text_linear.train()
    text_encoder.train()

    for epoch in range(total_epoch):
        epoch_loss = 0
        for img, text in tqdm(loader):
            img = img.to(device)
            text = text_encoder(text)
            text = text_linear(text)
            img = img_encoder(img)
            img = img_linear(img)

            logits = (img @ text.T) * math.exp(t)
            loss = compute_cross_entropy_loss(criterion, logits, device=device)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss /= len(loader)
        print(f'epoch {epoch}, loss = {epoch_loss}')
        torch.save(img_encoder.state_dict(), 'image_encoder.pth')
        torch.save(text_encoder.state_dict(), 'text_encoder.pth')
        torch.save(img_linear.state_dict(), 'img_linear.pth')
        torch.save(text_linear.state_dict(), 'text_linear.pth')


if __name__ == '__main__':
    tokenizer = MyToken()
    tokenizer.load_dic()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_encoder = ImgEncoder().to(device)
    text_encoder = TextEncoder(device=device, tokenizer=tokenizer).to(device)
    train_loader, _ = get_train_loader(batch_size=5)
    contrastive_training(img_encoder, text_encoder, train_loader)

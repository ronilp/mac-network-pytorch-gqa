import pickle

import torch
from torch import nn
from torch import optim

from model_gqa import MACNetwork

batch_size = 64
n_epoch = 20
dim = 512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    moving_loss = 0

    net.train()
    image = torch.randn(batch_size, 1024, 14, 14, device=device)
    question = torch.randint(0, 28, (batch_size, 30), dtype=torch.int64, device=device)
    answer = torch.randint(0, 28, (batch_size,), dtype=torch.int64, device=device)
    q_len = torch.tensor([30] * batch_size, dtype=torch.int64, device=device)

    for i in range(30):
        net.zero_grad()
        output = net(image, question, q_len)
        loss = criterion(output, answer)
        loss.backward()
        optimizer.step()
        correct = output.detach().argmax(1) \
                == answer
        correct = torch.tensor(correct, dtype=torch.float32).sum() \
                    / batch_size

        if moving_loss == 0:
            moving_loss = correct

        else:
            moving_loss = moving_loss * 0.99 + correct * 0.01

if __name__ == '__main__':
    with open('data/dic.pkl', 'rb') as f:
        dic = pickle.load(f)

    n_words = len(dic['word_dic']) + 1
    n_answers = len(dic['answer_dic'])

    net = MACNetwork(n_words, dim).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    train()
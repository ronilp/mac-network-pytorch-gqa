import multiprocessing
import pickle
import sys

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CLEVR, collate_data, transform, GQA
from model_gqa import MACNetwork

batch_size = 128
n_epoch = 25
dim_dict = {'CLEVR': 512,
            'gqa': 2048}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def train(epoch, dataset_type, lang='en'):
    if dataset_type == "CLEVR":
        dataset_object = CLEVR('data/CLEVR', transform=transform, lang=lang)
    else:
        dataset_object = GQA('data/gqa', transform=transform)

    train_set = DataLoader(
        dataset_object, batch_size=batch_size, num_workers=multiprocessing.cpu_count(), collate_fn=collate_data
    )

    dataset = iter(train_set)
    pbar = tqdm(dataset)
    moving_loss = 0

    net.train(True)
    for image, question, q_len, answer in pbar:
        image, question, answer = (
            image.to(device),
            question.to(device),
            answer.to(device),
        )

        net.zero_grad()
        output = net(image, question, q_len)
        loss = criterion(output, answer)
        loss.backward()
        optimizer.step()
        correct = output.detach().argmax(1) == answer
        correct = torch.tensor(correct, dtype=torch.float32).sum() / batch_size

        if moving_loss == 0:
            moving_loss = correct
        else:
            moving_loss = moving_loss * 0.99 + correct * 0.01

        pbar.set_description('Epoch: {}; Loss: {:.8f}; Acc: {:.5f}'.format(epoch + 1, loss.item(), moving_loss))

        accumulate(net_running, net)

    dataset_object.close()


def valid(epoch, dataset_type,lang='en'):
    if dataset_type == "CLEVR":
        dataset_object = CLEVR('data/CLEVR', 'val', transform=None, lang=lang)
    else:
        dataset_object = GQA('data/gqa', 'val', transform=None)

    valid_set = DataLoader(
        dataset_object, batch_size=4*batch_size, num_workers=multiprocessing.cpu_count(), collate_fn=collate_data
    )
    dataset = iter(valid_set)

    net_running.train(False)
    correct_counts = 0
    total_counts = 0
    running_loss = 0.0
    batches_done = 0
    with torch.no_grad():
        pbar = tqdm(dataset)
        for image, question, q_len, answer in pbar:
            image, question, answer = (
                image.to(device),
                question.to(device),
                answer.to(device),
            )

            output = net_running(image, question, q_len)
            loss = criterion(output, answer)
            correct = output.detach().argmax(1) == answer
            running_loss += loss.item()

            batches_done += 1
            for c in correct:
                if c:
                    correct_counts += 1
                total_counts += 1

            pbar.set_description('Epoch: {}; Loss: {:.8f}; Acc: {:.5f}'.format(epoch + 1, loss.item(), correct_counts / batches_done))

    with open('log/log_{}.txt'.format(str(epoch + 1).zfill(2)), 'w') as w:
        w.write('{:.5f}\n'.format(correct_counts / total_counts))

    print('Validation Accuracy: {:.5f}'.format(correct_counts / total_counts))
    print('Validation Loss: {:.8f}'.format(running_loss / total_counts))

    dataset_object.close()


if __name__ == '__main__':
    dataset_type = sys.argv[1]
    lang = sys.argv[2]
    with open(f'data/{dataset_type}_dic.pkl', 'rb') as f:
        dic = pickle.load(f)

    n_words = len(dic['word_dic']) + 1
    n_answers = len(dic['answer_dic'])

    net = MACNetwork(n_words, dim_dict[dataset_type], classes=n_answers, max_step=4).to(device)
    net_running = MACNetwork(n_words, dim_dict[dataset_type], classes=n_answers, max_step=4).to(device)
    accumulate(net_running, net, 0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    for epoch in range(n_epoch):
        train(epoch, dataset_type,lang=lang)
        valid(epoch, dataset_type,lang=lang)

        with open(f'checkpoint/checkpoint_{lang}.model'.format(str(epoch + 1).zfill(2)), 'wb') as f:
            torch.save(net_running.state_dict(), f)

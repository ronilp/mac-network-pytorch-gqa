import multiprocessing
import pickle
import sys

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import BASE_LR, TRAIN_EPOCHS, BATCH_SIZE, DEVICE, MAX_STEPS, USE_SELF_ATTENTION, \
    USE_MEMORY_GATE, MAC_UNIT_DIM
from dataset import CLEVR, collate_data, transform, GQA
from model_gqa import MACNetwork


def train(epoch, dataset_type):
    if dataset_type == "CLEVR":
        dataset_object = CLEVR('data/CLEVR_v1.0', transform=transform)
    else:
        dataset_object = GQA('data/gqa', transform=transform)

    train_set = DataLoader(
        dataset_object, batch_size=BATCH_SIZE, num_workers=multiprocessing.cpu_count(), collate_fn=collate_data
    )

    dataset = iter(train_set)
    pbar = tqdm(dataset)
    running_loss = 0
    correct_counts = 0
    total_counts = 0

    net.train()
    for image, question, q_len, answer in pbar:
        image, question, answer = (
            image.to(DEVICE),
            question.to(DEVICE),
            answer.to(DEVICE),
        )

        net.zero_grad()
        output = net(image, question, q_len)
        loss = criterion(output, answer)
        loss.backward()
        optimizer.step()

        correct = output.detach().argmax(1) == answer
        correct_counts += sum(correct).item()
        total_counts += image.size(0)

        correct = correct.clone().type(torch.FloatTensor).detach().sum() / BATCH_SIZE
        running_loss += loss.item() / BATCH_SIZE

        pbar.set_description(
            'Epoch: {}; Loss: {:.8f}; Acc: {:.5f}'.format(epoch + 1, loss.item(), correct))

    print('Training loss: {:8f}, accuracy: {:5f}'.format(running_loss / len(train_set.dataset),
                                                         correct_counts / total_counts))

    dataset_object.close()


def valid(epoch, dataset_type):
    if dataset_type == "CLEVR":
        dataset_object = CLEVR('data/CLEVR_v1.0', 'val', transform=None)
    else:
        dataset_object = GQA('data/gqa', 'val', transform=None)

    valid_set = DataLoader(dataset_object, batch_size=BATCH_SIZE, num_workers=multiprocessing.cpu_count(),
                           collate_fn=collate_data)
    dataset = iter(valid_set)

    net.eval()
    correct_counts = 0
    total_counts = 0
    running_loss = 0.0
    with torch.no_grad():
        pbar = tqdm(dataset)
        for image, question, q_len, answer in pbar:
            image, question, answer = (
                image.to(DEVICE),
                question.to(DEVICE),
                answer.to(DEVICE),
            )

            output = net(image, question, q_len)
            loss = criterion(output, answer)
            correct = output.detach().argmax(1) == answer
            correct_counts += sum(correct).item()
            total_counts += image.size(0)
            running_loss += loss.item() / BATCH_SIZE

            pbar.set_description(
                'Epoch: {}; Loss: {:.8f}; Acc: {:.5f}'.format(epoch + 1, loss.item(), correct_counts / total_counts))

    with open('log/log_{}.txt'.format(str(epoch + 1).zfill(2)), 'w') as w:
        w.write('{:.5f}\n'.format(correct_counts / total_counts))

    print('Training loss: {:8f}, accuracy: {:5f}'.format(running_loss / len(valid_set.dataset),
                                                         correct_counts / total_counts))

    dataset_object.close()


if __name__ == '__main__':
    dataset_type = sys.argv[1]
    with open(f'data/{dataset_type}_dic.pkl', 'rb') as f:
        dic = pickle.load(f)

    n_words = len(dic['word_dic']) + 1
    n_answers = len(dic['answer_dic'])

    net = MACNetwork(n_words, MAC_UNIT_DIM[dataset_type], classes=n_answers, max_step=MAX_STEPS,
                     self_attention=USE_SELF_ATTENTION, memory_gate=USE_MEMORY_GATE).to(DEVICE)
    net = nn.DataParallel(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=BASE_LR)

    for epoch in range(TRAIN_EPOCHS):
        train(epoch, dataset_type)
        valid(epoch, dataset_type)

        with open('checkpoint/checkpoint_{}.model'.format(str(epoch + 1).zfill(2)), 'wb') as f:
            torch.save(net.state_dict(), f)

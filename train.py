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


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


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
    moving_loss = 0

    net.train(True)
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
        correct = torch.tensor(correct, dtype=torch.float32).sum() / BATCH_SIZE

        if moving_loss == 0:
            moving_loss = correct
        else:
            moving_loss = moving_loss * 0.99 + correct * 0.01

        pbar.set_description('Epoch: {}; Loss: {:.8f}; Acc: {:.5f}'.format(epoch + 1, loss.item(), moving_loss))
        accumulate(net_running, net)

    dataset_object.close()


def valid(epoch, dataset_type):
    if dataset_type == "CLEVR":
        dataset_object = CLEVR('data/CLEVR_v1.0', 'val', transform=None)
    else:
        dataset_object = GQA('data/gqa', 'val', transform=None)

    valid_set = DataLoader(
        dataset_object, batch_size=BATCH_SIZE, num_workers=multiprocessing.cpu_count(), collate_fn=collate_data
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
                image.to(DEVICE),
                question.to(DEVICE),
                answer.to(DEVICE),
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

            pbar.set_description(
                'Epoch: {}; Loss: {:.8f}; Acc: {:.5f}'.format(epoch + 1, loss.item(), correct_counts / batches_done))

    with open('log/log_{}.txt'.format(str(epoch + 1).zfill(2)), 'w') as w:
        w.write('{:.5f}\n'.format(correct_counts / total_counts))

    print('Validation Accuracy: {:.5f}'.format(correct_counts / total_counts))
    print('Validation Loss: {:.8f}'.format(running_loss / total_counts))

    dataset_object.close()


if __name__ == '__main__':
    dataset_type = sys.argv[1]
    with open(f'data/{dataset_type}_dic.pkl', 'rb') as f:
        dic = pickle.load(f)

    n_words = len(dic['word_dic']) + 1
    n_answers = len(dic['answer_dic'])

    net = MACNetwork(n_words, MAC_UNIT_DIM[dataset_type], classes=n_answers, max_step=MAX_STEPS,
                     self_attention=USE_SELF_ATTENTION, memory_gate=USE_MEMORY_GATE).to(DEVICE)
    net_running = MACNetwork(n_words, MAC_UNIT_DIM[dataset_type], classes=n_answers, max_step=MAX_STEPS,
                             self_attention=USE_SELF_ATTENTION, memory_gate=USE_MEMORY_GATE).to(DEVICE)

    net = nn.DataParallel(net)
    net_running = nn.DataParallel(net_running)
    accumulate(net_running, net, 0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=BASE_LR)

    for epoch in range(TRAIN_EPOCHS):
        train(epoch, dataset_type)
        valid(epoch, dataset_type)

        with open('checkpoint/checkpoint_{}.model'.format(str(epoch + 1).zfill(2)), 'wb') as f:
            torch.save(net_running.state_dict(), f)

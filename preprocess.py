import os
import sys
import json
import pickle

import nltk
import tqdm
from torchvision import transforms
from PIL import Image
from transforms import Scale

image_index = {'CLEVR': 'image_filename',
               'gqa': 'imageId'}


def process_question(root, split, word_dic=None, answer_dic=None, dataset_type='CLEVR'):
    if word_dic is None:
        word_dic = {}

    if answer_dic is None:
        answer_dic = {}

    with open(os.path.join(root, 'questions', f'{dataset_type}_{split}_questions.json')) as f:
        data = json.load(f)

    result = []
    word_index = 1
    answer_index = 0
    i = 0
    NUM_QUESTIONS = 100

    for question in tqdm.tqdm(data['questions']):
        if i > NUM_QUESTIONS:
            break

        i += 1
        words = nltk.word_tokenize(question['question'])
        question_token = []

        for word in words:
            try:
                question_token.append(word_dic[word])

            except:
                question_token.append(word_index)
                word_dic[word] = word_index
                word_index += 1

        answer_word = question['answer']

        try:
            answer = answer_dic[answer_word]
        except:
            answer = answer_index
            answer_dic[answer_word] = answer_index
            answer_index += 1

        result.append((question[image_index[dataset_type]], question_token, answer))

    with open(f'data/{dataset_type}_{split}.pkl', 'wb') as f:
        pickle.dump(result, f)

    return word_dic, answer_dic


if __name__ == '__main__':
    root = sys.argv[1]
    dataset_type = 'gqa'
    word_dic, answer_dic = process_question(root, 'train', dataset_type=dataset_type)
    process_question(root, 'val', word_dic, answer_dic, dataset_type=dataset_type)

    with open(f'data/{dataset_type}_dic.pkl', 'wb') as f:
        pickle.dump({'word_dic': word_dic, 'answer_dic': answer_dic}, f)

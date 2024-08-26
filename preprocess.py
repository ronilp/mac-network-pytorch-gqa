import os
import sys
import json
import pickle
import jieba
import nltk
import tqdm
from torchvision import transforms
from PIL import Image
from transforms import Scale
import MeCab
from konlpy.tag import Okt
image_index = {'CLEVR': 'image_filename',
               'gqa': 'imageId'}


def process_question(root, split, word_dic=None, answer_dic=None, dataset_type='CLEVR',lang='en'):
    if word_dic is None:
        word_dic = {}

    if answer_dic is None:
        answer_dic = {}

    with open(os.path.join(root, 'questions', f'mini_{dataset_type}_{split}_questions.json'), encoding='utf-8') as f:
        data = json.load(f)

    result = []
    word_index = 1
    answer_index = 0

    for question in tqdm.tqdm(data['questions']):
        if lang == 'en':
            words = nltk.word_tokenize(question['question'])
        elif lang =='vi':
            words = nltk.word_tokenize(question['translated']['vietnamese'])
        elif lang == 'zh':
            words = jieba.lcut(question['translated']['chinese'])
        elif lang == 'ja':
            mecab = MeCab.Tagger("-Owakati")
            words = mecab.parse(question['translated']['japanese']).strip().split()
        elif lang == 'ko':
            okt = Okt()
            words = okt.morphs(question['translated']['korean'])
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

    with open(f'data/{dataset_type}_{split}_{lang}.pkl', 'wb') as f:
        pickle.dump(result, f)

    return word_dic, answer_dic


if __name__ == '__main__':
    dataset_type = sys.argv[1]
    root = sys.argv[2]
    lang = sys.argv[3]
    word_dic, answer_dic = process_question(root, 'train', dataset_type=dataset_type)
    process_question(root, 'val', word_dic, answer_dic, dataset_type=dataset_type)

    with open(f'data/{dataset_type}_{lang}_dic.pkl', 'wb') as f:
        pickle.dump({'word_dic': word_dic, 'answer_dic': answer_dic}, f)

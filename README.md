# mac-network-pytorch
Memory, Attention and Composition (MAC) Network for CLEVR/GQA from Compositional Attention Networks for Machine Reasoning (https://arxiv.org/abs/1803.03067) implemented in PyTorch

Requirements:
* Python 3.6
* PyTorch 1.0.1
* torch-vision
* Pillow
* nltk
* tqdm

To train:

1. Download and extract either <br />
CLEVR v1.0 dataset from http://cs.stanford.edu/people/jcjohns/clevr/ or <br />
GQA dataset from https://cs.stanford.edu/people/dorarad/gqa/download.html <br />
<br />
For GQA
```
cd data
mkdir gqa && cd gqa
wget https://nlp.stanford.edu/data/gqa/data1.2.zip
unzip data1.2.zip

mkdir questions
mv balanced_train_data.json questions/gqa_train_questions.json
mv balanced_val_data.json questions/gqa_val_questions.json
mv balanced_testdev_data.json questions/gqa_testdev_questions.json
cd ..

wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
wget http://nlp.stanford.edu/data/gqa/objectFeatures.zip
unzip objectFeatures.zip
cd ..
```

2. Preprocessing question data and extracting image features using ResNet 101 (Not required for GQA)
<br />
For CLEVR
<br />
a. Extract image features

```
python image_feature.py data/CLEVR_v1.0
```

b. Preprocess questions

```
python preprocess.py CLEVR data/CLEVR_v1.0
```

For GQA<br />
a. Merge object features (this may take some time)

```
python merge.py --objects
mv data/gqa_objects.hdf5 data/gqa_features.hdf5
```

b. Preprocess questions

```
python preprocess.py gqa data/gqa
```

!CAUTION! the size of file created by image_feature.py is very large! You may use hdf5 compression, but it will slow down feature extraction.
<br />
3. Run train.py with dataset type as argument (gqa or CLEVR)

```
python train.py gqa
```

CLEVR -> This implementation produces 95.75% accuracy at epoch 10, 96.5% accuracy at epoch 20.
<br />
Parts to the code borrowed from https://github.com/rosinality/mac-network-pytorch and <br />
https://github.com/stanfordnlp/mac-network.

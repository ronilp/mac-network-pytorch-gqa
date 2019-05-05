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

python merge.py --name objects
mv data/gqa_objects.hdf5 data/gqa_features.hdf5

python preprocess.py gqa data/gqa

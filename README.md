<h1><a id="macnetworkpytorch_0"></a>mac-network-pytorch</h1>
<p>Memory, Attention and Composition (MAC) Network for CLEVR/GQA from Compositional Attention Networks for Machine Reasoning (<a href="https://arxiv.org/abs/1803.03067">https://arxiv.org/abs/1803.03067</a>) implemented in PyTorch</p>
<p>Requirements:</p>
<ul>
<li>Python 3.6</li>
<li>PyTorch 1.0.1</li>
<li>torch-vision</li>
<li>Pillow</li>
<li>nltk</li>
<li>tqdm</li>
<li>block.bootstrap.pytorch murel.bootstrap.pytorch</li>
</ul>
<p>To train:</p>
<ol>
<li>Download and extract either<br>
CLEVR v1.0 dataset from <a href="http://cs.stanford.edu/people/jcjohns/clevr/">http://cs.stanford.edu/people/jcjohns/clevr/</a> or<br>
GQA dataset from <a href="https://cs.stanford.edu/people/dorarad/gqa/download.html">https://cs.stanford.edu/people/dorarad/gqa/download.html</a></li>
</ol>
<p>For GQA</p>
<pre><code>cd data
mkdir gqa &amp;&amp; cd gqa
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
</code></pre>
<ol start="2">
<li>Preprocessing question data and extracting image features using ResNet 101 (Not required for GQA)<br>
For CLEVR<br>
a. Extract image features</li>
</ol>
<pre><code>python image_feature.py data/CLEVR_v1.0
</code></pre>
<p>b. Preprocess questions</p>
<pre><code>python preprocess.py CLEVR data/CLEVR_v1.0
</code></pre>
<p>For GQA<br>
a. Merge object features (this may take some time)</p>
<pre><code>python merge.py --name objects
mv data/gqa_objects.hdf5 data/gqa_features.hdf5
</code></pre>
<p>b. Preprocess questions</p>
<pre><code>python preprocess.py gqa data/gqa
</code></pre>
<p>!CAUTION! the size of file created by image_feature.py is very large! You may use hdf5 compression, but it will slow down feature extraction.</p>
<ol start="3">
<li>Run <a href="http://train.py">train.py</a> with dataset type as argument (gqa or CLEVR)</li>
</ol>
<pre><code>python train.py gqa
</code></pre>
<p>CLEVR -&gt; This implementation produces 95.75% accuracy at epoch 10, 96.5% accuracy at epoch 20.</p>
<p>Parts of the code borrowed from <a href="https://github.com/rosinality/mac-network-pytorch">https://github.com/rosinality/mac-network-pytorch</a> and<br>
<a href="https://github.com/stanfordnlp/mac-network">https://github.com/stanfordnlp/mac-network</a>.</p>

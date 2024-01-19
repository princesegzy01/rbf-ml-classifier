import pandas as pd
import numpy as np
# import RBFLayer from rbf
import os
import sys
from nltk.corpus import stopwords
import logging
logging.getLogger('tensorflow').setLevel(logging.INFO)
import unicodedata
import re
from rbf import RBFLayer

X = np.load('k49-test-imgs.npz')['arr_0']
y = np.load('k49-test-labels.npz')['arr_0']


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def clean_stopwords_shortwords(w):
    stopwords_list=stopwords.words('english')
    words = w.split() 
    clean_words = [word for word in words if (word not in stopwords_list) and len(word) > 2]
    return " ".join(clean_words) 

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w=clean_stopwords_shortwords(w)
    w=re.sub(r'@\w+', '',w)
    return w


# df = pd.read_csv("data/imsdb.csv", on_bad_lines='skip')
# for index, row in df.iterrows():
#    f = open(os.path.join('data/content', str(row['id']) + ".txt"), "r")
#    content = f.read()

#    print(preprocess_sentence(content))

# sys.exit(0)

# print(X)

y = (y <= 25).astype(int)

from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.losses import binary_crossentropy

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(RBFLayer(10, 0.5))
model.add(Dense(1, activation='sigmoid', name='foo'))

model.compile(optimizer='rmsprop', loss=binary_crossentropy)
model.fit(X, y, batch_size=256, epochs=10)
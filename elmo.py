# -*- coding: utf-8 -*-
"""
MIT Licence

Zoghbi Abderraouf
Change data to your location 
"""

from sklearn.metrics import hamming_loss
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from textattack.augmentation import EasyDataAugmenter

import sys
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import multilabel_confusion_matrix, classification_report

import  numpy as np
# import necessary libraries

from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional

  
# Load pre trained ELMo model
  

# import keras_metrics

import numpy as np
import os
import sys
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
import logging
logging.getLogger('tensorflow').setLevel(logging.INFO)
import unicodedata
import re
from sklearn.preprocessing import LabelEncoder
from numpy import random
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from keras.initializers import RandomUniform, Initializer, Constant
from keras import backend as K
from keras.preprocessing.text import Tokenizer



from tensorflow.keras.layers import Layer, InputSpec
from keras.initializers import Initializer
from keras.models import Sequential 
from keras.layers.core import Dense
from keras.layers import  Flatten


class_type = "transformerx"

    
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow_hub as hub
# elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.utils import to_categorical
from keras import backend as K
# from keras.engine.topology import Layer
from keras.initializers import RandomUniform, Initializer, Constant
from tensorflow import keras
from keras.layers import Activation
from keras.initializers import Initializer
import tensorflow as tf
from tensorflow import keras
from keras import layers
    
def ELMoEmbedding(x):
    # embed = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
    embed = hub.Module("embeeding/elmo/v3", trainable=True)
    v = embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]
    print(v)
    return v


def build_model(): 
    input_text = tf.keras.layers.Input(shape=(1,), dtype="string")
    embedding = tf.keras.layers.Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)
    dense = Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(embedding)
    pred = Dense(1, activation='sigmoid')(dense)
    model = tf.keras.Model(inputs=[input_text], outputs=pred)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

model_elmo = build_model()


class attention(Layer):
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences

        super(attention,self).__init__()

    def build(self, input_shape):
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1), initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1), initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1))
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1)) 

        super(attention,self).build(input_shape)


    def call(self, x):
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        if self.return_sequences:

            return output
        return K.sum(output, axis=1)
        
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def clean_stopwords_shortwords(w):
    stopwords_list=stopwords.words('english')
    words = w.split() 
    clean_words = [word for word in words if (word not in stopwords_list) and len(word) > 2]
    return " ".join(clean_words) 

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    # w = re.sub(r"([?.!,¿])", r" ", w)
    w = re.sub(r'[" "]+', " ", w)
    # w = re.sub(r'["i\'m"]+', "i am", w)
    # w = re.sub(r'["... "]+', " ", w)
    # w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    # w=clean_stopwords_shortwords(w)
    w=re.sub(r'@\w+', '',w)
    w = re.sub('\n', '', w)
    w = re.sub('--', ' ', w)
    w = re.sub('\(', '', w)
    w = re.sub('\)', '', w)
    w = re.sub("cont\'d",'continued', w)
    

    return w

def format_document(document):
    
    # split the document by sentence
    list_text = document.split(".")
    
    # remove empty string in list
    list_text = [i.strip() for i in list_text if i]
    # print(len(list_text))
    
    # return ' '.join(list_text[0:100]) 
    
    return list_text

def init_memval(cluster_n, data_n):  
    U = np.random.random((cluster_n, data_n))
    val = sum(U)
    U = np.divide(U,np.dot(np.ones((cluster_n,1)),np.reshape(val,(1,data_n))))
    return U
   

class InitCentersRandom(Initializer):
    """ Initializer for initialization of centers of RBF network
        as random samples from the given data set.
    # Arguments
        X: matrix, dataset to choose the centers from (random rows
          are taken as centers)
    """

    def __init__(self, X):
        self.X = X

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]
        idx = np.random.randint(self.X.shape[0], size=shape[0])
        return self.X[idx, :]


class RBFLayer(Layer):
    """ Layer of Gaussian RBF units.
    # Example
    ```python
        model = Sequential()
        model.add(RBFLayer(10,
                           initializer=InitCentersRandom(X),
                           betas=1.0,
                           input_shape=(1,)))
        model.add(Dense(1))
    ```
    # Arguments
        output_dim: number of hidden units (i.e. number of outputs of the
                    layer)
        initializer: instance of initiliazer to initialize centers
        betas: float, initial value for betas
    """

    def __init__(self, output_dim, initializer=None, betas=1.0, **kwargs):

        self.output_dim = output_dim
        self.init_betas = betas
        if not initializer:
            self.initializer = RandomUniform(0.0, 1.0)
        else:
            self.initializer = initializer
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=True)
        self.betas = self.add_weight(name='betas',
                                     shape=(self.output_dim,),
                                     initializer=Constant(
                                         value=self.init_betas),
                                     # initializer='ones',
                                     trainable=True)

        super(RBFLayer, self).build(input_shape)

    def call(self, x):

        C = K.expand_dims(self.centers)
        H = K.transpose(C-K.transpose(x))
        return K.exp(-self.betas * K.sum(H**2, axis=1))

        # C = self.centers[np.newaxis, :, :]
        # X = x[:, np.newaxis, :]

        # diffnorm = K.sum((C-X)**2, axis=-1)
        # ret = K.exp( - self.betas * diffnorm)
        # return ret

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(RBFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class InitCentersKMeans(Initializer):
    """ Initializer for initialization of centers of RBF network
        by clustering the given data set.
    # Arguments
        X: matrix, dataset
    """

    def __init__(self, X, max_iter=100):
        self.X = X
        self.max_iter = max_iter

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]

        n_centers = shape[0]

        km = KMeans(n_clusters=n_centers, max_iter=self.max_iter, verbose=0)
        km.fit(self.X)
        return km.cluster_centers_


DATA_SIZE=1000000

def RBFModel(X_train, y_train, tokenizer):
    
    vocab_size = len(tokenizer.vocab.keys()) * 1
    
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(vocab_size)
    print(X_train.shape)
    print("***************************")
    
    
    model = Sequential()
    # rbflayer = RBFLayer(X_train.shape[0], initializer=InitCentersKMeans(X_train), betas=3.0,input_shape=(X_train.shape[1],))
    rbflayer = RBFLayer(vocab_size, initializer=InitCentersRandom(X_train), betas=3.0,input_shape=(X_train.shape[1],))
    # model.add(Embedding(vocab_size, 100, input_length=384, trainable=True))    
    # model.add(attention(return_sequences=True)) # receive 3D and output 3D  
    # model.add(Flatten())
    # rbflayer = RBFLayer(100, initializer=InitCentersRandom(X_train), betas=3.0,input_shape=(X_train.shape[1],))
    # rbflayer = RBFLayer(X_train.shape[0], initializer=InitCentersKMeans(X_train), betas=3.0,input_shape=(X_train.shape[1],))

    model.add(rbflayer)
    model.add(Dense(200, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(20, activation='relu'))
    # model.add(layers.GlobalAveragePooling1D())
    model.add(Dense(y_train.shape[1], activation='sigmoid'))
    return model

def Custom_Hamming_Loss(y_true, y_pred):
  return K.mean(y_true*(1-y_pred)+(1-y_true)*y_pred)

def mini_accuracy(yp, yt):
    
    acc = 0
    inner_accuracy = 0
    
    for i in range(len(yp)):
        _current_yp = yp[i]
        _current_yt = yt[i]
        
        total_match = 0
        
        
        for x in range(len(_current_yt)):

            if _current_yp[x] is _current_yt[x]:
                total_match = total_match + 1
                
        inner_accuracy = inner_accuracy + total_match/len(_current_yp)
    acc = inner_accuracy / len(yp)
    return acc

def elmo_vectors(x):
    elmo = hub.Module("embeeding/elmo/v3", trainable=False)
    embeddings = elmo(x, signature="default", as_dict=True)["elmo"]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        # return average of ELMo features
        return sess.run(tf.reduce_mean(embeddings,1))

        
n_target = {
    'Security' : [], 
    'Technology' : [], 
    'Earth-File' : [], 
    'Entertainment' : [], 
    'Relationship' : [], 
}

corpus = []
target = []


df = pd.read_csv("data/imsdb.csv", on_bad_lines='skip')
for index, row in df.iterrows():
    
    
    # print(str(index))
    # # print(n_target)
    # n_target['Security'].append(0)
    # n_target['Technology'].append(0)
    # n_target['Earth-File'].append(0)
    # n_target['Entertainment'].append(0)
    # n_target['Relationship'].append(0)


    # if not str(row['class_a']) == "nan" : 
    #     n_target[row['class_a']][index] = 1 
        
    # if not str(row['class_b']) == "nan" : 
    #     n_target[row['class_b']][index] = 1 
        
    # if not str(row['class_c']) == "nan" : 
    #     n_target[row['class_c']][index] = 1 
            
    # if not str(row['class_d']) == "nan": 
    #     n_target[row['class_d']][index] = 1 
        
    # if not str(row['class_e']) == "nan" : 
    #     n_target[row['class_e']][index] = 1 
        
    # if not str(row['class_f']) == "nan" : 
    #     n_target[row['class_f']][index] = 1   
    
    # if type(row['class_a']) != str:
    #     print("Continue this shit")
    #     continue    

    print(str(index) + " >>>>>>>>>>>>>>>>>>>")
    
    f = open(os.path.join('data/content', str(row['id']) + ".txt"), "r")
    clean_data = preprocess_sentence(f.read())
    
    
    document_split_step = 500
    list_clean_data = format_document(clean_data)
    
    for i in range(0,len(list_clean_data),document_split_step):
        # print(str(i) + " >>>>>>>>>>>>>>>>>>>")
     
        
        document  = list_clean_data[i:i+document_split_step]
        document = ' '.join(document)
        
        corpus.append(document)
        # target.append(row['class_a'])
        
        
        # augmented_list_text = eda_aug.augment(document)
        
        # print(augmented_list_text)
        # sys.exit(0)
        
        # print(n_target)
        n_target['Security'].append(0)
        n_target['Technology'].append(0)
        n_target['Earth-File'].append(0)
        n_target['Entertainment'].append(0)
        n_target['Relationship'].append(0)


        if not str(row['class_a']) == "nan" : 
            n_target[row['class_a']][index] = 1 
            
        if not str(row['class_b']) == "nan" : 
            n_target[row['class_b']][index] = 1 
            
        if not str(row['class_c']) == "nan" : 
            n_target[row['class_c']][index] = 1 
                
        if not str(row['class_d']) == "nan": 
            n_target[row['class_d']][index] = 1 
            
        if not str(row['class_e']) == "nan" : 
            n_target[row['class_e']][index] = 1 
            
        if not str(row['class_f']) == "nan" : 
            n_target[row['class_f']][index] = 1   
        
        if type(row['class_a']) != str:
            print("Continue this shit")
            continue    

    
        # print(len(document))
    
    # sys.exit(0)
    
    # corpus.append(clean_data)
    # target.append(row['class_a'])
    
    
   
    # augmented_list_text = eda_aug.augment(clean_data)
    # augmented_list_text = eda_aug.augment("This is a simple conversion")
    # print(len(augmented_list_text))
    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    if int(index) == DATA_SIZE:
        break

new_y = pd.DataFrame(n_target)

import tensorflow.compat.v1 as tf
import pickle


# elmo = hub.Module("embeeding/elmo/v2", trainable=False)
# embeddings = elmo(["the cat is on the mat", "what are you doing in evening"], signature="default", as_dict=True)["elmo"]
# # embeddings = elmo(corpus, signature="default", as_dict=True)["elmo"]
# with tf.compat.v1.Session() as sess:
#     # sess.run(tf.compat.v1.global_variables_initializer())
#     # sess.run(tf.compat.v1.tables_initializer())
    
#     # print(len(corpus))
#     # history = model.fit(np.asarray(df_train.clean_tweet),y_train,epochs=5,batch_size = 2,validation_split=0.2)
#     # history = model_elmo.fit(corpus, new_y,epochs=3,batch_size = 32,validation_split=0.2)
#     # model_elmo.save_weights('../models/model_elmo_weights.h5')
    
#     sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
#     message_embeddings = sess.run(embeddings)
    
    
#     print(message_embeddings)
list_train = [corpus[i:i+1] for i in range(0,len(corpus),1)]

# print(list_train)
print(">>>>>>>>>>>>>>>>>>>>>>>>>")
print(len(list_train))   

# elmo_train = [elmo_vectors(x) for x in list_train]
# print("<<<<<<<<<<<<<<<<<<<<<<<<")
# print(len(elmo_train))


index = 0

elmo_train = []
for x in list_train:
    
    index = index + 1
    print(">>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< " + str(index))
    # print(x)
    elmo_train.append(elmo_vectors(x))


elmo_train_new = np.concatenate(elmo_train, axis = 0)


# save elmo_train_new
pickle_out = open("elmo_train.pickle","wb")
pickle.dump(elmo_train_new, pickle_out)
pickle_out.close()

print(" *************** Done")
sys.exit(0)        
    

# def embed_elmo2(module):
#     import tensorflow.compat.v1 as tf

#     with tf.Graph().as_default():
#         sentences = tf.placeholder(tf.string)
#         embed = hub.Module(module)
#         embeddings = embed(corpus)
#         session = tf.train.MonitoredSession()
#     return lambda x: session.run(embeddings, {sentences: x})

embed_fn = embed_elmo2('embeeding/elmo/v3')

print(embed_fn(["i am sambit", "This is good"]).shape)

    
sys.exit(0)

scaler = MinMaxScaler()
model=scaler.fit(X)
X=model.transform(X)


from sklearn.model_selection import train_test_split


# new_y = new_y[['Security']]

# print(len(X))
# print(len(new_y))
# sys.exit(0)
X_train,X_test,y_train,y_test = train_test_split(X,new_y,test_size = 0.2,random_state=42)#80% train et 20% test

# tokenizer = Tokenizer(num_words=100)
# tokenizer.fit_on_texts(X_train
# X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
# X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')


# BERT + RBF
# model = OrdinaryModel(X_train, y_train, tokenizer)
model = RBFModel(X_train, y_train, tokenizer)
# model = TransformerModel(X_train, y_train, tokenizer)
# model = BiLstmModel(X_train, y_train, tokenizer)


optimizer = keras.optimizers.Adam(learning_rate=0.0001)
# optimizer = keras.optimizers.SGD(momentum=0.01, nesterov=True)
# optimizer = keras.optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), Custom_Hamming_Loss])
history = model.fit(X_train, y_train, batch_size=512, epochs=10)    
model.summary()

# sys.exit()
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5) 

v = [list(map(int,x)) for x in y_pred]

_y_test = y_test.values.tolist()






print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")


# acc =  accuracy_score(y_pred, y_test)
precision = precision_score(y_pred, y_test, average='micro')
recall = recall_score(y_pred, y_test, average='micro')
f1 =  f1_score(y_pred, y_test, average='micro')

cm = multilabel_confusion_matrix(y_test, y_pred)
print("Confusion Matrix : " + str(cm))


    
# print("Classification report")
# label_names = ['Security','Technology','Earth-File','Entertainment','Relationship']
# print(classification_report(y_test, y_pred,target_names=label_names))


print("Precision : " + str(precision))
print("Recall : " +  str(recall))
print("F1 Score : " + str(f1))
print("Mini Accuracy : " + str(mini_accuracy(_y_test, v)))
print("Accuracy : "  + str(accuracy_score(y_pred, y_test)))
sys.exit(0)


import matplotlib.pyplot as plt
plt.plot(history1.history['accuracy'])
plt.plot(history1.history['loss'])
plt.title('train accuracy and loss')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['accuracy', 'loss'], loc='upper left')
plt.show()

# # saving to and loading from file
# z_model = f"Z_model.h5"
# print(f"Save model to file {z_model} ... ", end="")
# model.save(z_model)
# print("OK")

#model already saved in file
# from tensorflow.keras.models import  load_model
# newmodel1= load_model("Zoghbio.h5", custom_objects={'RBFLayer': RBFLayer})
print("OK")

# Evaluate the model on the test data using `evaluate`
# print("Evaluate on test data")
# results = newmodel1.evaluate(X_test, y_test, batch_size=32)
# print("test loss:", results[0])
# print("test accuracy:",results[1]*100,'%')

# y_pred = newmodel1.predict(X_test)
# #Converting predictions to label
# pred = list()
# for i in range(len(y_pred)):
#     pred.append(np.argmax(y_pred[i]))
# #Converting one hot encoded test label to label
# test = list()
# for i in range(len(y_test)):
#     test.append(np.argmax(y_test[i]))

# from sklearn.metrics import accuracy_score
# a = accuracy_score(pred,test)
# print('Test Accuracy is:', a*100)
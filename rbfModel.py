# -*- coding: utf-8 -*-
"""
MIT Licence

Zoghbi Abderraouf
Change data to your location 
"""

from sklearn.metrics import hamming_loss
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
# from textattack.augmentation import EasyDataAugmenter
from keras.utils import plot_model

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
from keras.layers import  Flatten, Reshape


class_type = "transformer"
# eda_aug = EasyDataAugmenter()


if class_type == "transformer":
    
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    

    
else:

    import tensorflow_hub as hub
    elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)

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
    
# def ELMoEmbedding(x):
#     embed = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
#     return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]


# def build_model(): 
#     input_text = Input(shape=(1,), dtype="string")
#     embedding = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)
#     dense = Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(embedding)
#     pred = Dense(1, activation='sigmoid')(dense)
#     model = Model(inputs=[input_text], outputs=pred)
#     model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#     return model

# model_elmo = build_model()


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
        
        
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, trainable=False)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim, trainable=False)
    
    def call(self, x):
        maxlen = tf.shape(x)[-1]                
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
        
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
        
        print(" >> Inner shape " + str(shape))
        print(" >> X Shape " + str(self.X.shape))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
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


DATA_SIZE=10000000

def RBFModel(X_train, y_train, tokenizer):
    
    vocab_size = len(tokenizer.vocab.keys()) * 1
    
    model = Sequential()
    # rbflayer = RBFLayer(X_train.shape[0], initializer=InitCentersKMeans(X_train), betas=3.0,input_shape=(X_train.shape[1],))
    # rbflayer = RBFLayer(X_train.shape[0], initializer=InitCentersRandom(X_train), betas=3.0,input_shape=(X_train.shape[1],))
    # model.add(Embedding(vocab_size, 100, input_length=384, trainable=False))    
    # model.add(attention(return_sequences=True)) # receive 3D and output 3D  
    # model.add(Reshape((100, 384)))
    # model.add(Flatten())
    rbflayer = RBFLayer(vocab_size, initializer=InitCentersRandom(X_train), betas=3.0,input_shape=(X_train.shape[1],))
    # rbflayer = RBFLayer(vocab_size, initializer=InitCentersRandom(X_train), betas=3.0,input_shape=(X_train.shape[1],))
    # rbflayer = RBFLayer(X_train.shape[0], initializer=InitCentersKMeans(X_train), betas=3.0,input_shape=(X_train.shape[1],))


    model.add(rbflayer)
    model.add(Dense(200, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(20, activation='relu'))
    # model.add(layers.GlobalAveragePooling1D())
    model.add(Dense(y_train.shape[1], activation='sigmoid'))
    return model

# def OrdinaryModel(X_train, y_train, tokenzier):
    
#     vocab_size = len(tokenzier.vocab.keys()) * 1
    
    
#     model = Sequential()
#     model.add(Embedding(vocab_size, 100, input_length=384, trainable=False))    
#     model.add(attention(return_sequences=True)) # receive 3D and output 3D
#     # model.add(Dense(100, activation='relu'))
#     model.add(Dense(20, activation='relu'))
#     model.add(layers.GlobalAveragePooling1D())
#     model.add(Dense(y_train.shape[1], activation='sigmoid'))
#     return model

def TransformerModel(x_train, y_train, tokenizer):
  
    embed_dim = 100 #32   Embedding size for each token
    num_heads = 10  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer
    
    vocab_size = len(tokenizer.vocab.keys()) * 1 #/ 1000  # Only consider the top 20k words
    maxlen = 768  # Only consider the first 200 words of each movie review
    
    x_train = pd.DataFrame(x_train)
    
    
    # print(x_train)
    # pd.DataFrame(x_train).to_csv('emb.csv')    
    
    inputs = layers.Input(shape=(x_train.shape[1],))
    # inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    
    
    # randomLabel = np.random.randint(2, size=100)
    
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    # x = layers.Dropout(0.1)(x)
    x = layers.Dense(200, activation="relu")(x)
    x = layers.Dense(100, activation="relu")(x)
    x = layers.Dense(10, activation="relu")(x)
    # outputs = layers.Dense(2, activation="softmax")(x)
    outputs = layers.Dense(y_train.shape[1], activation='sigmoid')(x)
    # outputs = layers.Dense(1, activation='sigmoid')(x)


    model = keras.Model(inputs=inputs, outputs=outputs)

     
    return model


def BiLstmModel(x_train, y_train, tokenzier):    
    
    # emb_dim = 100
    # maxlen = 768
    vocab_size = len(tokenzier.vocab.keys()) * 1
    
    model2 = Sequential()
    model2.add(Embedding(vocab_size, 100, input_length=384, trainable=True))
   
    model2.add(Bidirectional(LSTM(64, return_sequences=True)))
    model2.add(attention(return_sequences=True)) # receive 3D and output 3D
    # model2.add(Dropout(0.5))
    # model2.add(Dense(1, activation='sigmoid')) 
    # model2.add(Flatten())
    model2.add(Dense(200, activation='relu'))
    model2.add(Dense(100, activation='relu'))
    model2.add(Dense(10, activation='relu'))
    
    model2.add(layers.GlobalAveragePooling1D())
    model2.add(Dense(y_train.shape[1], activation='sigmoid'))
    return model2

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

    f = open(os.path.join('data/content', str(row['id']) + ".txt"), "r")
    clean_data = preprocess_sentence(f.read())
    
    
    document_split_step = 500
    list_clean_data = format_document(clean_data)
    
    for i in range(0,len(list_clean_data),document_split_step):
        
        
        total_len = len(clean_data)
        
        # print(">>>>>>>>>>>>>>>>>>>>>>>.. " + str(total_len))
        print(">>>>>>>>>>>>>>>>>>>>>>>.. " + str(i))
        continue
        
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
wordVector = "BERT"
X = ""
tokenizer = None


if wordVector == "BERT":
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    X = embedder.encode(corpus, show_progress_bar=True, normalize_embeddings=True)
    tokens = embedder.tokenize(corpus)
    
    # the tokenizer is just here:
    tokenizer = embedder.tokenizer  # BertTokenizerFast

    # # and the vocabulary itself is there, if needed:
    vocab = tokenizer.vocab  # dict of length 30522



scaler = MinMaxScaler()
model=scaler.fit(X)
X=model.transform(X)

print(X.shape)
sys.exit(0)


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


optimizer = keras.optimizers.Adam(learning_rate=0.001)
# optimizer = keras.optimizers.SGD(momentum=0.01, nesterov=True)
# optimizer = keras.optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)

# model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), Custom_Hamming_Loss])
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', Custom_Hamming_Loss])



history = model.fit(X_train, y_train, batch_size=32, validation_split=0.2,  epochs=10)    
model.summary()


plot_model(model, to_file='model_result/rbf.png', show_shapes=True, show_dtype=True, expand_nested=True)
# or save to csv: 
hist_csv_file = 'model_result/rbf.csv'
# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history) 
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
    
    

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
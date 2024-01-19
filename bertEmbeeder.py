import numpy as np
import math
import re
import pandas as pd
from bs4 import BeautifulSoup
import random

# try:
#     tensorflow_version 2.x
# except Exception:
#     pass
import tensorflow as tf

import tensorflow_hub as hub
from tensorflow.keras import layers
import bert

def clean_tweet(tweet):
  tweet = BeautifulSoup(tweet, 'lxml').get_text()
  tweet = re.sub(r"@[A-Za-z0-9]+", ' ', tweet)
  tweet = re.sub(r"https?://[A-Za-z0-9./]+", ' ', tweet)
  tweet = re.sub(r"[^a-zA-Z.!?']", ' ', tweet)
  tweet = re.sub(r" +", ' ', tweet)
  return tweet

def encode_sentence(sent):
    return ["[CLS]"] + tokenizer.tokenize(sent) + ["[SEP]"]

data = [
    'This is a very good tweet to 34 start with',
    'tweets are good source of dataset'
]

values = [0,1]

data_clean = (clean_tweet(tweet) for tweet in data)

# print(data_clean))

# Tokenize dataset
FullTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=False)

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = FullTokenizer(vocab_file, do_lower_case)


print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize("don't be so judgmental")))


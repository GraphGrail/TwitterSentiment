import pickle

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re

def review_to_words(review_text):
    words = review_text.lower().split()                             
    meaningful_words = list(map(lambda x: x if x[0] not in ['@', '#'] else x[1:], words))
    return(" ".join( meaningful_words ))

class SentimentAnalizer:
    def __init__(self):
        max_fatures = 2000
        embed_dim = 128
        lstm_out = 196

        self._model = Sequential()
        self._model.add(Embedding(max_fatures, embed_dim))
        self._model.add(SpatialDropout1D(0.4))
        self._model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
        self._model.add(Dense(2,activation='softmax'))
        self._model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
        self._model.load_weights("model2.h5")
        
        with open('tokenizer.pickle', 'rb') as handle:
            self._tokenizer = pickle.load(handle)
        
        #self._tokenizer = Tokenizer(num_words=max_fatures, split=' ')
        
    def process(self, doc):
        text = review_to_words(doc.lower())
        twt = [text]
        twt = self._tokenizer.texts_to_sequences(twt)
        twt = pad_sequences(twt, maxlen=40, dtype='int32', value=0)
        sentiment = self._model.predict(twt)[0]
        if(sentiment[0] > sentiment[1]):
            print(-sentiment[0])
            mark = -sentiment[0]
        else:
            print(sentiment[1])
            mark = sentiment[1]
        return mark
    
    # Fields:
    
    _model = None
    _tokenizer = None
    
    
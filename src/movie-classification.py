#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import model_selection
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score


from nltk.corpus import stopwords

import lightgbm as lgb
import os


POS_DIR = "../data/text_classification/aclImdb/train/pos/"
NEG_DIR = "../data/text_classification/aclImdb/train/neg/"


def load_data(data_dir, review_type):
    uid, rating, text = [], [], []
    data = pd.DataFrame()
    for i in tqdm(os.listdir(data_dir)):
        uid.append(i.split('_')[0])
        rating.append(i.split('_')[1].split('.')[0])
        text.append(open(data_dir + i).read())

    data['uid'] = uid
    data['rating'] = rating
    data['text'] = text
    data['review'] = review_type

    return data

pos_df = load_data(POS_DIR, 'positive')
neg_df = load_data(NEG_DIR, 'negative')

frames = [pos_df, neg_df]
train_df = pd.concat(frames)

train_df = shuffle(train_df)

count_vec =  CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vec.fit(train_df['text'])

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(train_df['text'], train_df['review'])

label_encoder = preprocessing.LabelEncoder()
train_y = label_encoder.fit_transform(train_y)
valid_y = label_encoder.fit_transform(valid_y)

x_train_count = count_vec.transform(train_x)
x_valid_count = count_vec.transform(valid_x)


def convert_to_bol(pred, threshold=0.5):
    for i in range(0, len(pred)):
        if pred[i] >= threshold:
            pred[i] = 1
        else:
            pred[i] = 0

    return pred


def train_model(x, y, num_rounds):
    params = {}
    params['learning_rate'] = 0.03
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'binary'
    params['metric'] = 'binary_logloss'
    params['sub_feature'] = 0.5
    params['num_leaves'] = 21
    params['min_data'] = 280
    params['max_depth'] = 21
    params['ntrees'] = 200

    x_train = x.astype('float64')
    y_train = y.astype('float64')


    d_train = lgb.Dataset(x_train, label=y_train)
    watchlist = [d_train]
    clf = lgb.train(params, d_train, num_rounds, watchlist, verbose_eval=500)

    return clf


def evaluate_model(x, y, model, threshold=0.5):
    x_validate = x.astype('float64')
    y_validate = y.astype('float64')

    output = model.predict(x_validate)
    output = convert_to_bol(output, threshold)

    print("Accuracy          : {:}".format(accuracy_score(output, y_validate)))
    print("cohen_kappa_score : {:}".format(cohen_kappa_score(output, y_validate)))
    print("F1-Score          : {:}".format(f1_score(output, y_validate)))


model = train_model(x=x_train_count, y=train_y_lgb)

evaluate_model(x=x_valid_count, y=valid_y, model=model)


def rate_movie(string, model=model, vec=count_vec):
    comment = [string]
    vector = vec.transform(comment).astype('float64')
    out = model.predict(vector)
    if out <= 0.5:
        rat = "NEGATIVE"
    else:
        rat = 'POSITIVE'

    print("Score for the Movie  : {:2} % \nRating for the Movie : {}".format(round(out[0]*100, 2), rat))

rate_movie("I'm not sure what accomplished director/producer/cinematographer Joshua Caldwell was thinking taking on this project. \
            his film has got to be the epitome of terrible writing and should be a classroom example of 'what not to do' when writing \
            a screenplay. Why would Joshua take on (clearly) amateur writer Adam Gaines script is beyond me. Even his good directing and \
            excellent cinematography could not save this disaster. Aside from the super obvious plot holes and very poor story overall,\
            the dragged-out unnecessary dialogue made this film unbearable and extremely boring. The way too long 1h 39min film length felt\
            like 4 hours and I  found myself saying get on with it already, who cares! when the two leads would just ramble on about nothing \
            relevant. This movie may have been interesting if it was a 30 min short film which oddly enough is the only minimal writing experience\
            Adam Gaines has")


rate_movie("I thought that this movie was incredible. I absolutely loved it, even though my brothers didn't that much. \
            The special effects were outstanding, and this movie is about my favorite sport; golf. The only thing that \
            was disappointing about this amazing movie is that it is hard to watch two times or more in a row. This movie \
            just absolutely tops everything else I have ever seen. It was everything I would expect out of a movie. I just loved it.\
            Also, it was pretty kid-friendly. This movie helped me realize that when you put your mind to it, anything is possible.\
            I would give it a pure 10/10! It was better than The Legend of Baggar Vants, and the two Pirates of the Caribbean movies\
            combined. Absolutely amazing. Loved it.")


# Using TF-IDF

tf_idf_vec = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)

tf_idf_vec.fit(train_df['text'])

x_train_tfidf = tf_idf_vec.transform(train_x)
x_valid_tfidf = tf_idf_vec.transform(valid_x)

model_tfidf = train_model(x=x_train_tfidf, y=train_y, num_rounds=2500)

evaluate_model(x=x_valid_tfidf, y=valid_y, model=model_tfidf, threshold=0.5)

rate_movie("I'm not sure what accomplished director/producer/cinematographer Joshua Caldwell was thinking taking on this project. \
            his film has got to be the epitome of terrible writing and should be a classroom example of 'what not to do' when writing \
            a screenplay. Why would Joshua take on (clearly) amateur writer Adam Gaines script is beyond me. Even his good directing and \
            excellent cinematography could not save this disaster. Aside from the super obvious plot holes and very poor story overall,\
            the dragged-out unnecessary dialogue made this film unbearable and extremely boring. The way too long 1h 39min film length felt\
            like 4 hours and I  found myself saying get on with it already, who cares! when the two leads would just ramble on about nothing \
            relevant. This movie may have been interesting if it was a 30 min short film which oddly enough is the only minimal writing experience\
            Adam Gaines has",
            model=model_tfidf, 
            vec=tf_idf_vec)


rate_movie("I thought that this movie was incredible. I absolutely loved it, even though my brothers didn't that much. \
            The special effects were outstanding, and this movie is about my favorite sport; golf. The only thing that \
            was disappointing about this amazing movie is that it is hard to watch two times or more in a row. This movie \
            just absolutely tops everything else I have ever seen. It was everything I would expect out of a movie. I just loved it.\
            Also, it was pretty kid-friendly. This movie helped me realize that when you put your mind to it, anything is possible.\
            I would give it a pure 10/10! It was better than The Legend of Baggar Vants, and the two Pirates of the Caribbean movies\
            combined. Absolutely amazing. Loved it.",
            model=model_tfidf, 
            vec=tf_idf_vec)





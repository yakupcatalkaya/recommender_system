# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:15:22 2022

@author: yakupcatalkaya
"""

import os
import sys
import getopt
import requests
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import keras
from keras.models import Model
from keras.layers import Dropout, Flatten, Input, Embedding, Dot, Dense
from keras.optimizers import Adam


def extact_zip(directory):
    for item in os.listdir():
        if ".zip" in item:
            with zipfile.ZipFile(directory + "/" + item, 'r') as zip_ref:
                try:
                    zip_ref.extractall(directory)
                except Exception as e:
                    print(e)


def util_matrix(matrix):
    matrix.userId = matrix.userId.astype('category').cat.codes.values
    matrix.movieId = matrix.movieId.astype('category').cat.codes.values
    matrix['userId'].value_counts(ascending=True)
    util_matrix = pd.pivot_table(data=matrix, values='rating', index='userId',
                                 columns='movieId', fill_value=0)
    return util_matrix


def plotter(model):
    plt.plot(model.history['loss'], 'g')
    plt.plot(model.history['val_loss'], 'b')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid(True)
    plt.show()


def download_file(url):
    local_filename = url.split('/')[-1]
    with requests.get(url, stream=True) as r:
        with open(local_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    
    
argument_list  = sys.argv[1:]
options = "tpued:"
long_options = ["train", "path", "userId", "extract", "download"]
train_model = "True"
current_directory = os.getcwd()
extract = "True"
user_index = 123
download = "True"

try:
    arguments, values = getopt.getopt(argument_list, options, long_options)
    for currentArgument, currentValue in arguments:
        if currentArgument in ("-t", "--train"):
            train_model = currentValue
     
        elif currentArgument in ("-p", "--path"):
            current_directory = currentValue
            
        elif currentArgument in ("-u", "--userId"):
            user_index = currentValue
        
        elif currentArgument in ("-e", "--extract"):
            extract = currentValue
            
        elif currentArgument in ("-d", "--download"):
            download = currentValue
            
except Exception as e:
    print(e)
    
if download=="True":
    download_file("https://files.grouplens.org/datasets/movielens/ml-25m.zip")

if extract=="True":                     
    extact_zip(current_directory)           

os.chdir(current_directory + "/ml-25m")

ratings = pd.read_csv("ratings.csv", delimiter=(","))[:1000070]
ratings = ratings.drop(columns="timestamp")
movie_names = pd.read_csv("movies.csv", delimiter=(","))

util_matrix_ratings = util_matrix(ratings)

os.chdir(current_directory)

users = ratings.userId.unique()
movies = ratings.movieId.unique()

userid2idx = {o:i for i,o in enumerate(users)}
movieid2idx = {o:i for i,o in enumerate(movies)}

ratings['userId'] = ratings['userId'].apply(lambda x: userid2idx[x])
ratings['movieId'] = ratings['movieId'].apply(lambda x: movieid2idx[x])

split = np.random.rand(len(ratings)) < 0.8
train = ratings[split]
valid = ratings[~split]

n_latent_factors = 50
n_movies=len(ratings['movieId'].unique())
n_users=len(ratings['userId'].unique())

if train_model=="True":
    user_input = Input(shape=(1,), name='user_input', dtype='int64')
    user_embedding = Embedding(n_users, n_latent_factors, name='user_embedding')(user_input)
    user_vec = Flatten(name='FlattenUsers')(user_embedding)
    user_vec = Dropout(0.40)(user_vec)
    
    movie_input = Input(shape=(1,), name='movie_input', dtype='int64')
    movie_embedding = Embedding(n_movies, n_latent_factors, name='movie_embedding')(movie_input)
    movie_vec = Flatten(name='FlattenMovies')(movie_embedding)
    movie_vec = Dropout(0.40)(movie_vec)
    
    similarity = Dot(axes=1, name='Simalarity-Dot-Product')([user_vec, movie_vec])
    model = keras.models.Model([user_input, movie_input], similarity)
    model.summary()
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse', 
                  metrics=['mse'])
    
    batch_size = 128
    epochs = 20
    
    History = model.fit([train.userId, train.movieId], train.rating, batch_size=batch_size,
                                  epochs=epochs, validation_data=([valid.userId, valid.movieId], 
                                                                  valid.rating), verbose = 1)
    
    plotter(History)
    
    nn_inp = Dense(96, activation='relu')(similarity)
    nn_inp = Dropout(0.4)(nn_inp)
    nn_inp = Dense(1, activation='relu')(nn_inp)
    nn_model = Model([user_input, movie_input], nn_inp)
    nn_model.summary()
    
    nn_model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse', 
                     metrics=['mse'])
    
    batch_size = 128
    epochs = 20
    
    History_nn = nn_model.fit([train.userId,train.movieId], train.rating, 
                           batch_size=batch_size, epochs=epochs, 
                           validation_data=([valid.userId, valid.movieId], 
                                            valid.rating), verbose = 1)
    
    plotter(History_nn)
    
    model.save("embedding_model.h5")
    nn_model.save("neuralnet_model.h5")

else:
    model = keras.models.load_model("embedding_model.h5")
    nn_model = keras.models.load_model("neuralnet_model.h5")


x_test = pd.DataFrame(np.stack(([0]*len(movies), movies),axis=1), columns=train.columns.tolist()[:2])

y_predicted = model.predict([x_test.userId, x_test.movieId])[:,0]
y_predicted_nn = nn_model.predict([x_test.userId, x_test.movieId])[:,0]

y_pred = pd.DataFrame(np.stack((y_predicted, movies),axis=1), columns=["rating","movieId"])
y_pred_nn = pd.DataFrame(np.stack((y_predicted_nn, movies),axis=1), columns=["rating","movieId"])

y_pred.sort_values(by=['rating'], inplace=True, ascending=False)
y_pred_nn.sort_values(by=['rating'], inplace=True, ascending=False)

recommend_count = 0
already_watched = ratings[ratings["userId"]==123]["movieId"].tolist()

print("__User", user_index, "Recommendations__\n")
for index, row in y_pred_nn.iterrows():
    if not recommend_count==5:
        if not int(row.movieId) in already_watched:
            recommend_count += 1
            moviee_row = list(movie_names.loc[movie_names['movieId'] == int(row.movieId)].values[0])
            moviee = " ".join(str(i) for i in moviee_row[1:])
            print(recommend_count, "-", moviee)
    else:
        break

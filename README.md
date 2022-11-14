# recommender_system  
by Yakup Çatalkaya

# __Question 1__
For dependencies, you should install all requirements by writing 
```console
pip install requirements.txt
```
I have chosen MovieLens 25M Dataset. The first 1M line which contains 6748 user and 16000 movies and corresponding ratings.
All Nan values filled with 0.0 in the sparse matrix.
"ratings.csv" contains userId, movieId, rating, timestamp columns.
The lines within this file are ordered first by userId, then, within user, by movieId.
Ratings are made on a 5-star scale, with half-star increments (0.5 stars - 5.0 stars).

I have chosen matrix factorization method to capture interactions between users and items.

For loss function, I have used Mean Squared Error (MSE) and tried to optimize it.

By using Tensorflow, I have created Input, Embedding and Flatten layers for recommendation system.
After that, I have created Simple Feed-Forward Neural Network stacked to Embedding layer.
By splitting data into train and validation set, I havel also tried to tune hyperparameter.

To get top 5 recommended movies, you should write
```console
python recommend.py userId
```

# __Question 2__

The deployable deficiency of the model is the dataset does not contain sensitive formats such as age values. 
The age value is important to filter out the adult contents for nonadult users. Therefore, the recommendation
system could cause legal problems. There could be another csv file that gives us the information about 
the adult/general content label. So we can ignore adoult content while recommending a movie to a nonadult user.

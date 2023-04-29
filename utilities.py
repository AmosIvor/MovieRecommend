import pandas as pd
import numpy as np
import tensorflow as tf
import json

# Read file ratings.csv
rating_df = pd.read_csv('./dataset/ratings.csv')

# Read file movies.csv
movie_df = pd.read_csv("./dataset/movies.csv")

# Prepare dataset
user_ids = rating_df["userId"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}

movie_ids = rating_df["movieId"].unique().tolist()
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}

rating_df["user"] = rating_df["userId"].map(user2user_encoded)
rating_df["movie"] = rating_df["movieId"].map(movie2movie_encoded)

# A new user has just signed in
from sklearn.metrics.pairwise import cosine_similarity

# One-hot encode the movie genres
genres_df = movie_df['genres'].str.get_dummies()

# Combine the ratings and genres data
movie_data = pd.concat([movie_df['movieId'], genres_df], axis=1)

rating_data = pd.merge(rating_df, movie_data, on='movieId') # original 

# Update
rating_count = rating_data.groupby('movieId')['rating'].count().reset_index()
rating_count.columns = ['movieId', 'rating_count']
rating_data = rating_data.merge(rating_count, on='movieId', how='left')

# Create a user-item matrix
user_item_matrix = rating_data.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

# Compute the movie similarities
movie_similarities = cosine_similarity(user_item_matrix.T)

# Define a function to generate movie recommendations
def get_movie_recommendations(genres, top_n=10, movie_data = movie_data):
    # Split the genres string into a list of genres
    genres_list = genres.split(',')
    
    # Filter the movie data by the specified genres
    genre_movies_df = movie_data[movie_data[genres_list].isin([1]).any(axis=1)]

    # Compute the average rating for each movie
    movie_ratings = rating_data.groupby('movieId')['rating'].mean().to_frame()

    # Handle mean ratings with two column: movieId and mean_rating
    ratings_mean = movie_ratings.reset_index()
    ratings_mean = movie_ratings.reset_index().rename(columns={'rating': 'mean_rating'})

    # Compute the number of ratings for each movie
    movie_ratings_count = rating_data.groupby('movieId')['rating'].count().to_frame()

    # Combine the movie ratings and counts data
    movie_data = pd.merge(movie_ratings, movie_ratings_count, on='movieId')
    movie_data = pd.merge(movie_data, movie_df[['movieId', 'title']], on='movieId')
    movie_data = pd.merge(movie_data, genre_movies_df, on='movieId')

    # Merge with the rating data
    movie_data = pd.merge(movie_data, rating_data[['movieId', 'rating', 'rating_count']], on='movieId')

    # Compute the weighted average rating for each movie
    movie_data['weighted_rating'] = (movie_data['rating'] * movie_data['rating_count']) / (movie_data['rating_count'] + 1000)

    # Compute the similarity between the selected genres and all movies
    genre_vector = genre_movies_df.mean().values.reshape(1, -1)
    similarity_scores = cosine_similarity(genre_vector, movie_data.iloc[:, 4:-2].values)[0]

    # Sort the movies by their similarity score and weighted rating
    movie_data['similarity_score'] = similarity_scores
    movie_data = movie_data.sort_values(['similarity_score', 'weighted_rating'], ascending=False)
    movie_data.drop_duplicates(subset='movieId', keep='first', inplace=True)

    # Select the top N recommended movies
    recommended_movies = movie_data.head(top_n)
    recommended_movies = pd.merge(recommended_movies, movie_df[['movieId', 'genres']], on='movieId')
    recommended_movies = pd.merge(recommended_movies, ratings_mean[['movieId', 'mean_rating']], on='movieId')

    # Format the mean_rating column
    recommended_movies['mean_rating'] = recommended_movies['mean_rating'].apply(lambda x: f"{x:.1f}")

    # return recommended_movies[['movieId', 'title', 'genres', 'mean_rating', 'weighted_rating', 'similarity_score']]
    # return json.loads(recommended_movies[['movieId', 'title', 'genres', 'mean_rating']].to_json(orient='records'))
    return json.dumps(recommended_movies[['movieId', 'title', 'genres', 'mean_rating']].to_dict('records'),indent=4)

if __name__ == "__main__":
    genres = 'Action,Thriller'
    movie_recommendation_json = get_movie_recommendations(genres)
    print(movie_recommendation_json)
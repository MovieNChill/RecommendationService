print('IMPORT')

from flask import Flask, request, jsonify
from surprise import Reader, Dataset, SVD
import pandas as pd

import requests
import zipfile

recommendation_init_finished = False
app = Flask(__name__)



@app.route('/init', methods=['GET'])
def init_recommandation():
    #Extraction donnée
    print('BEGIN INIT')
    url = 'https://files.grouplens.org/datasets/movielens/ml-latest.zip'
    r = requests.get(url, allow_redirects=True)
    
    open('data.zip', 'wb').write(r.content)
    print('finish download Zip')

    with zipfile.ZipFile("data.zip","r") as zip_ref:
        zip_ref.extractall(".")
    # Create the Flask app
    print('finish Extract All')
    # Presentation
    print("--------------------------------------------")
    print("    Recommendation System - MovieNChill")
    print("--------------------------------------------")

    print("----------------DATA LOADING----------------")
    # Load the dataset with data ratings
    print("[LOG] Loading ratings dataset...")
    ratings_data = pd.read_csv('ml-latest/ratings.csv')
    ratings_reader = Reader()
    ratings_data = Dataset.load_from_df(ratings_data[['userId', 'movieId', 'rating']], ratings_reader)
    print("[SUCCESS] Ratings dataset loaded!")

    # Load the dataset with movies informations
    print("[LOG] Loading movies dataset...")
    movies_data = pd.read_csv('ml-latest/movies.csv')
    print("[SUCCESS] Movies dataset loaded!")

    print("----------------TRAIN MODEL----------------")
    print("[LOG] Training the model...")
    print("[LOG] This may take a while...")
    algo = SVD()
    trainset = ratings_data.build_full_trainset()
    algo.fit(trainset)
    print("[SUCCESS] Model trained!")
    recommendation_init_finished = True
    return "Model trained"
@app.route('/', methods=['GET'])
def hello_world():
    return "Hello World!"
@app.route('/recommend', methods=['POST'])
def recommend():
    # Get the user_id and desired_genre from the request
    print("[LOG] Received request!")
    user_id = request.json['user_id']
    desired_genre = request.json['desired_genre']
    print("User ID: " + str(user_id))
    print("Desired genre: " + desired_genre)

    # Search of movies that match
    print("[LOG] Searching for movies that match..")
    genre_movies = movies_data.loc[movies_data['genres'] == desired_genre]
    genre_predictions = []
    for index, row in genre_movies.iterrows():
        prediction = algo.predict(user_id, row['movieId'])
        genre_predictions.append((row['movieId'], prediction[3]))
    genre_predictions.sort(key=lambda x: x[1], reverse=True)
    print("[SUCCESS] Movies found!")

    # Get the movie with the highest rating and return it to the Java BACKEND
    recommended_movie_id = genre_predictions[0][0]
    recommended_movie_info = movies_data.loc[movies_data['movieId'] == recommended_movie_id]
    return jsonify({'recommended_movie': recommended_movie_info.to_dict()})

if __name__ == '__main__':
  app.run()





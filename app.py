from flask import Flask, request, jsonify
from surprise import Reader, Dataset, SVD
import pandas as pd
import threading

import requests
import zipfile
import os.path

# SHARED VALUES
recommendation_init_finished = False
algo = SVD()
movies_data = None

def parallelize_functions(*functions):
    processes = []
    for function in functions:
        p = threading.Thread(target=function)
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

def init_recommandation():
    global recommendation_init_finished
    global algo
    global movies_data
    #Extraction donnée

  
    if(os.path.exists("./ml-latest/ratings.csv") == False or os.path.exists("./ml-latest/movies.csv") == False ) :
        print('BEGIN INIT')
        url = 'https://files.grouplens.org/datasets/movielens/ml-latest.zip'
        r = requests.get(url, allow_redirects=True)
        open('data.zip', 'wb').write(r.content)
        print('finish download Zip')
    
        with zipfile.ZipFile("data.zip","r") as zip_ref:
            zip_ref.extractall(".")
        # Create the Flask app
        print('finish Extract All')
    else:
        print('File already exist')

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
   
    trainset = ratings_data.build_full_trainset()
    algo.fit(trainset)
    print("[SUCCESS] Model trained!")
    recommendation_init_finished = True

       

app = Flask(__name__)
@app.route('/', methods=['GET'])
def hello_world():
    global recommendation_init_finished
    return jsonify({"title": "Welcome to the movieNChill recommendation algorithm!", "ready": recommendation_init_finished })

@app.route('/recommend', methods=['POST'])
def recommend():
    global algo
    global movies_data
    global recommendation_init_finished
    if(recommendation_init_finished == False):
        return "algo recommendation has not finished training, wait a moment"

    # Get the user_id and desired_genre from the request
    print("[LOG] Received request!")
    user_id = request.json['user_id']
    desired_genre = request.json['desired_genre']
    print("User ID: " + str(user_id))
    print("Desired genre: " + desired_genre)

    # Search of movies that matchs
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

def runAPP():
    print('RUN APP')
    app.run( use_reloader=False, host='0.0.0.0', port=8000)

if __name__ == '__main__':
     parallelize_functions(runAPP, init_recommandation)
  




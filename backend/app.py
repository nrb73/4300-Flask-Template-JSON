import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import jsonpickle
import numpy as np
import sys

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = "/Users/meer/Desktop/4300/4300-Flask-Template-JSON/backend/spotify_api/database_jsons/big_artist_dataframes.json"
json_file_path2 = "/Users/meer/Desktop/4300/4300-Flask-Template-JSON/backend/spotify_api/database_jsons/small_artist_dataframes.json"

# Assuming your JSON data is stored in a file named 'init.json'
big_songs_df = pd.read_json(json_file_path, orient = "split")

small_songs_df = pd.read_json(json_file_path2, orient="split")
small_songs_np = small_songs_df.to_numpy()

with open("/Users/meer/Desktop/4300/4300-Flask-Template-JSON/backend/spotify_api/database_jsons/big_songs_set.json", "r") as openfile: 
  big_songs_json_obj = json.load(openfile)
#retrieve set of artists in the database
big_songs_set = jsonpickle.decode(big_songs_json_obj)

# with open("/Users/meer/Desktop/4300/4300-Flask-Template-JSON/backend/spotify_api/database_jsons/small_songs_set.json", "r") as openfile: 
#   small_songs_json_obj = json.load(openfile)
#retrieve set of artists in the database
small_songs_list = small_songs_df.index.tolist()

debug = True

np.set_printoptions(threshold=sys.maxsize)

app = Flask(__name__)
CORS(app)

# Sample search using json with pandas
def json_search(query):
    if query in big_songs_set:
        song_ft = big_songs_df.loc[query]
        song_vect = song_ft.to_numpy()
        if song_vect.ndim > 1:
            song_vect = song_vect[0]
        print("song vector")
        print(song_vect)
        similarity_index = np.dot(song_vect, small_songs_np.T) / (np.linalg.norm(song_vect) * np.linalg.norm(small_songs_np))
        rankings = np.argsort(similarity_index)
        print("similarity index")
        print(np.shape(similarity_index))
        print(similarity_index)
        print("rankings:")
        print(rankings)
        res = []
        for r in rankings[:10]:
            res.append(small_songs_list[r])
        print(res)
        return json.dumps(res)
    else: 
        return json.dumps(small_songs_list[:10])

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/songs")

def song_search():
    text = request.args.get("title")
    return json_search(text)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)
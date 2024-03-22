import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import jsonpickle
import numpy as np

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

    

app = Flask(__name__)
CORS(app)

# Sample search using json with pandas
def json_search(query):
    if query in big_songs_set:
        song_ft = big_songs_df.loc[query]
        song_vect = song_ft.to_numpy()
        print(np.shape(song_vect))
        print(np.shape(small_songs_np))
        similarity_index = np.dot(song_vect, small_songs_np)
        print(similarity_index)

    return small_songs_df.to_json()

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/episodes")

def episodes_search():
    text = request.args.get("title")
    return json_search(text)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)
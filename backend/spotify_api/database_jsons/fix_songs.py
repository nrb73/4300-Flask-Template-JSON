import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd
import jsonpickle
import numpy as np
import sys

with open("../backend/spotify_api/database_jsons/big_songs_set.json", "r") as openfile: 
  big_songs_json_obj = json.load(openfile)
#retrieve set of artists in the database
big_songs_set = jsonpickle.decode(big_songs_json_obj)

big_songs_list = list(big_songs_set)

# new_songs_json_obj = json.encoder(big_songs_list)
with open("/Users/meer/Desktop/4300/4300-Flask-Template-JSON/backend/spotify_api/database_jsons/big_songs_list.json", "w") as openfile:
  json.dump(big_songs_list, openfile)
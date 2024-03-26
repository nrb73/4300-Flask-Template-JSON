import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd
import jsonpickle
import numpy as np
import sys

# new_songs_json_obj = json.encoder(big_songs_list)
with open("/Users/meer/Desktop/4300/4300-Flask-Template-JSON/backend/spotify_api/database_jsons/big_songs_list.json", "r") as openfile: 
  big_songs_list = json.load(openfile)

for i in big_songs_list:
  print(f'<option value="{i}"></option>')
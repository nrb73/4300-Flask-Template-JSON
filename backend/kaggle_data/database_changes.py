import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd
import jsonpickle
import numpy as np
import sys

artist_df = pd.read_csv('/Users/meer/Desktop/4300/4300-Flask-Template-JSON/backend/kaggle_data/new_data.csv')

artist_set = artist_df.loc[:, "artist"]
lst = []
for i in artist_set:
  lst.append(str(i))
print(lst)
print("50 Cent" in lst)
from dotenv import load_dotenv
import os
import base64
from requests import post, get
import json
import numpy as np
import pandas as pd

load_dotenv()

feature_dict = {
  "acousticness" : 0,
  "danceability" : 1,
  "energy" : 2,
  "instrumentalness" : 3,
  "key" : 4,
  "liveness" : 5,
  "loudness" : 6,
  "mode" : 7,
  "speechiness" : 8,
  "tempo" : 9,
  "time_signature" : 10,
  "valence" : 11 
}

collumns = ["acousticness","danceability","energy","instrumentalness",
"key","liveness","loudness","mode","speechiness","tempo","time_signature",
"valence"]

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

def get_token():
  #Gets new token from Spotify
  auth_bytes = (client_id + ":" + client_secret).encode("utf-8")
  auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")

  url = "https://accounts.spotify.com/api/token"
  headers = {
    "Authorization" : "Basic " + auth_base64,
    "Content-Type" : "application/x-www-form-urlencoded"
  }

  data = {"grant_type": "client_credentials"}
  result = post(url, headers = headers, data=data)
  json_result = json.loads(result.content)
  token = json_result["access_token"]
  return token


def get_header(token):
  #returns header based on token
  return{"Authorization" : "Bearer " + token}


def search_artist(token, artist_name):
#get artist id using artist name, returns 0 if artist not found
  url = "https://api.spotify.com/v1/search"
  headers = get_header(token)
  query = f"?q={artist_name}&type=artist&limit=1"
  query_url = url + query 
  result = get(query_url, headers=headers)
  json_result = json.loads(result.content)["artists"]["items"]
  if len(json_result) == 0:
    return 0 
  result = json_result[0]
  artist_id = result["id"]
  return artist_id


def get_artist_top_songs(token, artist_id):
#get artist's top 10 songs
  url = f"https://api.spotify.com/v1/artists/{artist_id}/top-tracks"
  headers = get_header(token)
  result = get(url, headers=headers)
  json_result = json.loads(result.content)["tracks"]
  return json_result


def matrix_function(json):
  #creates matrix of features of 10 songs by one artist
  song_id_to_matrix_index  = {}
  song_by_feature_mat = np.zeros((len(json), 12))
  for i, song in enumerate(json):
    song_id_to_matrix_index[song["id"]] = i
    for f in feature_dict.keys():
      song_by_feature_mat[i][feature_dict[f]] = song[f]
  return song_by_feature_mat, song_id_to_matrix_index

def make_artist_matrix(token, artist_ids, artist_to_song_dict={}):
  song_to_artist_dict = {}
  song_id_to_name = {}
  for a in artist_ids.keys():
    id = artist_ids[a]
    #get top 10 songs by artist
    songs = get_artist_top_songs(token, id)
    song_ids = ""
    for i, song in enumerate(songs):
      #populate song to artist dict
      song_to_artist_dict[song["name"]] = a
      song_id_to_name[song["id"]] =  song["name"]
      song_ids += f"{song['id']},"

    song_ids = song_ids[:-1]
    url = f"https://api.spotify.com/v1/audio-features?ids={song_ids}"
    headers = get_header(token)
    result = get(url, headers=headers)
    json_result = json.loads(result.content)["audio_features"]
    #send audio features of 10 tracks to get song by feature matrix
    song_by_feature_mat, song_id_to_matrix_index = matrix_function(json_result)
    #returns a matrix for songs by features and a song id to matrix index
    artist_to_song_dict [a] = (song_by_feature_mat, song_id_to_matrix_index)
    
  return artist_to_song_dict , song_to_artist_dict, song_id_to_name


def create_top_songs_matrix_by(token, artist_list):
  #returns a tuple of dictionaries
  #artist list: list of artist names
  artist_ids = {}
  #get artist id's from names list
  for artist in artist_list:
    artist_ids[artist] = search_artist(token, artist)
  #send artist ids to matrix maker, returns a tuple with a artist_to_song_dict and a song to artist dictionary. The artist to song dict has a tuple for every artist. The tuple has a song by feature vector at index 0 for that artist's top 10 songs and song_id_to_matrix_index
  return  make_artist_matrix(token, artist_ids)

def print_artist_track_matrix(artist_to_song_dict, song_to_artist_d, song_id_to_name, a):
  song_by_feature_mat, song_id_to_matrix_index = artist_to_song_dict[a]
  r, c = song_by_feature_mat.shape
  row_labels = ["unknown"] * r
  for k in song_id_to_matrix_index.keys():
    row_labels[song_id_to_matrix_index[k]] = song_id_to_name[k]
  df = pd.DataFrame(song_by_feature_mat, columns = collumns, index = row_labels)
  # df = pd.DataFrame(song_by_feature_mat)
  print(df)

curr_token = get_token()
# token = "current token"
artist_list = ["baalti", "pompie", "drake"]
artist_to_song_dict , song_to_artist_dict, song_id_to_name = create_top_songs_matrix_by(curr_token, artist_list)
for a in artist_list:
  print_artist_track_matrix(artist_to_song_dict, song_to_artist_dict,song_id_to_name, a)










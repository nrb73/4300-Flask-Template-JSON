from dotenv import load_dotenv
import os
import base64
from requests import post, get
import json
import numpy as np
import pandas as pd
import jsonpickle
from json import JSONEncoder
from pathlib import Path

load_dotenv()

#GLOBAL VARIABLES

# with open("backend/spotify_api/database_jsons/small_artist_set.json", "r") as openfile: 
#   artist_json_obj = json.load(openfile)
#retrieve set of artists in the database

# artist_set = jsonpickle.decode(artist_json_obj)
artist_set = set()


# with open("backend/spotify_api/database_jsons/small_songs_set.json", "r") as openfile: 
#   song_json_obj = json.load(openfile)
#retrieve set of artists in the database

# songs_set = jsonpickle.decode(song_json_obj)
songs_set = set()


#retrieve dataframes
# with open("backend/spotify_api/database_jsons/artist_dataframes.json", "r") as openfile: 
#   dataframes = [pd.read_json("/Users/meer/Desktop/4300/4300-Flask-Template-JSON/backend/spotify_api/database_jsons/artist_dataframes.json", orient = "split")]

dataframes = []

#Feature to index for song by feature matrix. Please update this as features 
#increase. This vector contains numbers only (to be used for similarity analysis)
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

collumns = feature_dict.keys()

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
    if song != None and song["duration_ms"] != 0:
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
  return make_artist_matrix(token, artist_ids)

def make_artist_track_matrix(artist_to_song_dict, song_to_artist_d, song_id_to_name, a):
  song_by_feature_mat, song_id_to_matrix_index = artist_to_song_dict[a]
  r, c = song_by_feature_mat.shape
  row_labels = ["unknown"] * r
  for k in song_id_to_matrix_index.keys():
    row_labels[song_id_to_matrix_index[k]] = song_id_to_name[k]
  df = pd.DataFrame(song_by_feature_mat, columns = collumns, index = row_labels)
  return df

def update_artist_songs(res):
  #needs to be implemented
  for i in res.index:
    if i in songs_set:
      res.drop(i)
    else:
      songs_set.add(i)
  return res
  



#################################################################################################
# Main Function here onwards:
  
curr_token = get_token()
# token = "current token"
#add artists in the list below
# artist_list = ['21 Savage',
    # 'Adele',
    # 'Anuel AA',
    # 'Arctic Monkeys',
    # 'Ariana Grande',
    # 'Arijit Singh',
    # 'Ayra Starr',
    # 'BTS',
    # 'Bad Bunny',
    # 'Benson Boone',
    # 'Beyoncé',
    # 'Billie Eilish',
    # 'Bizarrap',
    # 'Bruno Mars',
    # 'Burna Boy',
    # 'CYRIL',
    # 'Carin Leon',
    # 'Chris Brown',
    # 'Coldplay',
    # 'Creepy Nuts',
    # 'Cris Mj',
    # 'Dadju',
    # 'David Guetta',
    # 'Diljit Dosanjh',
    # 'Disturbed',
    # 'Djo',
    # 'Doja Cat',
    # 'Drake',
    # 'Dua Lipa',
    # 'Ed Sheeran',
    # 'Emilia',
    # 'Eminem',
    # 'Feid',
    # 'Frank Ocean',
    # 'Fuerza Regida',
    # 'Future',
    # 'Gazo',
    # 'Grupo Frontera',
    # 'Harry Styles',
    # 'Hozier',
    # 'JUL',
    # 'Jack Harlow',
    # 'Jere Klein',
    # 'Jimin',
    # 'Jung Kook',
    # 'Junior H',
    # 'Justin Bieber',
    # 'Justin Timberlake',
    # 'KAROL G',
    # 'Kacey Musgraves',
    # 'Kanye West',
    # 'LE SSERAFIM',
    # 'LINKIN PARK',
    # 'Lady Gaga',
    # 'Lana Del Rey',
    # 'Luke Combs',
    # 'Maluma',
    # 'Manuel Turizo',
    # 'Metro Boomin',
    # 'Miley Cyrus',
    # 'Milo j',
    # 'Mora',
    # 'Morgan Wallen',
    # 'Mrs. Green Apple',
    # 'Myke Towers',
    # 'Natanael Cano',
    # 'Natasha Bedingfield',
    # 'Ninho',
    # 'Noah Kahan',
    # 'Ofenbach',
    # 'Olivia Rodrigo',
    # 'One Direction',
    # 'Peso Pluma',
    # 'Playboi Carti',
    # 'Pritam',
    # 'Quevedo',
    # 'Rauw Alejandro',
    # 'Rihanna',
    # 'SZA',
    # 'Shakira',
    # 'Tate McRae',
    # 'Tayc',
    # 'Taylor Swift',
    # 'Teddy Swims',
    # 'The Weeknd',
    # 'Tiakola',
    # 'Tiésto',
    # 'Tony Effe',
    # 'Travis Scott',
    # 'Ty Dolla $ign',
    # 'Tyla',
    # 'V',
    # 'Werenoi',
    # 'Xavi',
    # 'YG Marley',
    # 'YOASOBI',
    # 'Zach Bryan',
    # 'floyymenor',
    # '¥$' ]
artist_list = [ "Anna B Savage",
    "Automatic",
    "Bambara",
    "Barrie",
    "Bbymutha",
    "Bendik Giske",
    "Billy Woods",
    "Black Country, New Road",
    "Brutus",
    "Cassandra Jenkins",
    "Chastity Belt",
    "Chloe Moriondo",
    "Chris Cohen",
    "Christian Lee Hutson",
    "Circuit des Yeux",
    "Claud",
    "Cleo Sol",
    "Clipping.",
    "Crack Cloud",
    "Dehd",
    "Disq",
    "Dorian Electra",
    "Dry Cleaning",
    "Duval Timothy",
    "Ela Minus",
    "Empath",
    "Fenne Lily",
    "Fieh",
    "Field Medic",
    "Gabriels",
    "Ganser",
    "Girlpool",
    "Gleemer",
    "Goat Girl",
    "Hachiku",
    "Hand Habits",
    "Hannah Diamond",
    "Hatchie",
    "Hater",
    "Helena Deland",
    "Hilary Woods",
    "Horse Jumper of Love",
    "Ian Sweet",
    "Illuminati Hotties",
    "Jaye Jayle",
    "Jeff Rosenstock",
    "Jesca Hoop",
    "Jess Williamson",
    "Jockstrap",
    "Kaina",
    "Katie Dey",
    "Katy Kirby",
    "Keeley Forsyth",
    "Kelly Moran",
    "Kero Kero Bonito",
    "Khruangbin",
    "Kikagaku Moyo",
    "Klein",
    "Lala Lala",
    "Land of Talk",
    "Lingua Ignota",
    "Lisa/Liza",
    "Little Simz",
    "Lomelda",
    "Lonnie Holley",
    "Lucy Dacus",
    "Mabe Fratti",
    "Madeline Kenney",
    "Mapache",
    "Maria Somerville",
    "Marie Davidson",
    "Marlowe",
    "Mary Lattimore",
    "Maston",
    "Matana Roberts",
    "Melkbelly",
    "Mereba",
    "Methyl Ethel",
    "Mia Gargaret",
    "Michael Nau",
    "Molly Burch",
    "Molly Sarlé",
    "Moor Jewelry",
    "MorMor",
    "Nadia Reid",
    "Natalie Prass",
    "Nation of Language",
    "Negative Gemini",
    "Nick Hakim",
    "Nicolas Jaar",
    "Nite Jewel",
    "No Joy",
    "Norman Westberg",
    "Noveller",
    "Okay Kaya",
    "Omni",
    "Oso Oso",
    "Overcoats",
    "Palm",
    "Pan Amsterdam",
    "Pendant",
    "Perfume Genius",
    "Pile",
    "Pinegrove",
    "Porridge Radio",
    "Posse",
    "Protomartyr",
    "Puma Blue",
    "Pure X",
    "Quelle Chris",
    "R.A.P. Ferreira",
    "Ratboys",
    "Remember Sports",
    "Renata Zeiguer",
    "Ric Wilson",
    "Richard Dawson",
    "Rosie Tucker",
    "Sammus",
    "Sarah Mary Chadwick",
    "Sassy 009",
    "Shamir",
    "Shannon Lay",
    "Sheer Mag",
    "Shygirl",
    "Sleaford Mods",
    "Snapped Ankles",
    "Soccer Mommy",
    "Soccer96",
    "Solange",
    "Spellling",
    "Squid",
    "Stella Donnelly",
    "Steve Lacy",
    "Sudan Archives",
    "Sufjan Stevens",
    "Summer Walker",
    "Sunflower Bean",
    "Surf Curse",
    "Sylvan Esso",
    "Tasha",
    "Tennis",
    "The Beths",
    "The Comet Is Coming",
    "The Garden",
    "The Japanese House",
    "The Weather Station",
    "Thou",
    "Tirzah",
    "Tomberlin",
    "TOPS",
    "Toro y Moi",
    "Trace Mountains",
    "Vagabon",
    "Vessel",
    "Waxahatchee",
    "Weatherday",
    "Wednesday",
    "Weyes Blood",
    "White Reaper",
    "Wiki",
    "William Tyler",
    "Wye Oak",
    "Xenia Rubinos",
    "Yaeji",
    "Yves Tumor",
    "Zola Jesus"]
artist_to_song_dict , song_to_artist_dict, song_id_to_name = create_top_songs_matrix_by(curr_token, artist_list)

for a in artist_list:
  #returns dataframe of artist a's song by feature matrix
  res = make_artist_track_matrix(artist_to_song_dict, song_to_artist_dict,song_id_to_name, a)
  update_artist_songs(res)
  artist_set.add(a)
  dataframes.append(res)
  # print(res)

#return set of artists in the database
new_artist_json_obj = jsonpickle.encode(artist_set)
with open("/Users/meer/Desktop/4300/4300-Flask-Template-JSON/backend/spotify_api/database_jsons/small_artist_list.json", "w") as openfile:
  json.dump(new_artist_json_obj, openfile)

#return songs set to database
new_songs_json_obj = jsonpickle.encode(songs_set)
with open("/Users/meer/Desktop/4300/4300-Flask-Template-JSON/backend/spotify_api/database_jsons/small_songs_set.json", "w") as openfile:
  json.dump(new_songs_json_obj, openfile)

#return list of dataframes
p = Path("/Users/meer/Desktop/4300/4300-Flask-Template-JSON/backend/spotify_api/database_jsons/small_artist_dataframes.json")
fin_dataframe = pd.concat(dataframes, join='outer', axis=0)
print(fin_dataframe)
fin_dataframe.to_json(path_or_buf=p, orient='split')











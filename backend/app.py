import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import numpy as np
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MultiLabelBinarizer
import html_to_json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script

# json_file_path = "../backend/spotify_api/database_jsons/big_artist_dataframes.json"
# json_file_path2 = "../backend/spotify_api/database_jsons/small_artist_dataframes.json"

# # Assuming your JSON data is stored in a file named 'init.json'
# big_songs_df = pd.read_json(json_file_path, orient = "split")

# small_songs_df = pd.read_json(json_file_path2, orient="split")
# small_songs_np = small_songs_df.to_numpy()

# with open("../backend/spotify_api/database_jsons/big_songs_list.json", "r") as openfile: 
#   big_songs_set = json.load(openfile)

# with open("/Users/meer/Desktop/4300/4300-Flask-Template-JSON/backend/spotify_api/database_jsons/small_songs_set.json", "r") as openfile: 
#   small_songs_json_obj = json.load(openfile)
# #retrieve set of artists in the database
# small_songs_list = small_songs_df.index.tolist()

artist_df = pd.read_csv(current_directory + '/kaggle_data/final_artist_dataframe.csv')

artist_df = artist_df.dropna()

with open(current_directory + '/spotify_api/database_jsons/artist_images_dataframes.json', "r") as openfile: 
  artist_images_json = json.load(openfile)


# defining weights for features
# weight_titles = 0.3 #since song titles did not give accurate results, weighted less
# weight_lyrics = 1.5 #weighted more since it was more accurate
# weight_views = 1.0 
# weight_tags = 1.0
  
weight_titles = 0.3 #since song titles did not give accurate results, weighted less
weight_lyrics = 1.5 #weighted more since it was more accurate
weight_views = 1.0 
weight_tags = 1.0
weight_score = 0.5
weight_reviews = 1.4 

features_combined_weighted = None

query_artist = None

artist_to_image = {}
for pair in artist_images_json["data"]:
    artist_name = pair[0]
    artist_img = pair[1]
    artist_to_image[artist_name] = artist_img

lst = artist_df.loc[:, "artist"]
artist_set = []
for i in lst:
  artist_set.append(str(i))


debug = False

np.set_printoptions(threshold=sys.maxsize)

app = Flask(__name__)
CORS(app)

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

def lyric_vectorization (artist_df, query_artist):
    tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1,2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(artist_df['concatenated_lyrics'])

    # finding cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    cosine_sim_df = pd.DataFrame(cosine_sim, index=artist_df['artist'], columns=artist_df['artist'])

    # get similarity scores for Eminem
    similar_artists = cosine_sim_df[f'{query_artist}'].sort_values(ascending=False)
    return (similar_artists[1:], tfidf_matrix)

def song_name_vectorization (artist_df, query_artist):
    # FINDING ARTIST SIMILARITY USING ONLY SONG TITLES
    # print("start func snv")
    #finding tfidf scores
    tfidf_vectorizer_title = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1,2))
    # print(artist_df['song_titles'].index(np.nan))
    # print(artist_df['song_titles'])
    tfidf_matrix_title = tfidf_vectorizer_title.fit_transform(artist_df['song_titles'])

    #finding cosine similarity
    cosine_sim_title = cosine_similarity(tfidf_matrix_title, tfidf_matrix_title)
    cosine_sim_df_title = pd.DataFrame(cosine_sim_title, index=artist_df['artist'], columns=artist_df['artist'])

    # get similarity scores for Eminem
    similar_artists_title = cosine_sim_df_title[f'{query_artist}'].sort_values(ascending=False)
    print("end snv")
    return (similar_artists_title[1:], tfidf_matrix_title)

def review_vectorization (artist_df, query_artist):
    stop_words = set(stopwords.words('english'))
    extended_stopwords = set(stopwords.words('english'))
    additional_stopwords = {
    'album', 'albums', 'song', 'songs', 'music', 'sound', 'track', 'tracks', 'record', 'records', 'single', 'singles',
    'artist', 'artists', 'band', 'bands', 'release', 'releases', 'released', 'make', 'makes', 'made', 'say', 'says',
    'put', 'puts', 'get', 'gets', 'got', 'go', 'goes', 'going', 'seem', 'seems', 'seemed', 'include', 'includes',
    'included', 'featuring', 'feature', 'features', 'featured', 'feel', 'feels', 'felt', 'keep', 'keeps', 'kept',
    'great', 'good', 'big', 'large', 'small', 'new', 'old', 'young', 'real', 'better', 'best', 'bad', 'worst',
    'major', 'minor', 'own', 'same', 'different', 'high', 'low', 'long', 'short', 'first', 'last', 'next', 'previous',
    'early', 'late', 'modern', 'year', 'years', 'time', 'times', 'day', 'days', 'week', 'weeks',
    'month', 'months', 'like', 'just', 'also', 'well', 'still', 'back', 'even', 'way', 'much', 'ever', 'never',
    'every', 'around', 'another', 'many', 'few', 'lots', 'lot', 'part', 'one', 'two', 'three', 'four', 'five',
    'six', 'seven', 'eight', 'nine', 'ten', 'several', 'various', 'whether', 'however', 'though', 'although',
    'fashioned', 'im'
    }

    extended_stopwords.update(additional_stopwords)

    tfidf_vectorizer_reviews = TfidfVectorizer(max_features=7500, stop_words=stop_words, ngram_range=(1,2))
    tfidf_matrix_reviews = tfidf_vectorizer_reviews.fit_transform(artist_df['Reviews'])

    cosine_sim_reviews = cosine_similarity(tfidf_matrix_reviews, tfidf_matrix_reviews)
    cosine_sim_df_reviews = pd.DataFrame(cosine_sim_reviews, index=artist_df['artist'], columns=artist_df['artist'])

    # get similarity scores for Eminem
    similar_artists_review = cosine_sim_df_reviews[f'{query_artist}'].sort_values(ascending=False)

    return (similar_artists_review[1:], tfidf_matrix_reviews)

def composite_vector (artist_df, tfidf_matrix_lyrics, tfidf_matrix_title, tfidf_matrix_reviews):
    global features_combined_weighted
    scaler = MinMaxScaler()
    view_counts_normalized = scaler.fit_transform(artist_df[['average_views']])
    tag_binarizer = MultiLabelBinarizer()
    tags_transformed = tag_binarizer.fit_transform(artist_df['tags'])
    review_scores_normalized = scaler.fit_transform(artist_df[['average_score']])

    # defining weights for features - can change this
    weight_titles = 0.3 #since song titles did not give accurate results, weighted less
    weight_lyrics = 1.5 #weighted more since it was more accurate
    weight_views = 1.0 
    weight_tags = 1.0
    weight_score = 0.5
    weight_reviews = 1.4 #captures intricate similarities between artists, so weighted higher

# applying weights to features
    weighted_titles = tfidf_matrix_title.toarray() * weight_titles
    weighted_lyrics = tfidf_matrix_lyrics.toarray() * weight_lyrics
    weighted_views = view_counts_normalized * weight_views
    weighted_tags = tags_transformed * weight_tags
    weighted_score = review_scores_normalized * weight_score
    weighted_reviews = tfidf_matrix_reviews.toarray() * weight_reviews

#MAKING THE COMPOSITE VECTOR WITH REVIEWS

    features_combined_weighted_review = np.hstack((weighted_titles, weighted_lyrics, weighted_views, weighted_tags, weighted_score, weighted_reviews))

#COSINE SIMILARITY FOR COMPOSITE VECTOR

    cosine_sim_weighted_review = cosine_similarity(features_combined_weighted_review, features_combined_weighted_review)
    cosine_sim_df_weighted_review = pd.DataFrame(cosine_sim_weighted_review, index=artist_df['artist'], columns=artist_df['artist'])

    return cosine_sim_df_weighted_review


def mean_vector(indices, feature_matrix):
    return np.mean(feature_matrix[indices, :], axis=0)

def update_vector(relevant_artists=[], irrelevant_artists=[]):
    # print(artist_df['artist'])
    # print("between")
    # print(f'{query_artist}')
    # print(artist_df.index[artist_df['artist'] == query_artist])
    query_index = artist_df.index[artist_df['artist'] == query_artist].tolist()[0]
    relevant_indices = artist_df.index[artist_df['artist'].isin(relevant_artists)].tolist()
    irrelevant_indices = artist_df.index[artist_df['artist'].isin(irrelevant_artists)].tolist()



    # finding relevant and irrelevant vectors
    original_query_vector = features_combined_weighted[query_index, :]
    if relevant_indices != []:
        mean_relevant = mean_vector(relevant_indices, features_combined_weighted)
    else:
        mean_relevant = 0
    if irrelevant_indices != []:
        mean_irrelevant = mean_vector(irrelevant_indices, features_combined_weighted)
    else:
        mean_irrelevant = 0
    # rocchio parameters
    alpha, beta, gamma = 1.0, 0.75, 0.25

    # update query vector after performing rocchio
    updated_query_vector = (alpha * original_query_vector +
                        beta * mean_relevant -
                        gamma * mean_irrelevant)

    # updated recommendations
    updated_similarities = cosine_similarity(updated_query_vector.reshape(1, -1), features_combined_weighted)
    updated_similarities_df = pd.DataFrame(updated_similarities, columns=artist_df['artist'], index=['Similarity']).T
    updated_recommendations = updated_similarities_df.sort_values(by='Similarity', ascending=False)

    fin_list = []
    temp_count = 0
    i = 0
    while temp_count < 10:
        a_name = updated_recommendations.index[i]
        if a_name not in irrelevant_artists:
            fin_list.append(a_name)
            count += 1
        else:
            pass
        i += 1
    # for i in updated_recommendations.index[1:11]:
    #     fin_list.append(i)
    return fin_list

# Normalizes certain features in features_vec
def normalize_feature_vec(features_vec):
    key_idx =feature_dict["key"]
    features_vec[key_idx] = (features_vec[key_idx] + 1) / 12
    loudness_idx =feature_dict["loudness"]
    features_vec[loudness_idx] = features_vec[loudness_idx] / (-60)
    tempo_idx =feature_dict["tempo"]
    features_vec[tempo_idx] = (features_vec[tempo_idx] - 50) / 150
    time_sig_idx =feature_dict["time_signature"]
    features_vec[time_sig_idx] = (features_vec[time_sig_idx] - 3) / 4

    return features_vec

# Normalizes cetrain features in features_mat
def normalize_feature_mat(features_mat):
    for row in range(len(features_mat)):
        features_mat[row] = normalize_feature_vec(features_mat[row])
    
    return features_mat
    

# Sample search using json with pandas
def json_search(query):
    global query_artist
    query_artist = query
    if query in artist_set:
        # print(query)
        _, tfidf_matrix = lyric_vectorization(artist_df, query)
        _, tfidf_matrix_title = song_name_vectorization(artist_df, query)
        _, tfidf_matrix_reviews = review_vectorization(artist_df, query)
        cosine_sim_df_weighted = composite_vector (artist_df, tfidf_matrix, tfidf_matrix_title, tfidf_matrix_reviews)
        similar_artists_composite = cosine_sim_df_weighted[f'{str(query)}'].sort_values(ascending=False)
        # print(similar_artists_composite[1:])  # start from index 1 to skip artist himself       
        rankings = similar_artists_composite[1:]
    # if query in big_songs_set:
    #     song_ft = big_songs_df.loc[query]
    #     song_vect = song_ft.to_numpy()
    #     if song_vect.ndim > 1:
    #         song_vect = song_vect[0]
    #     print("song vector")
    #     print(song_vect)
    #     norm_song_vect = normalize_feature_vec(song_vect)
    #     norm_small_songs = normalize_feature_mat(small_songs_np)
    #     similarity_index = np.dot(norm_song_vect, norm_small_songs.T) / (np.linalg.norm(norm_song_vect) * np.linalg.norm(norm_small_songs))
        # rankings = np.argsort(similarity_index)[::-1]
        # print("similarity index")
        # print(np.shape(similarity_index))
        # print(similarity_index)
        # print("rankings indicies:")
        # print(rankings)
        res = []
        ranks = []
        for r in rankings.keys()[:10]:
            res.append([r, artist_to_image[r]])
            ranks.append(rankings[r])
        print("res:")
        print(res)
        return json.dumps(res)
    else: 
        return json.dumps([])

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/songs")

def song_search():
    text = request.args.get("title")
    return json_search(text)

@app.route("/update")

def update_search():
    global artist_to_image
    # print(query_artist)
    temp = request.args.get("json")
    # print("printing html:")
    # print(temp)
    curr_json = html_to_json.convert(temp)
    # print("printing json:")
    # print(curr_json)
    temp_lst = []
    df = curr_json['div']

    relevant_artists = []
    irrelevant_artists = []
    for i in range(1, len(df)):
        artist_frame = df[i]
        artist_name = artist_frame['div'][0]['h3'][0]['_value'][13:]

        if int(artist_frame['div'][0]['div'][0]['div'][0]['span'][1]['_value']) != 0:
            relevant_artists.append(artist_name)
        elif int(artist_frame['div'][0]['div'][0]['div'][1]['span'][1]['_value']) != 0:
            irrelevant_artists.append(artist_name)

    lst = update_vector(relevant_artists, irrelevant_artists)

    for artist in lst:
        temp_lst.append([artist, artist_to_image[artist]])
            
    print("printing list:")
    print(temp_lst)
    new_json = json.dumps(temp_lst)
    print("pring new_json:")
    print(new_json)

    return new_json



if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)


# debug = True

# np.set_printoptions(threshold=sys.maxsize)

# app = Flask(__name__)
# CORS(app)

# # Sample search using json with pandas
# def json_search(query):
#     if query in big_songs_set:
#         song_ft = big_songs_df.loc[query]
#         song_vect = song_ft.to_numpy()
#         if song_vect.ndim > 1:
#             song_vect = song_vect[0]
#         print("song vector")
#         print(song_vect)
#         similarity_index = np.dot(song_vect, small_songs_np.T) / (np.linalg.norm(song_vect) * np.linalg.norm(small_songs_np))
#         rankings = np.argsort(similarity_index)
#         print("similarity index")
#         print(np.shape(similarity_index))
#         print(similarity_index)
#         print("rankings:")
#         print(rankings)
#         res = []
#         for r in rankings[:10]:
#             res.append(small_songs_list[r])
#         print(res)
#         return json.dumps(res)
#     else: 
#         return json.dumps(small_songs_list[:10])

# @app.route("/")
# def home():
#     return render_template('base.html',title="sample html")

# @app.route("/songs")

# def song_search():
#     text = request.args.get("title")
#     return json_search(text)

# if 'DB_NAME' not in os.environ:
#     app.run(debug=True,host="0.0.0.0",port=5000)

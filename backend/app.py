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
from bs4 import BeautifulSoup
import scipy.sparse
import html.parser
import plotly.express as px
import plotly
# nltk.download('stopwords')
# nltk.download('wordnet')

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

artist_df = pd.read_csv(current_directory + '/kaggle_data/final_artist_dataframe.csv')
artist_df['Reviews'] = artist_df['Reviews'].fillna(value='')
artist_df['song_titles'] = artist_df['song_titles'].fillna(value='')
artist_df['average_score'] = artist_df['average_score'].fillna(value=0.0)

with open(current_directory + '/spotify_api/database_jsons/artist_images_dataframes.json', "r") as openfile: 
  artist_images_json = json.load(openfile)

save_path_1 = current_directory + '/kaggle_data/tfidf_matrix_lyrics.npz'
save_path_2 = current_directory + '/kaggle_data/tfidf_matrix_titles.npz'
save_path_3 = current_directory + '/kaggle_data/tfidf_matrix_reviews.npz'

tfidf_matrix_lyrics = scipy.sparse.load_npz(save_path_1)

tfidf_matrix_title = scipy.sparse.load_npz(save_path_2)

tfidf_matrix_reviews = scipy.sparse.load_npz(save_path_3)

cosine_sim_lyrics = cosine_similarity(tfidf_matrix_lyrics, tfidf_matrix_lyrics)
cosine_sim_lyrics_df = pd.DataFrame(cosine_sim_lyrics, index=artist_df['artist'], columns=artist_df['artist'])

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
for pair in artist_images_json:
    artist_name = pair["artist"]
    artist_img = pair["image"]
    artist_to_image[artist_name] = artist_img
    # except:
    #     artist_name = ""
    #     artist_img = ""
    #     artist_to_image[] = ""


lst = artist_df["artist"].tolist()
artist_set = []
for i in lst:
  artist_set.append(str(i))

print("first: ", artist_set)

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
    
    cosine_sim = cosine_similarity(tfidf_matrix_lyrics, tfidf_matrix_lyrics)
    cosine_sim_df = pd.DataFrame(cosine_sim, index=artist_df['artist'], columns=artist_df['artist'])

    # get similarity scores for Eminem
    similar_artists = cosine_sim_df[f'{query_artist}'].sort_values(ascending=False)
    return (similar_artists[1:], tfidf_matrix_lyrics)

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

    tfidf_vectorizer_reviews = TfidfVectorizer(max_features=7500, stop_words="english", ngram_range=(1,2))
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

    features_combined_weighted = features_combined_weighted_review
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

    print(updated_recommendations[1:10])

    res = []
    t_count = 0
    temp_i = 0
    while t_count < 10:
        a_name = updated_recommendations.index[temp_i]
        if (a_name in irrelevant_artists) or (str(query_artist) in a_name):
            pass
        else:
            #create graphs and add as strings

            lyric_score = round((cosine_sim_lyrics_df[query_artist][a_name])*100, 2)
            # try:
            overall_score = round(updated_recommendations.loc[a_name]["Similarity"]*100, 2)
            # except:
            #     overall_score = 0
            #add artist image
            try:
                res.append([a_name, artist_to_image[a_name], "", str(lyric_score), str(overall_score)])
            except:
                res.append([a_name, "", "", str(lyric_score), str(overall_score)])
            # print("res:")
            # print(res)
            t_count += 1
        temp_i += 1
    return json.dumps(res)

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
    # print(artist_set)
    if query in artist_set:
        # Specify the full path where you want to save the file
        cosine_sim_df_weighted = composite_vector(artist_df, tfidf_matrix_lyrics, tfidf_matrix_title, tfidf_matrix_reviews)
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
        t_count = 0
        temp_i = 0
        t_list =  list(rankings.keys())
        print(type(temp_i))

        while t_count < 10:
            r = t_list[int(temp_i)]
            if query_artist not in str(r):
                #create graphs and add as strings

                fig = px.scatter(x=range(10), y=range(10))
                graph_1 = plotly.io.to_html(fig, full_html=False, include_plotlyjs="cdn", default_width="300px", default_height="200px")

                print("printing graph html below")
                print(rankings[r])
                print("graph html finished")
                lyric_score = round((cosine_sim_lyrics_df[query_artist][r])*100, 2)

                overall_score = round(rankings[r]*100, 2)
                #add artist image
                try:
                    res.append([r, artist_to_image[r], graph_1, str(lyric_score), str(overall_score)])
                except:
                    res.append([r, "", graph_1, str(lyric_score), str(overall_score)])

                
                
                ranks.append(rankings[r])
                # print("res:")
                # print(res)
                t_count += 1
            else:
                pass
            temp_i += 1
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
    
    # print(curr_json)
    temp_lst = []
    relevant_artists = []
    irrelevant_artists = []
    bs_ob = BeautifulSoup(temp, "html.parser")
    for i in bs_ob.children:
        data_temp= (i.get_text())
        string = ''.join(list(map(lambda x: x.strip(), data_temp.split())))
        a_name, _, rating = string.partition("thumb_up")
        if a_name != "update":
            if int(rating[0]) != 0:
                relevant_artists.append(artist_name)
            elif int(rating[-1]) != 0:
                irrelevant_artists.append(artist_name)
    return update_vector(relevant_artists, irrelevant_artists)



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

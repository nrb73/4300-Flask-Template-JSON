import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv("/Users/meer/Desktop/4300/4300-Flask-Template-JSON/backend/kaggle_data/test_df.csv")
df.head()

artist_df = df.groupby('artist').agg({
    'title': ' '.join,  
    'views': 'mean',           
    'lyrics': ' '.join,    
    'tag': lambda x: list(x.unique())
}).reset_index()

artist_df.rename(columns={'title': 'song_titles', 'views': 'average_views', 'lyrics': 'concatenated_lyrics', 'tag': 'tags'}, inplace=True)

#PREPROCESSING LYRIC DATA

#lower case
artist_df['concatenated_lyrics'] = artist_df['concatenated_lyrics'].apply(lambda x: x.lower())

#removing stop words
stop_words = set(stopwords.words('english'))
artist_df['concatenated_lyrics'] = artist_df['concatenated_lyrics'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

#removing punctuation and special characters
artist_df['concatenated_lyrics'] = artist_df['concatenated_lyrics'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

#removing numbers
artist_df['concatenated_lyrics'] = artist_df['concatenated_lyrics'].apply(lambda x: ''.join([i for i in x if not i.isdigit()]))

#converting words to lemmatized/base form
lemmatizer = WordNetLemmatizer()
artist_df['concatenated_lyrics'] = artist_df['concatenated_lyrics'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

print(artist_df)

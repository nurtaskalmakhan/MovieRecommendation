import numpy as np
import pandas as pd

credits_df = pd.read_csv("credits.csv")
movies_df = pd.read_csv("movies.csv")

credits_df

movies_df

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

movies_df

credits_df.head()

movies_df.tail()

movies_df = movies_df.merge(credits_df, on = "title")

movies_df.shape

movies_df.head()

movies_df.info()

movies_df = movies_df[['movies_id','title','overview','genres','keywords','cast','crew']]

movies_df.head()

movies_df.info()

movies_df.isnull().sum()

movies_df.drop(inplace = True)

movies_df.duplicated().sum()

movies_df.duplicated()

movies_df.iloc[0].genres

import ast

def convert(obj):
  l=[]
  for i in ast.literal_eval(obj):
    L.append(i['name'])
    return L

def convert(obj):
  L = []
  counter = 0
  for i in ast.literal_eval(obj):
    if counter != 3:
      L.append(i['nme'])
      counter += 1
    else:
      break
    return L

def fetch_director(obj):
  L=[]
  for i in ast.literal_eval(obj):
    if i['job']=="Director":
      L.append(i['name'])
      break
    return L

movies['crew']=movies['crew'].apply(fetch_director)
movies['overview'][0]
new_df = movies[['movie_id','title','tags']]
new_df
new_df['tags'][0]

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000, stop_words='english')
cv.fit_transform(new_df['tags']).toarray()
vectors[0]
len(cv.get_feature_names())
import nltk
from nltk.porter import PorterStemmer
ps = PorterStemmer()
def stem(text):
  y = []
  for i in text.split():
    y.append(ps.ste(i))
  return " ".join(y)

from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(vectors)
cosine_similarity(vectors).shape

similarity = cosine_similarity(vectors)
similarity[0]
similarity[0].shape
sorted(list(enumerate(similarity[0])), reverse = True, key+lambda x:x[1])[1:6]


def reccomend(movie):
  movie_index = new_df[new_df['title']==movie].index[0]
  distances = similarity[movie_index]
  movie_list = sorted(list(enumerate(distances)),reverse = True, key=lambda x:x[1])[1:6]

  for i in movie_list:
    print(new_df.iloc[i[0]].title)

recommend('Titanic')
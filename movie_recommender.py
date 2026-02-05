import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Data files are being uploaded...")

movies = pd.read_csv('data/tmdb_5000_movies.csv')
credits = pd.read_csv('data/tmdb_5000_credits.csv')
series = pd.read_csv('data/tmdb_tv_shows.csv')

movies = movies.merge(credits, on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies['type'] = 'Movie'

def safe_literal_eval(val):
    try:
        if isinstance(val, str):
            return [i['name'] for i in eval(val)]
        return []
    except:
        return []

movies['genres'] = movies['genres'].apply(safe_literal_eval)
movies['keywords'] = movies['keywords'].apply(safe_literal_eval)
movies['cast'] = movies['cast'].apply(lambda x: safe_literal_eval(x)[:3])
movies['crew'] = movies['crew'].apply(lambda x: [i for i in safe_literal_eval(x) if 'Director' in str(i)][:1])
movies['overview'] = movies['overview'].apply(lambda x: str(x).split())

series.rename(columns={'name': 'title', 'id': 'movie_id'}, inplace=True)

series['type'] = 'Series'

for col in ['keywords', 'cast', 'crew']:
    series[col] = [[] for _ in range(len(series))]

series = series[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'type']]

series['genres'] = series['genres'].apply(lambda x: x.split (', ') if isinstance(x, str) else [])
series['overview'] = series['overview'].apply(lambda x: str(x).split())

all_content = pd.concat([movies, series], ignore_index=True)

def collapse(obj):
    return [i.replace(" ","") for i in obj]

all_content['genres'] = all_content['genres'].apply(collapse)
all_content['keywords'] = all_content['keywords'].apply(collapse)
all_content['cast'] = all_content['cast'].apply(collapse)
all_content['crew'] = all_content['crew'].apply(collapse)

all_content['tags'] = all_content['overview'] + all_content['genres'] + all_content['keywords'] + all_content['cast'] + all_content['crew']

del movies
del credits
del series

new_df = all_content[['title', 'tags', 'type']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

cv = CountVectorizer(max_features=5000, stop_words='english', dtype=np.float32)
vectors = cv.fit_transform(new_df['tags'])


def recommend(content_name):
    matches = new_df[new_df['title'] == content_name]
    
    if matches.empty:
        print(f"\n '{content_name} not found!")
        return

    movie_index = matches.index[0]

    similarity_scores = cosine_similarity(vectors[movie_index], vectors).flatten()

    top_indices = np.argpartition(similarity_scores, -6)[-6:]

    sorted_indices = top_indices[np.argsort(similarity_scores[top_indices])][::-1]

    print(f"\nIf you liked '{content_name}', you might like:\n")

    counter = 1
    for i in sorted_indices:
        if i == movie_index: continue
        title = new_df.iloc[i].title
        ctype = new_df.iloc[i].type
        print(f"{counter}. {title} ({ctype})")
        counter += 1


recommend('Avatar')
recommend('Breaking Bad')
recommend('Game of Thrones')





import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('data/tmdb_5000_movies.csv')
credits = pd.read_csv('data/tmdb_5000_credits.csv')

movies = movies.merge(credits, on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)

def convert(obj):
    name = []
    for i in ast.literal_eval(obj):
        name.append(i['name'])
    return name

def convert3(obj):
    name = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            name.append(i['name'])
            counter += 1
        else:
            break
    return name

def fetch_director(obj):    
    name = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            name.append(i['name'])
            break
    return name 

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3)
movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df = movies[['movie_id', 'title', 'tags']]

new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x)) 
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vectors)


def recommend(movie):
    try:
        movie_index = new_df[new_df['title'] == movie].index[0] 
        similarity_point = similarity[movie_index]
        movie_list = sorted(list(enumerate(similarity_point)), reverse = True, key = lambda x: x[1])[1:6]

        print(f"\nIf you liked the '{movie}', you'll like these too:\n")

        for order, i in enumerate(movie_list, 1):
            print(f"{order}. {new_df.iloc[i[0]].title}")

    except IndexError:
        print(f"\n'{movie}' not found! Please check the movie name.")

    
recommend('Avatar')
recommend('Batman Begins')





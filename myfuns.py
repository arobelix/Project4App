import pandas as pd
import numpy as np
import requests

# Define the URL for movie data
myurl = "https://liangfgithub.github.io/MovieData/movies.dat?raw=true"

# Fetch the data from the URL
response = requests.get(myurl)

# Split the data into lines and then split each line using "::"
movie_lines = response.text.split('\n')
movie_data = [line.split("::") for line in movie_lines if line]

# Create a DataFrame from the movie data
movies = pd.DataFrame(movie_data, columns=['movie_id', 'title', 'genres'])
movies['movie_id'] = movies['movie_id'].astype(int)

genres = list(
    sorted(set([genre for genres in movies.genres.unique() for genre in genres.split("|")]))
)

def get_displayed_movies():
    return movies.head(100)

def get_recommended_movies(new_user_ratings):
    newuser = np.empty(100)
    newuser[:] = np.nan
    for idx, val in enumerate(new_user_ratings.values()):
        newuser[idx] = val
    output = [int(s[1:]) for s in myIBCF(newuser)]
    ret = movies[movies['movie_id'] == output[0]]
    for i in range(1, 10):
        ret = pd.concat([ret, movies[movies['movie_id'] == output[i]]], axis = 0)

    return ret

def myIBCF(newuser):
    S = pd.read_csv("S.csv")
    best_movies = pd.read_csv("best_movies.csv")
    temp_S = np.nan_to_num(S)
    temp_newuser = np.nan_to_num(newuser).reshape(-1,1)
    part1 = temp_S @ temp_newuser
    newuser_binary = newuser.copy()    
    newuser_binary[~np.isnan(newuser_binary)] = 1.0
    newuser_binary = np.nan_to_num(newuser_binary).reshape(-1, 1)
    part2 = temp_S @ newuser_binary
    part2[np.isclose(part2, 0, atol=1e-4)] = np.nan
    pred_ratings = (part1 / part2).reshape(-1) 
    count = np.sum(~np.isnan(pred_ratings))
    output = []
    
    while len(output) < 10 and len(output) <= len(pred_ratings):
        output.append(S.columns[np.argsort(np.nan_to_num(pred_ratings))[-(len(output)+1)]])
    i = 0
    while len(output) < 10:
        if best_movies[i] not in output:
            output.append(best_movies[i])
        i+=1
    return output
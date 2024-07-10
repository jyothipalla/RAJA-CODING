#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd

# https://files.grouplens.org/datasets/movielens/ml-25m.zip

movies = pd.read_csv("C:\\Users\\91891\\Desktop\\devi\\movies.csv")
print("One ---------------------------------------------------------")
print(movies)


import re

def clean_title(title):
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    return title
movies["clean_title"] = movies["title"].apply(clean_title)
print("Two-------------------------------------------------------")
print(movies)




from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2))

tfidf = vectorizer.fit_transform(movies["clean_title"])





from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def search(title):
    #title = "1995"
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices].iloc[::-1]
    # print(results)
    return results



# In[2]:



import ipywidgets as widgets
from IPython.display import display

movie_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=False
)
#print(movie_input)



movie_list = widgets.Output()
#print(movie_list)

def on_type(data):
    with movie_list:
        movie_list.clear_output()
        #display(data)
        title = data["new"]
        if len(title) > 5:
            display(search(title))

print(movie_input.observe(on_type, names='value'))


print(display(movie_input, movie_list))
#movie_id = 89745


# In[3]:


import pandas as pd

ratings = pd.read_csv("C:\\Users\\91891\\Desktop\\ratings.csv")
print("---------------------------------------------------------------------")
print(ratings)
print("---------------------------------------------------------------------")
print(ratings.dtypes)



movie_id=1
similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
print("---------------------------------------------------------------------")
print(similar_users)
similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
print("---------------------------------------------------------------------")
print(similar_user_recs)
similar_user_recs = similar_user_recs.value_counts() / len(similar_users)
print("---------------------------------------------------------------------")
print(similar_user_recs)

similar_user_recs = similar_user_recs[similar_user_recs > .10]
print("---------------------------------------------------------------------")
print(similar_user_recs)

all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
print("---------------------------------------------------------------------")
print(all_users)
all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
print("---------------------------------------------------------------------")
print(all_user_recs)

rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
rec_percentages.columns = ["similar", "all"]
print("-------------------------------------------------------------------------")
print(rec_percentages)


rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]

print("-------------------------------------------------------------------------")
print(rec_percentages)
rec_percentages = rec_percentages.sort_values("score", ascending=False)

print("-------------------------------------------------------------------------")
print(rec_percentages)
rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")

print("-------------------------------------------------------------------------")
print(rec_percentages)





# In[4]:




def find_similar_movies(movie_id):
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

    similar_user_recs = similar_user_recs[similar_user_recs > .10]
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]
    
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)
    return rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]










import ipywidgets as widgets
from IPython.display import display

movie_name_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=False
)
recommendation_list = widgets.Output()

def on_type(data):
    with recommendation_list:
        recommendation_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            results = search(title)
            movie_id = results.iloc[0]["movieId"]
            display(find_similar_movies(movie_id))

movie_name_input.observe(on_type, names='value')

display(movie_name_input, recommendation_list)


# In[ ]:





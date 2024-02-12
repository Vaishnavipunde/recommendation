# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 08:25:42 2023

@author: sai
"""

import pandas as pd
anime=pd.read_csv("C:/2-dataset/anime.csv")
anime.shape

anime.columns
# only genre column
anime.genre

from sklearn.feature_extraction.text import TfidfVectorizer
#this is the term frequency inverse document.each row is trated as document
tfidf=TfidfVectorizer(stop_words='english')
#it is going to seperate out all words from row 
anime["genre"].isnull().sum()

anime["genre"]=anime["genre"].fillna("general")

#it was cread=ted sparse matrix it means that we have 47 genre on this particular matrix we want to do utem based recommendation if a user have we are watched gadar then you can recommended shershah movie
tfidf_matrix=tfidf.fit_transform(anime.genre)
tfidf_matrix.shape



from sklearn.metrics.pairwise import linear_kernel
#it was measuring similarity
cosine_sim_matrix=linear_kernel(tfidf_matrix,tfidf_matrix)
#each element of tfidf matrix compare with eacch elenemt of yfidf matrix

#we will try to match movie name with its index
anime_index=pd.Series(anime.index,index=anime["name"]).drop_duplicates()
#we are converting anime_index into series 
anime_id=anime_index["Assassins (1995)"]
anime_id
def get_recommendation(Name,topN):
    anime_id=anime_index[Name]


#we want to capture wholw row for that purpose we apply cosine
cosine_scores=list(enumerate(cosine_sim_matrix[anime_id]))

#the cosine scre is captured we want to arrange it in descending order according to their score
cosine_scores=sorted(cosine_scores,key=lambda x:x[1],reverse=True)

cosine_scores_N=cosine_scores[0: topN+1]
#getting movie index
anime_idx=[i[0] for i in cosine_scores_N]
#getting cosine score
anime_scores=[i[1] for i in cosine_scores_N]

anime_similar_show=pd.DataFrame(columns=["name","score"])
anime_similar_show["name"]=anime.loc[anime_idx,"name"]
anime_similar_show["score"]=anime_scores


anime_similar_show.reset_index(inplace=True)
anime_similar_show

get_recommendation('bad boys (1995)', topN=10)






























# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 20:03:46 2023

@author: rajendra


Topic: Recommendation Engine
Instructions:
Please share your answers filled in-line in the word document. Submit code separately wherever applicable.
	
Please ensure you update all the details:
Name: _____________ Batch ID: ___________
Topic: Recommender Engine
Hints:
1.	Business Problem
1.1.	What is the business objective?
1.1.	Are there any constraints?

2.	Work on each feature of the dataset to create a data dictionary as displayed in the image below:


3.	Data Pre-processing
2.1 Data Cleaning and Data Mining.
4.	Exploratory Data Analysis (EDA):
4.1.	Summary.
4.2.	Univariate analysis.
4.3.	Bivariate analysis.
	
5.	Model Building
5.1	Build the Recommender Engine model on the given data sets.

6.	Write about the benefits/impact of the solution - in what way does the business (client) benefit from the solution provided?


"""

#Problem Statement: -

#The Entertainment Company, which is an online movie watching platform, wants to improve its collection of movies and showcase those that are highly rated and recommend those movies to its customer by their movie watching footprint. For this, the company has collected the data and shared it with you to provide some analytical insights and also to come up with a recommendation algorithm so that it can automate its process for effective recommendations. The ratings are between -9 and +9.



import pandas as pd
anime=pd.read_csv("C:/2-dataset/Entertainment.csv")
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
























































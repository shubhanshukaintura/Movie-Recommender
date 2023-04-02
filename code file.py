import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#data collection & preprocessing
#loading the data from csv to pandas data frame
movies_data = pd.read_csv('movies.csv')
 
#printing first 5 rows of the data set
movies_data.head()

#selecting the relevant features for recommendation
selected_features = ['genres','keywords','tagline','cast','director']
print(selected_features)

#replacing the null values with null string
for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('')

#combining all the 5 selected features
combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']
print(combined_features)

#converting the text data to feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

#Cosine Similarity

#getting the similarity scores using consine similarity
similarity = cosine_similarity(feature_vectors)
movie_name = input('Enter Your Favourite Movie : ')

#creating a list with all the movie names given in the dataset
list_of_all_titles = movies_data['title'].tolist()

#finding the close match for the movie name given by the user
find_close_match = difflib.get_close_matches(movie_name,list_of_all_titles)

close_match = find_close_match[0]

index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

#getting a list of similar movies
similarity_score = list(enumerate(similarity[index_of_the_movie]))

#sorting the movies based on their similarity score
sorted_similar_movies = sorted(similarity_score,key = lambda x:x[1], reverse = True)
print(sorted_similar_movies)

#print the name of similar movies based on the index
print('Movies suggested for you : \n')

i=1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if(i<30):
    print(i,'.',title_from_index)
    i+=1

import pickle
pickle.dump(movies_data,open('movie_recm.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))

np.array(movies_data['title'])
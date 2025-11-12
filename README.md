# 🎬 Movie Recommendation System

## 🪄 Introduction
This notebook presents a **simple implementation** of a movie recommendation system using Python.  
It focuses on **content-based filtering**, where movies are recommended based on how similar their descriptions, genres, and keywords are.

The goal of this project is to demonstrate the **core logic** behind recommendation systems without using complex machine learning algorithms or large-scale user data.  
It is an **educational and simplified version** meant for learning and understanding how content-based systems work.

---

## 🎯 Project Objective
The main objective of this project is to:
- Build a basic recommendation engine using movie metadata.  
- Apply **text processing** techniques to represent movie information.  
- Calculate **similarity** between movies using **TF-IDF** and **Cosine Similarity**.  
- Recommend the top 5 movies that are most similar to a given movie.

---

## 🧱 Project Steps

### 1️⃣ Import Libraries
The required Python libraries are imported for data handling, text processing, and similarity computation.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import difflib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```
### 2️⃣ Load Data
```python
movies_data = pd.read_csv('movies.csv')
```
### 3️⃣ Feature Engineering
Combine relevant textual features into a single field called **combined_features** to represent each movie’s content.
```python
combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']
```
### 4️⃣ TF-IDF Vectorization
Transform the text data into numerical vectors using TF-IDF (Term Frequency–Inverse Document Frequency).
```python
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
```
### 5️⃣ Cosine Similarity
```python
similarity = cosine_similarity(feature_vectors)
```
### 6️⃣ Movies Recommendation System Function
Define a function to recommend the top 20 most similar movies.
```python
def movie_recommandation_system():
  movie_name_test = input(' Enter your favourite movie name : ')
  find_close_match_test = difflib.get_close_matches(movie_name_test, list_of_all_titles)
  index_of_the_movie_test = movies_data[movies_data.title == find_close_match_test[0]]['index'].values[0]
  similarity_score_test = list(enumerate(similarity[index_of_the_movie_test]))
  sorted_similar_movies_test = sorted(similarity_score_test, key = lambda x:x[1], reverse = True) 
  print('Movies suggested for you : \n')
  i = 1
  for movie in sorted_similar_movies_test:
    index = movie[0]
    title_from_index = movies_data[movies_data.index==index]['title'].values[0]
    if (i<21):
      print(i, '.',title_from_index)
      i+=1
```
Example:
```python
Enter your favourite movie name : the godfather
```
📊 Output Example:
```python
Movies suggested for you : 
1 . The Godfather
2 . The Godfather: Part III
3 . Apocalypse Now
4 . Closer
5 . The Godfather: Part II
6 . Mickey Blue Eyes
7 . August Rush
8 . Leaving Las Vegas
9 . Machete
10 . Dracula
11 . The Conversation
12 . Superman
13 . West Side Story
14 . American Graffiti
15 . The Score
16 . Peggy Sue Got Married
17 . Insomnia
18 . Love Actually
19 . This Thing of Ours
20 . The Son of No One
```
### ⚙️ Requirements
Install dependencies:
```python
pip install pandas numpy scikit-learn
```

### 🧠 Concepts Used
- **TF-IDF Vectorization** – converts movie descriptions into numerical vectors.
- **Cosine Similarity** – measures the closeness between two movie vectors.
- **Content-Based Filtering** – recommends movies based on content similarity.
### 🚀 Future Improvements
- Adding a Collaborative Filtering approach using user ratings.
- Building a Hybrid Model that combines both methods.
- Creating a web interface using Streamlit or Flask.

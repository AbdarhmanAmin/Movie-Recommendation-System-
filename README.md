
# 🎬 Movie Recommendation System

A Machine Learning–based Movie Recommendation System that suggests similar movies based on user selection using **Content-Based Filtering** and **Cosine Similarity**.

The system analyzes movie metadata such as genres, cast, crew, keywords, and overview to recommend the most relevant movies.

---

##  Project Overview

With the huge number of movies available today, users often struggle to find what to watch next.

This project solves that problem by building an intelligent recommendation engine that:

* Understands movie content
* Measures similarity between movies
* Suggests the most relevant recommendations

---

##  Features

* 🔎 Search movies by name
* 🎯 Recommend top similar movies
* 🧠 Content-Based Filtering
* 📊 Cosine Similarity Algorithm
* ⚡ Fast response using precomputed data
* 🌐 Interactive UI built with Streamlit

---

## 🛠️ Tech Stack

### 👨‍💻 Programming Language

* Python

### 📚 Libraries

* Pandas
* NumPy
* Scikit-learn
* NLTK
* Pickle

### 🌐 Deployment / Interface

* Streamlit

---

## 📂 Dataset

The model is trained on the **TMDB Movie Dataset**.

Dataset includes:

* Movie Title
* Genres
* Keywords
* Cast
* Crew
* Overview

---

## ⚙️ Project Workflow

### 1️⃣ Data Collection

* Load movies & credits datasets
* Merge them into one dataframe

---

### 2️⃣ Data Preprocessing

Steps:

* Remove null values
* Select important columns
* Extract cast & crew names
* Combine features into one column called **tags**

Example features used:

```
Genres + Keywords + Cast + Crew + Overview
```

---

### 3️⃣ Text Vectorization

We convert textual data into numerical vectors using **CountVectorizer**.

```python
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(
    max_features=10000,
    stop_words='english'
)

vector = cv.fit_transform(
    df['tags'].values.astype('U')
).toarray()
```

#### Explanation

* Converts text → numbers
* Removes common English words
* Limits vocabulary to top 10,000 words

---

### 4️⃣ Similarity Calculation

We calculate similarity using **Cosine Similarity**.

```python
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vector)
```

This generates a similarity matrix between all movies.

---

### 5️⃣ Saving Processed Files

To avoid recomputation, we save processed data using Pickle.

```python
import pickle

pickle.dump(df, open('movies.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
```

Saved files:

* **movies.pkl** → Movie dataframe
* **similarity.pkl** → Similarity matrix

---

## 🧠 Recommendation Function

```python
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    for i in movies_list:
        print(movies.iloc[i[0]].title)
```

---

## 🖥️ Streamlit Web App

The project includes an interactive UI where users can:

* Select a movie from dropdown
* Or type movie name
* Get instant recommendations

Run locally:

```bash
streamlit run app.py
```

---

## 📦 Project Structure

```
Movie-Recommendation-System/
│
├── app.py
├── movies.pkl
├── similarity.pkl
├── notebook.ipynb
├── requirements.txt
└── README.md
```

---

## 📥 Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/Movie-Recommendation-System.git
cd Movie-Recommendation-System
```

### 2️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Application

```bash
streamlit run "app.py"
```

---



## 🧪 Future Improvements

* 🎞️ Add movie posters
* ⭐ Add user ratings
* 🤖 Hybrid recommendation system
* 🧠 Deep Learning model
* ☁️ Cloud deployment (AWS / GCP)

---

## 🤝 Contributing

Contributions are welcome.

Steps:

1. Fork the repository
2. Create new branch
3. Commit changes
4. Open Pull Request

---
## 👨‍💻 Author

**Abdarhman Magdy Amin**
---

⭐ If you like this project, don’t forget to star the repo!

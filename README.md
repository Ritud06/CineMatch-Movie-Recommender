# 🎬 CineMatch — AI Powered Movie Recommendation System

A complete data science web application that recommends 
similar movies using Machine Learning and NLP.
Built on 5,000 real movies from the TMDB dataset.

## 🚀 Live Demo
Coming soon...

## ✨ Features
- 🍿 **Movie Recommender** — Get 5 similar movies instantly using ML
- ℹ️ **Movie Details** — Full rating, cast, director, budget, revenue
- 🏆 **Top Rated by Genre** — Best movies in any of 19 genres
- 📊 **Statistics Dashboard** — 5 data visualizations of film industry
- ⚖️ **Movie Comparison** — Compare 2 movies head to head with verdict
- 🌙 **Dark / Light Mode** — Theme toggle for comfort

## 🛠️ Tech Stack
| Technology | Purpose |
|---|---|
| Python | Primary programming language |
| Pandas | Data cleaning and processing |
| NumPy | Numerical computing |
| Scikit-learn | CountVectorizer and Cosine Similarity |
| NLTK | Porter Stemmer for NLP |
| Matplotlib | Data visualizations |
| Streamlit | Web application framework |
| OMDb API | Fetching real movie posters |
| Pickle | Saving and loading ML model |

## 🧠 How It Works
1. Movie features — genres, cast, director, keywords, plot — are combined into tags
2. Porter Stemmer normalizes the text (loving → love)
3. CountVectorizer converts tags into numerical vectors
4. Cosine Similarity computes similarity between all 4,809 movies
5. Top 5 most similar movies are returned as recommendations

## ▶️ How to Run Locally

**Step 1 — Clone the repository**
```bash
git clone https://github.com/Ritud06/CineMatch-Movie-Recommender.git
```

**Step 2 — Install required libraries**
```bash
pip install -r requirements.txt
```

**Step 3 — Generate similarity.pkl**
Run the Jupyter notebook completely first:

## 📊 Dataset
TMDB 5000 Movie Dataset from Kaggle
- 4,809 movies with full metadata
- Cast and crew information
- Ratings, budget, revenue data

## 🎯 Key Highlights
- Built completely independently during internship
- Self-learned ML and NLP before formally studying them
- Goes beyond basic recommender with 4 extra features
- Professional UI with dark and light mode
- Real movie posters via OMDb API integration

## 👩‍💻 Built By
**Ritu**
Data Science 

## 📧 Contact
GitHub: https://github.com/Ritud06


import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Load data
file = Path(r'D:\Netflix Content Analysis\netflix_titles.csv')
df = pd.read_csv(file)

# Cleaning
df['director'] = df['director'].fillna('Unknown')
df['description'] = df['description'].fillna('')
df['rating'] = df['rating'].fillna('NR')
df['country'] = df['country'].fillna('Not Specified')

# Weight successful content (higher rating gets more importance)
rating_weight = {
    'G':1, 'PG':2, 'PG-13':3, 'R':4, 'TV-MA':4, 'TV-14':3,
    'TV-PG':2, 'TV-Y':1, 'TV-Y7':1.5, 'NR':1, 'NR':1
}
df['weight'] = df['rating'].map(rating_weight).fillna(1)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X = tfidf.fit_transform(df['description'])

# Nearest Neighbors Model
nn = NearestNeighbors(n_neighbors=5, metric='cosine')
nn.fit(X)

# Save models
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump(nn, "genre_decision_model.pkl")

print("✅ Model training complete and saved!")

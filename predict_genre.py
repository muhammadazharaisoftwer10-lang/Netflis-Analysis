import pandas as pd
import joblib
from pathlib import Path

# Load trained models
tfidf = joblib.load("tfidf_vectorizer.pkl")
nn = joblib.load("genre_decision_model.pkl")

# Load dataset again
file = Path(r'D:\Netflix Content Analysis\netflix_titles.csv')
df = pd.read_csv(file)
df['description'] = df['description'].fillna('')
df['director'] = df['director'].fillna('Unknown')

# Input for genre suggestion
job = input("Enter content idea (example: crime drama, AI documentary etc): ")

# Transform input
vec = tfidf.transform([job])

# Find closest content
dist, idx = nn.kneighbors(vec)

# Get genre suggestions
genres = df.iloc[idx[0]]['listed_in']
genre_list = []
for g in genres:
    for item in g.split(','):
        genre_list.append(item.strip())

from collections import Counter
final_genre = Counter(genre_list).most_common(1)[0][0]

print("\n🎯 Recommended Genre for Netflix next year:", final_genre)

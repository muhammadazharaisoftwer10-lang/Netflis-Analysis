import pandas as pd
import streamlit as st
import plotly.express as px
import joblib
from pathlib import Path
from collections import Counter

# ----------------------------- LOAD DATA -----------------------------
file = Path(r'D:\Netflix Content Analysis\netflix_titles.csv')
df = pd.read_csv(file)

# ----------------------------- DATA CLEANING -----------------------------
df['director'] = df['director'].fillna('Unknown')
df['description'] = df['description'].fillna('')
df['country'] = df['country'].fillna('Not Specified')
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
df['year_added'] = df['date_added'].dt.year
df['month_added'] = df['date_added'].dt.month_name()

# Fill missing ratings
df['rating'] = df['rating'].fillna('Not Rated')

# ----------------------------- LOAD ML MODELS -----------------------------
tfidf_path = Path("tfidf_vectorizer.pkl")
nn_path = Path("genre_decision_model.pkl")
if tfidf_path.exists() and nn_path.exists():
    tfidf = joblib.load(tfidf_path)
    nn = joblib.load(nn_path)
else:
    tfidf = nn = None

# ----------------------------- STREAMLIT PAGE CONFIG -----------------------------
st.set_page_config(page_title="Netflix Executive Dashboard", layout="wide", page_icon="🎬")
st.title("🎬 Netflix Content Analysis — Executive Dashboard")

# ----------------------------- METRICS -----------------------------
m1, m2, m3, m4, m5 = st.columns(5, gap="large")
m1.metric("Total Content", len(df))
m2.metric("Total Movies", len(df[df['type'] == "Movie"]))
m3.metric("Total TV Shows", len(df[df['type'] == "TV Show"]))
m4.metric("Unique Directors", df['director'].nunique())
m5.metric("Unique Countries", df['country'].nunique())

st.divider()

# ----------------------------- SIDEBAR FILTERS -----------------------------
with st.sidebar:
    st.header("🔍 Filters")
    content_type = st.multiselect(
        "Content Type",
        options=["Movie", "TV Show"],
        default=["Movie", "TV Show"]
    )
    country_filter = st.multiselect(
        "Country",
        options=sorted(df['country'].unique()),
        default=sorted(df['country'].unique())
    )
    year_filter = st.slider(
        "Year Added",
        min_value=int(df['year_added'].min()),
        max_value=int(df['year_added'].max()),
        value=(int(df['year_added'].min()), int(df['year_added'].max()))
    )

# ----------------------------- FILTER DATA -----------------------------
df_filtered = df[
    (df['type'].isin(content_type)) &
    (df['country'].isin(country_filter)) &
    (df['year_added'] >= year_filter[0]) &
    (df['year_added'] <= year_filter[1])
]

# ----------------------------- CONTENT TREND -----------------------------
st.subheader("📈 Content Trend (Movies vs TV Shows)")
dist_year = df_filtered.groupby(['year_added', 'type']).size().reset_index(name='count')
fig1 = px.line(dist_year, x='year_added', y='count', color='type',
               markers=True, title="Content Added Over Years")
st.plotly_chart(fig1, use_container_width=True)

st.divider()

# ----------------------------- GENRE DISTRIBUTION -----------------------------
st.subheader("🎭 Genre Distribution")
df_genres = df_filtered.copy()
df_genres['listed_in'] = df_genres['listed_in'].str.split(',')
df_genres = df_genres.explode('listed_in')
df_genres['listed_in'] = df_genres['listed_in'].str.strip()
genre_counts = df_genres['listed_in'].value_counts().reset_index()
genre_counts.columns = ['Genre', 'Count']
fig2 = px.bar(genre_counts, x='Genre', y='Count', title="Most Frequent Genres",
              color='Count', text='Count')
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ----------------------------- RELEASES BY MONTH -----------------------------
st.subheader("🔥 Releases by Month")
month_order = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]
month_counts = df_filtered['month_added'].value_counts().reindex(month_order).fillna(0)
fig3 = px.imshow([month_counts.values], x=month_order, y=["Releases"], color_continuous_scale='Oranges',
                 text_auto=True, title="Monthly Release Heatmap")
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ----------------------------- RATING PATTERNS -----------------------------
st.subheader("⭐ Rating Patterns")
rating_counts = df_filtered['rating'].value_counts().reset_index()
rating_counts.columns = ['Rating', 'Count']
fig4 = px.pie(rating_counts, names='Rating', values='Count', title="Content Ratings Distribution")
st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ----------------------------- GENRE RECOMMENDER -----------------------------
st.subheader("🤖 Genre Decision AI")
if tfidf and nn:
    idea = st.text_input("Enter content idea (example: sci-fi crime, AI documentary etc)")
    if st.button("🔮 Recommend Genres"):
        vec = tfidf.transform([idea])
        _, idx = nn.kneighbors(vec)
        genres = df.iloc[idx[0]]['listed_in']
        genre_list = []
        for g in genres:
            for item in g.split(','):
                genre_list.append(item.strip())
        top_genres = [g for g, _ in Counter(genre_list).most_common(3)]
        st.success(f"🎯 Recommended Genres: {', '.join(top_genres)}")
else:
    st.warning("Model not trained yet! Run: python train_model.py")

st.divider()

# ----------------------------- DOWNLOAD DATA -----------------------------
st.subheader("⬇ Cleaned Dataset Download")
st.download_button(
    "Download CSV",
    df_filtered.to_csv(index=False).encode('utf-8'),
    "cleaned_netflix_data.csv",
    "text/csv"
)

st.success("🚀 Dashboard Updated — Ready for Executive Use!")

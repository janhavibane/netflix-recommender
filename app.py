# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# App Title
st.title("🎬 Netflix Data Analysis & Recommendation System")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv")
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df['year_added'] = df['date_added'].dt.year
    df['month_added'] = df['date_added'].dt.month
    df['description'] = df['description'].fillna("")
    for col in ['country', 'rating', 'cast', 'director']:
        df[col] = df[col].fillna("Unknown")
    return df

df = load_data()

# Sidebar
st.sidebar.header("Navigation")
options = st.sidebar.radio("Go to", ["Dataset Overview", "Visualizations", "WordCloud", "Recommendation"])

# Dataset Overview
if options == "Dataset Overview":
    st.subheader("🧾 Dataset Preview")
    st.dataframe(df.head())
    st.write("Shape:", df.shape)

# Visualizations
elif options == "Visualizations":
    st.subheader("📊 Visual Insights")

    # Content type
    st.markdown("**Content Type Distribution**")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='type', data=df, ax=ax1, palette='Set2')
    st.pyplot(fig1)

    # Year added
    st.markdown("**Titles Added per Year**")
    by_year = df['year_added'].value_counts().sort_index()
    fig2, ax2 = plt.subplots()
    sns.lineplot(x=by_year.index, y=by_year.values, marker='o', ax=ax2)
    st.pyplot(fig2)

    # Top countries
    st.markdown("**Top Countries Producing Content**")
    top_countries = df['country'].value_counts().head(10)
    fig3, ax3 = plt.subplots()
    sns.barplot(y=top_countries.index, x=top_countries.values, ax=ax3, palette='magma')
    st.pyplot(fig3)

    # Top genres
    st.markdown("**Top Genres**")
    genres = df['listed_in'].str.split(',').explode().value_counts().head(10)
    fig4, ax4 = plt.subplots()
    genres.plot(kind='barh', ax=ax4, color='teal')
    st.pyplot(fig4)

# WordCloud
elif options == "WordCloud":
    st.subheader("☁️ WordCloud from Titles")
    title_words = ' '.join(df['title'].dropna())
    wc = WordCloud(width=800, height=400, background_color='white').generate(title_words)
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    ax5.imshow(wc, interpolation='bilinear')
    ax5.axis('off')
    st.pyplot(fig5)

# Recommendation
elif options == "Recommendation":
    st.subheader("🎯 Movie Recommendation System")
    
    # TF-IDF and cosine similarity
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['description'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df['title'].str.lower()).drop_duplicates()

    title_input = st.text_input("Enter a Movie Title (e.g. The Matrix)").lower()

    if st.button("Recommend"):
        if title_input not in indices:
            st.error("Title not found. Try a different one.")
        else:
            idx = indices[title_input]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
            movie_indices = [i[0] for i in sim_scores]
            st.success(f"🎬 Recommendations for '{title_input.title()}':")
            for i in df['title'].iloc[movie_indices]:
                st.write("- ", i)

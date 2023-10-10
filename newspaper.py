import streamlit as st
import requests
import plotly.express as px
import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from googletrans import Translator

# Set Streamlit page configuration
st.set_page_config(page_title="News Search and Sentiment Analysis", page_icon=":newspaper:", layout="wide")

# Function to load search history
def load_search_history():
    try:
        with open("search_history.txt", mode="r") as file:
            return file.read().splitlines()[-5:]  # Limit to the last 5 entries
    except FileNotFoundError:
        return []

# Function to search for news articles
def search_news(keyword):
    response = requests.get(f"https://newsapi.org/v2/everything?q={keyword}&apiKey=89de75b718bb45ba884f256d3b1710cc")
    articles = []

    if response.status_code == 200:
        data = response.json()
        if 'articles' in data:
            articles = data['articles']

    return articles

# Function to perform sentiment analysis and return sentiment label
def get_sentiment_label(content, sia):
    sentiment_scores = sia.polarity_scores(content)
    if sentiment_scores['compound'] > 0:
        return "Positive"
    elif sentiment_scores['compound'] < 0:
        return "Negative"
    else:
        return "Neutral"

# Function to display articles and sentiment analysis
def display_articles(articles, sia):
    positive_count = 0
    negative_count = 0
    neutral_count = 0

    for index, article in enumerate(articles):
        with st.expander(f"Article {index + 1} - {article.get('title', 'No title available')}"):
            title = article.get('title', 'No title available')
            author = article.get('author', 'No author available')
            url = article.get('url', '#')
            content = article.get('content', '')

            sentiment = get_sentiment_label(content, sia)

            if sentiment == "Positive":
                positive_count += 1
            elif sentiment == "Negative":
                negative_count += 1
            else:
                neutral_count += 1

            st.write(f"Title: {title}")
            st.write(f"Author: {author}")
            st.write(f"Link: [Read More]({url})")
            st.write(f"Sentiment: {sentiment}")

    return positive_count, negative_count, neutral_count

# Function to display sentiment distribution chart
def display_sentiment_chart(positive_count, negative_count, neutral_count):
    sentiment_data = {
        "Sentiment": ["Positive", "Negative", "Neutral"],
        "Count": [positive_count, negative_count, neutral_count]
    }
    sentiment_df = pd.DataFrame(sentiment_data)

    fig_sentiment = px.bar(sentiment_df, x="Sentiment", y="Count", color="Sentiment",
                           labels={'Count': 'Number of Articles'}, title='Sentiment Analysis Distribution')
    st.plotly_chart(fig_sentiment, use_container_width=True)

# Function to display articles per source chart
def display_articles_per_source(articles):
    sources = [article.get('source', {}).get('name', 'Unknown') for article in articles]
    unique_sources = list(set(sources))
    source_counts = [sources.count(source) for source in unique_sources]

    fig = px.bar(x=unique_sources, y=source_counts, labels={'x': 'Source', 'y': 'Number of Articles'},
                 title='Number of Articles per Source', text=source_counts)
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

# Function to display word cloud
def display_word_cloud(articles):
    all_content = " ".join([article.get('content', '') for article in articles])

    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_content)

    st.subheader("Word Cloud")
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

# Function to display search history
def display_search_history(search_history):
    if search_history:
        st.subheader("Search History")
        for idx, query in enumerate(search_history):
            st.write(f"{idx+1}. {query}")

# Function to display trending topics
def display_trending_topics(search_history):
    if search_history:
        st.subheader("Trending Topics")
        query_counts = Counter(search_history)
        trending_topics = query_counts.most_common(5)
        for idx, (topic, count) in enumerate(trending_topics):
            st.write(f"{idx+1}. {topic} ({count} searches)")

# Create a list to store bookmarked articles
bookmarked_articles = []

# Function to toggle bookmarking an article
def toggle_bookmark(article_title):
    if article_title in bookmarked_articles:
        bookmarked_articles.remove(article_title)
    else:
        bookmarked_articles.append(article_title)

# Function to display bookmarked articles
def display_bookmarked_articles():
    if bookmarked_articles:
        st.subheader("Bookmarked Articles")
        for article_title in bookmarked_articles:
            st.write(f"- {article_title}")

# Function to translate text
def translate_text(text, target_language):
    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    return translation.text

# Function to display translation feature
def display_translation_feature(articles):
    st.subheader("Language Translation")

    article_to_translate = st.selectbox("Select an article to translate:", [article['title'] for article in articles])

    target_language = st.selectbox("Select the target language:", ["en", "es", "fr", "de"])  # Add more languages as needed

    if st.button("Translate"):
        article = next((article for article in articles if article['title'] == article_to_translate), None)
        if article:
            translated_content = translate_text(article['content'], target_language)
            st.subheader(f"Translated Content ({target_language.upper()}):")
            st.write(translated_content)

# Function to display related news articles based on the selected article
def display_related_articles(articles, selected_article_title):
    st.subheader(f"Related Articles to '{selected_article_title}':")
    
    # Filter articles with similar keywords in the title
    related_articles = [article for article in articles if selected_article_title.lower() in article['title'].lower()]
    
    if related_articles:
        for index, article in enumerate(related_articles):
            st.write(f"Article {index + 1} - {article.get('title', 'No title available')}")
            st.write(f"Link: [Read More]({article.get('url', '#')})")
    else:
        st.write("No related articles found.")

# Function to clear the search history
def clear_search_history():
    with open("search_history.txt", mode="w") as file:
        file.write("")

# Main Streamlit app
def main():
    st.title("News Search and Sentiment Analysis")
    st.caption("Welcome! Start searching keywords on the web and visualize the sentiment ")

    search_history = load_search_history()

    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()

    input_data = st.text_input("Enter keywords")

    if input_data:
        search_history.append(input_data)

        with open("search_history.txt", mode="w") as file:
            file.write("\n".join(search_history))

        articles = search_news(input_data)

        st.info(f"Found {len(articles)} articles")

        positive_count, negative_count, neutral_count = display_articles(articles, sia)

        display_sentiment_chart(positive_count, negative_count, neutral_count)

        if articles:
            display_articles_per_source(articles)
            display_word_cloud(articles)
            display_translation_feature(articles)  # Add the translation feature
            display_related_articles(articles, input_data)  # Display related articles based on search input

    display_search_history(search_history)
    display_trending_topics(search_history)

    # Disruptive functions
    if st.button("Clear Search History"):
        clear_search_history()
        st.success("Search history has been cleared.")

    if st.button("Start a New Search"):
        st.text_input("Enter keywords")  # Allow the user to enter a new search query
        st.button("Search")  # Trigger the search again

if __name__ == "__main__":
    main()

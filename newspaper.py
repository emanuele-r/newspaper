import streamlit as st
import requests
import plotly.express as px
import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

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

# Function to clear the search history
def clear_search_history():
    with open("search_history.txt", mode="w") as file:
        file.write("")

# Function to display sentiment distribution by source
def display_sentiment_by_source(articles):
    st.subheader("Sentiment Distribution by Source")
    
    source_list = list(set([article.get('source', {}).get('name', 'Unknown') for article in articles]))
    selected_source = st.selectbox("Select a news source:", source_list)
    
    filtered_articles = [article for article in articles if article.get('source', {}).get('name', 'Unknown') == selected_source]
    
    if filtered_articles:
        positive_count, negative_count, neutral_count = display_articles(filtered_articles, sia)
        display_sentiment_chart(positive_count, negative_count, neutral_count)
    else:
        st.warning(f"No articles found for the selected source: {selected_source}")

# Function to display sentiment trends over time
def display_sentiment_over_time(articles):
    st.subheader("Sentiment Trends Over Time")
    
    start_date = st.date_input("Select a start date")
    end_date = st.date_input("Select an end date", max_value=start_date)
    
    filtered_articles = [article for article in articles if start_date <= pd.to_datetime(article.get('publishedAt')).date() <= end_date]
    
    if filtered_articles:
        positive_count, negative_count, neutral_count = display_articles(filtered_articles, sia)
        display_sentiment_chart(positive_count, negative_count, neutral_count)
    else:
        st.warning("No articles found within the selected date range.")

# Function to highlight keywords in article content
def highlight_keywords(articles):
    st.subheader("Keyword Highlighting")
    
    keyword = st.text_input("Enter a keyword to highlight")
    
    for article in articles:
        content = article.get('content', '')
        if keyword in content:
            st.subheader(f"Article - {article.get('title', 'No title available')}")
            highlighted_content = content.replace(keyword, f"**{keyword}**")
            st.write(f"Title: {article.get('title', 'No title available')}")
            st.write(f"Author: {article.get('author', 'No author available')}")
            st.write(highlighted_content)

# Function to recommend related articles based on sentiment
def recommend_related_articles(articles):
    st.subheader("Related Articles Recommendation")
    
    selected_article = st.selectbox("Select an article for recommendations:", [article['title'] for article in articles])
    selected_article_sentiment = get_sentiment_label([article['content'] for article in articles if article['title'] == selected_article][0], sia)
    
    recommended_articles = [article['title'] for article in articles if get_sentiment_label(article['content'], sia) == selected_article_sentiment and article['title'] != selected_article]
    
    if recommended_articles:
        st.write(f"Articles with similar sentiment to '{selected_article}':")
        for title in recommended_articles:
            st.write(f"- {title}")
    else:
        st.warning("No related articles found.")

# ... Rest of the code ...

# Inside the main function
if input_data:
    search_history.append(input_data)

    with open("search_history.txt", mode="w") as file:
        file.write("\n".join(search_history))

    articles = search_news(input_data)

    st.info(f"Found {len(articles)} articles")

    display_sentiment_filter(articles)  # Add sentiment filter
    display_article_summarization(articles)  # Add article summarization
    export_to_csv(articles)  # Add data export feature

# ... Rest of the code ...

# Run the app
if __name__ == "__main__":
    main()

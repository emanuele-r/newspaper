import streamlit as st
import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd  # Import pandas for data analytics

# Set Streamlit page configuration
st.set_page_config(
    page_title="News Search and Sentiment Analysis",
    page_icon=":newspaper:",
    layout="wide"
)

nltk.download("vader_lexicon")

search_history = []
user_score = 0
sia = SentimentIntensityAnalyzer()
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
lda = LatentDirichletAllocation(n_components=5, random_state=42)

# Function to load search history
def load_search_history():
    try:
        with open("search_history.txt", mode="r") as file:
            return file.read().splitlines()[-5:]  # Limit to the last 5 entries
    except FileNotFoundError:
        return []

# Function to search for news articles
def search_news(keyword):
    api_key = "89de75b718bb45ba884f256d3b1710cc"
    response = requests.get(f"https://newsapi.org/v2/everything?q={keyword}&apiKey={api_key}")
    articles = []

    if response.status_code == 200:
        data = response.json()
        if 'articles' in data:
            articles = data['articles']

    return articles

# Function to perform sentiment analysis and return sentiment label
def get_sentiment_label(content):
    sentiment_scores = sia.polarity_scores(content)
    if sentiment_scores['compound'] > 0:
        return "Positive"
    elif sentiment_scores['compound'] < 0:
        return "Negative"
    else:
        return "Neutral"

# Function to extract topics from articles
def extract_topics(articles):
    content = [article.get('content', '') for article in articles]
    tfidf = tfidf_vectorizer.fit_transform(content)
    lda.fit(tfidf)
    return lda

def generate_summary(article):
    content = article.get('content', '')
    return summarizer_model(content)

def display_articles(articles):
    positive_count, negative_count, neutral_count = 0, 0, 0
    article_data = []

    for index, article in enumerate(articles):
        title = article.get('title', 'No title available')
        content = article.get('content', '')
        author = article.get("author", "")
        link = article.get("link", "")

        # Generate a summary
        summary = generate_summary(article)

        sentiment = get_sentiment_label(content)

        with st.expander(f"Article {index + 1} - {title}"):
            st.write(f"Title: {title}")
            st.write(f"Author: {author}")
            st.write(f"Link to News: {link}")
            st.write(f"Summary: {summary}")  # Display the summary
            st.write(f"Sentiment: {sentiment}")

            user_answer = st.text_input(f"Answer for Article {index + 1}", key=f"answer_{index}")
            correct_answer = "Your Correct Answer"  # Set the correct answer

            if user_answer.lower() == correct_answer.lower():
                st.success("Correct! You earned points.")
                user_score += 10
            else:
                st.error("Sorry, that's incorrect.")

        if sentiment == "Positive":
            positive_count += 1
        elif sentiment == "Negative":
            negative_count += 1
        else:
            neutral_count += 1

        # Collect data for analytics
        article_data.append({
            "Title": title,
            "Author": author,
            "Link": link,
            "Summary": summary,
            "Sentiment": sentiment,
        })

    return positive_count, negative_count, neutral_count, article_data

# Function to display topics and analytics
def display_topics_and_analytics(articles, article_data):
    st.subheader("Topics Tags")
    lda = extract_topics(articles)

    for topic_idx, topic in enumerate(lda.components_):
        st.write(f"Topic {topic_idx + 1}:")
        top_words = [tfidf_vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
        st.write(", ".join(top_words))

    st.subheader("Data Analytics")
    # Create a pandas DataFrame from the article data
    df = pd.DataFrame(article_data)
    st.dataframe(df)

    st.subheader("Sentiment Distribution")
    sentiment_counts = df["Sentiment"].value_counts()
    st.bar_chart(sentiment_counts)

# Inside the main function
input_data = st.text_input("Enter a keyword to search for news")
if input_data:
    search_history.append(input_data)

    with open("search_history.txt", mode="w") as file:
        file.write("\n".join(search_history))

    articles = search_news(input_data)
    article_data = display_articles(articles)

    st.info(f"Found {len(articles)} articles")

    display_topics_and_analytics(articles, article_data)

# Display the user's score
st.write(f"Your Score: {user_score}")

# Run the app
if __name__ == "__main__":
    st.write("Welcome to the News Search and Sentiment Analysis app.")

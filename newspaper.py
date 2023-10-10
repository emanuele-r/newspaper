import streamlit as st
import requests
import plotly.express as px
import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import speech_recognition as sr
from gtts import gTTS  # Added for text-to-speech

# Set Streamlit page configuration
st.set_page_config(page_title="News Search and Sentiment Analysis", page_icon=":newspaper:", layout="wide")

nltk.download("vader_lexicon")
nltk.download("punkt")

search_history = []
bookmarked_articles = []
user_score = 0  # Initialize user's score
user_emotions = {}
sia = SentimentIntensityAnalyzer()

# Initialize the SpeechRecognition recognizer
r = sr.Recognizer()

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
def get_sentiment_label(content):
    sentiment_scores = sia.polarity_scores(content)
    if sentiment_scores['compound'] > 0:
        return "Positive"
    elif sentiment_scores['compound'] < 0:
        return "Negative"
    else:
        return "Neutral"

# Function to display articles and sentiment analysis
def display_articles(articles):
    positive_count, negative_count, neutral_count = 0, 0, 0

    for index, article in enumerate(articles):
        with st.expander(f"Article {index + 1} - {article.get('title', 'No title available')}"):
            title = article.get('title', 'No title available')
            author = article.get('author', 'No author available')
            url = article.get('url', '#')
            content = article.get('content', '')

            sentiment = get_sentiment_label(content)

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

            # Add a quiz or challenge to each article
            user_answer = st.text_input("Answer the question related to the article")
            correct_answer = "Your Correct Answer"  # Set the correct answer

            if user_answer.lower() == correct_answer.lower():
                st.success("Correct! You earned points.")
                user_score += 10  # Assign points to the user
            else:
                st.error("Sorry, that's incorrect.")

            # Add Text-to-Speech feature
            if st.button("Listen to Article"):
                tts = gTTS(content)
                st.audio(tts.get_audio_data(format="audio/wav"))

    return positive_count, negative_count, neutral_count

# Function to perform voice search
def voice_search():
    with sr.Microphone() as source:
        st.info("Listening for your voice command...")
        audio = r.listen(source)

    try:
        query = r.recognize_google(audio)
        st.text_input("Voice Search", query)

        # Perform the search using the recognized query
        articles = search_news(query)
        display_articles(articles)

    except sr.UnknownValueError:
        st.warning("Could not understand the audio.")
    except sr.RequestError:
        st.error("Could not request results; check your network connection.")

# Rest of your code (functions)...

# Inside the main function
input_data = st.text_input("Enter a keyword to search for news")
if input_data:
    search_history.append(input_data)

    with open("search_history.txt", mode="w") as file:
        file.write("\n".join(search_history))

    articles = search_news(input_data)

    st.info(f"Found {len(articles)} articles")

    display_articles(articles)  # Add quizzes/challenges to articles

    # Rest of your Streamlit app here...

# Add a button for voice search
if st.button("Voice Search"):
    voice_search()

# Display the user's score
st.write(f"Your Score: {user_score}")

# Run the app
if __name__ == "__main__":
    st.write("Welcome to the News Search and Sentiment Analysis app.")

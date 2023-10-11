import streamlit as st
import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Set Streamlit page configuration
st.set_page_config(page_title="News Search and Sentiment Analysis", page_icon=":newspaper:", layout="wide")

nltk.download("vader_lexicon")

search_history = []
user_score = 0  # Initialize user's score
sia = SentimentIntensityAnalyzer()

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

            user_answer = st.text_input(f"Answer for Article {index + 1}", key=f"answer_{index}")
            correct_answer = "Your Correct Answer"  # Set the correct answer

            if user_answer.lower() == correct_answer.lower():
                st.success("Correct! You earned points.")
                user_score += 10  # Assign points to the user
            else:
                st.error("Sorry, that's incorrect.")

    return positive_count, negative_count, neutral_count

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

# Display the user's score
st.write(f"Your Score: {user_score}")

# Run the app
if __name__ == "__main__":
    st.write("Welcome to the News Search and Sentiment Analysis app.")

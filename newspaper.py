import streamlit as st
import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd

# Set Streamlit page configuration
st.set_page_config(
    page_title="News Search and Sentiment Analysis",
    page_icon=":newspaper:",
    layout="wide"
)

nltk.download("vader_lexicon")

# Initialize variables
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
        articles = data.get('articles', [])

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

def extract_topics(articles):
    content = [article.get('content', '') for article in articles if article.get('content')]
    
    if not content:
        st.warning("No content available for topic extraction.")
        return None

    tfidf = tfidf_vectorizer.fit_transform(content)
    lda.fit(tfidf)
    return lda




# Function to display articles and return statistics
def display_articles(articles):
    positive_count, negative_count, neutral_count = 0, 0, 0
    article_data = []

    for index, article in enumerate(articles):
        title = article.get('title', 'No title available')
        content = article.get('content', '')
        author = article.get("author", "")
        link = article.get("url", "")

        sentiment = get_sentiment_label(content)

        with st.expander(f"Article {index + 1} - {title}"):
            st.write(f"Title: {title}")
            st.write(f"Author: {author}")
            st.write(f"Link to News: {link}")
            st.write(f"Sentiment: {sentiment}")

            user_answer = st.text_input(f"Answer for Article {index + 1}", key=f"answer_{index}")
            correct_answer = "Yes"

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
            "Sentiment": sentiment,
        })

    return positive_count, negative_count, neutral_count, article_data

def display_topics_and_analytics(articles, article_data):
    st.subheader("Topics Tags")
    lda = extract_topics(articles)

    for topic_idx, topic in enumerate(lda.components_):
        st.write(f"Topic {topic_idx + 1}:")
        top_words = [tfidf_vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
        st.write(", ".join(top_words))

    st.subheader("Data Analytics")

    if article_data and "Sentiment" in article_data[0]:
        # Create a pandas DataFrame from the article data
        df = pd.DataFrame(article_data)
        st.dataframe(df)

        st.subheader("Sentiment Distribution")
        sentiment_counts = df["Sentiment"].value_counts()
        st.bar_chart(sentiment_counts)
    else:
        st.info("No sentiment data available for analytics.")


# Main function
def main():
    # Load search history
    search_history = load_search_history()

    # Input for keyword search
    input_data = st.text_input("Enter a keyword to search for news")
    if input_data:
        search_history.append(input_data)

        try:
            with open("search_history.txt", mode="w") as file:
                file.write("\n".join(search_history))
        except Exception as e:
            st.error(f"An error occurred while saving the search history: {str(e)}")

        articles = search_news(input_data)
        article_data = display_articles(articles)

        st.info(f"Found {len(articles)} articles")

        display_topics_and_analytics(articles, article_data)

        # Display the user's score
        st.write(f"Your Score: {user_score}")

# Run the app
if __name__ == "__main__":
    main()
    st.write("Welcome to the News Search and Sentiment Analysis app.")

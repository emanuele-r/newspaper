import streamlit as st
import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd

# Set Streamlit page configuration
st.set_page_config(
    page_title="Personalized News Dashboard",
    page_icon=":newspaper:",
    layout="wide"
)

nltk.download("vader_lexicon")

# Initialize variables
search_history = []
user_score = 0
sia = SentimentIntensityAnalyzer()
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
lda = None  # Initialize LDA model
bookmarks = {}  # Create a dictionary to store bookmarks

# Function to load search history
def load_search_history():
    try:
        with open("search_history.txt", mode="r") as file:
            return file.read().splitlines()[-5:]  # Limit to the last 5 entries
    except FileNotFoundError:
        return []

# Function to search for news articles
def search_news(keyword):
    api_key = "Y"  # Repla89de75b718bb45ba884f256d3b1710cc" with your News API key
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
    global tfidf_vectorizer, lda  # Declare them as global variables
    content = [article.get('content', '') for article in articles]
    
    # Filter out very short or empty documents
    content = [doc for doc in content if len(doc) > 10]  # You can adjust the minimum length as needed
    
    if not content:
        st.warning("No valid content to extract topics from.")
        return
    
    tfidf = tfidf_vectorizer.fit_transform(content)

    # Initialize the LDA model
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(tfidf)


def display_articles(articles):
    global user_score  # Declare user_score as a global variable

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

    if lda is not None:
        term_topic_matrix = lda.transform(tfidf_vectorizer.transform([article["content"] for article in articles]))
        num_top_words = 10

        for topic_idx in range(lda.n_components):
            st.write(f"Topic {topic_idx + 1}:")
            topic = term_topic_matrix[:, topic_idx]
            top_word_indices = topic.argsort()[-num_top_words:][::-1]
            top_words = [tfidf_vectorizer.get_feature_names_out()[i] for i in top_word_indices]
            st.write(", ".join(top_words))
    else:
        st.warning("Topic extraction is not available because 'lda' is not initialized.")

    st.subheader("Data Analytics")

    if isinstance(article_data, list) and all("Sentiment" in article for article in article_data):
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
    global bookmarks, tfidf_vectorizer, lda 

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
        extract_topics(articles)  # Call the topic extraction function
        article_data = display_articles(articles)

        st.info(f"Found {len(articles)} articles")

        display_topics_and_analytics(articles, article_data)
        # Display the user's score
        st.write(f"Your Score: {user_score}")

    # Bookmarks
    st.sidebar.header("Bookmarks")
    if st.sidebar.button("Add Bookmark"):
        bookmark_name = st.sidebar.text_input("Bookmark Name")
        bookmarks[bookmark_name] = articles

    selected_bookmark = st.sidebar.selectbox("Select Bookmark", list(bookmarks.keys()))

    if selected_bookmark:
        st.subheader(selected_bookmark)
        display_articles(bookmarks[selected_bookmark])
        
# Run the app
if __name__ == "__main__":
    main()
    st.write("Welcome to the Personalized News Dashboard app.")

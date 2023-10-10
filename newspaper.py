import streamlit as st
import requests
import plotly.express as px
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Set Streamlit page configuration
st.set_page_config(page_title="Keyword news search", page_icon=":newspaper:")

st.title("Search Keyword News")

# Load existing search history from a text file
def load_search_history():
    try:
        with open("search_history.txt", mode="r") as file:
            return file.read().splitlines()
    except FileNotFoundError:
        return []

search_history = load_search_history()


# def api_keys():
#     with open("api.txt", mode="r") as f:
#         api = f.readline().strip()  # Remove leading/trailing whitespace
#     return api

def search_news(keyword: str):
    api_key = api_keys()  # Load the NewsAPI key securely

    response = requests.get(f"https://newsapi.org/v2/everything?q={keyword}&apiKey=89de75b718bb45ba884f256d3b1710cc")

    articles = []

    if response.status_code == 200:
        data = response.json()

        if 'articles' in data:
            articles = data['articles']

    return articles

# Initialize the VADER sentiment analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

input_data = st.text_input("Enter keywords")



if input_data:
    # Add the current search query to the search history
    search_history.append(input_data)

    # Save the updated search history to the text file
    with open("search_history.txt", mode="w") as file:
        file.write("\n".join(search_history))

    # Fetch news articles
    articles = search_news(input_data)

    # Display the number of articles found
    st.info(f"Found {len(articles)} articles")

    # Display each article with title, author, link, extracted topics, and sentiment analysis
    for index, article in enumerate(articles):
        with st.expander(f"Article {index + 1} - {article.get('title', 'No title available')}"):
            title = article.get('title', 'No title available')
            author = article.get('author', 'No author available')
            url = article.get('url', '#')
            content = article.get('content', '')
            
            # Extract topics from the article content
            topics = extract_topics(content)
            
            # Perform sentiment analysis
            sentiment_scores = sia.polarity_scores(content)
            sentiment = "Positive" if sentiment_scores['compound'] > 0 else "Negative" if sentiment_scores['compound'] < 0 else "Neutral"
            
            st.write(f"Title: {title}")
            st.write(f"Author: {author}")
            st.write(f"Link: [Read More]({url})")
            st.write(f"Sentiment: {sentiment}")

# Data Visualization
if input_data and articles:
    # Create a bar chart to visualize the number of articles per source
    sources = [article.get('source', {}).get('name', 'Unknown') for article in articles]
    unique_sources = list(set(sources))  # Get unique source names
    source_counts = [sources.count(source) for source in unique_sources]

    # Create an interactive bar chart using Plotly
    fig = px.bar(x=unique_sources, y=source_counts, labels={'x': 'Source', 'y': 'Number of Articles'},
                 title='Number of Articles per Source', text=source_counts)
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

# Display the search history
if search_history:
    st.subheader("Search History")
    for query in search_history:
        st.write(query)

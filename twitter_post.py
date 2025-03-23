import random
import datetime
import sqlite3
import os
import tweepy
import json
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import logging
import time
import sys
import schedule
from functools import wraps

# Configure system encoding for Unicode support
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("twitter_bot.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load Twitter API credentials from .env
load_dotenv()

# Database setup
DB_FILE = "tweets.db"

def retry_with_backoff(initial_delay=1, max_delay=60, max_retries=5):
    """Decorator for API calls that implements exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.RequestException as e:
                    retries += 1
                    if retries >= max_retries:
                        logger.error(f"Max retries reached for {func.__name__}: {e}")
                        raise
                    sleep_time = min(delay * (2 ** (retries - 1)), max_delay)
                    logger.warning(f"Request failed, retrying in {sleep_time}s: {e}")
                    time.sleep(sleep_time)
        return wrapper
    return decorator

def setup_database():
    """Create a database table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create the table with all required columns including twitter_id
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tweets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            keyword TEXT,
            article_title TEXT,
            article_link TEXT,
            article_content TEXT,
            key_points TEXT,
            tweet TEXT,
            validated INTEGER DEFAULT 0,
            posted INTEGER DEFAULT 0,
            twitter_id TEXT
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Database setup complete with updated schema")

# Keywords for trending topics with weights
KEYWORDS = {
    "AI": 5,
    "Tech": 4,
    "Space Exploration": 3,
    "Climate Change": 3,
    "Finance": 2
}

@retry_with_backoff()
def get_hackernews(keyword):
    """Search HackerNews for articles related to a keyword with retry logic."""
    # Using the Algolia API that powers HackerNews search
    search_url = f"https://hn.algolia.com/api/v1/search?query={keyword}&tags=story"
    logger.info(f"Searching HackerNews: {search_url}")

    # Add timeout to prevent hanging
    response = requests.get(search_url, timeout=10)
    
    if response.status_code != 200:
        logger.error(f"Error: Status code {response.status_code}")
        return None, None, None
        
    data = response.json()
    hits = data.get("hits", [])
    
    logger.info(f"Found {len(hits)} HackerNews stories.")
    
    if hits:
        # Filter for stories with URLs and points > 10 for quality
        stories_with_urls = [hit for hit in hits if hit.get("url") and hit.get("points", 0) > 10]
        
        if stories_with_urls:
            # Pick a random story with higher points weighted more heavily
            stories_with_urls.sort(key=lambda x: x.get("points", 0), reverse=True)
            top_stories = stories_with_urls[:min(5, len(stories_with_urls))]
            story = random.choice(top_stories)
            
            article_title = story["title"]
            article_link = story["url"]
            
            # Get the HackerNews discussion URL
            hn_link = f"https://news.ycombinator.com/item?id={story['objectID']}"
            
            logger.info(f"Found article: {article_title}")
            logger.info(f"Source URL: {article_link}")
            logger.info(f"HackerNews discussion: {hn_link}")
            
            # Extract article content - Return None for now, will be populated by separate function
            return article_title, article_link, None
        else:
            logger.warning("No stories with URLs and sufficient points found.")
    
    logger.warning("No suitable HackerNews stories found.")
    return None, None, None

@retry_with_backoff()
def extract_article_content(article_link):
    """Extract the main content of an article from the URL."""
    try:
        logger.info(f"Extracting content from: {article_link}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(article_link, headers=headers, timeout=15)
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch article: Status code {response.status_code}")
            return None
            
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script_or_style in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
            script_or_style.decompose()
            
        # Try to find main content
        # Look for common content container elements
        main_content = None
        for selector in ['article', 'main', '.content', '#content', '.post', '.article-content', '.post-content']:
            if soup.select(selector):
                main_content = ' '.join([p.get_text() for p in soup.select(f"{selector} p")])
                break
                
        # If no specific content container found, use all paragraphs
        if not main_content:
            paragraphs = [p.get_text().strip() for p in soup.find_all('p') if len(p.get_text().strip()) > 100]
            main_content = ' '.join(paragraphs)
        
        # Clean up the content
        main_content = ' '.join(main_content.split())
        
        if main_content:
            logger.info(f"Successfully extracted content ({len(main_content)} chars)")
            return main_content
        else:
            logger.warning("Failed to extract meaningful content")
            return None
            
    except Exception as e:
        logger.error(f"Error extracting article content: {e}")
        return None

def extract_key_points_with_vectors(article_title, article_content, keyword, num_points=5):
    """Extract key points from article using vector search."""
    try:
        if not article_content or len(article_content) < 200:
            logger.warning("Article content too short or empty for vector search")
            return None
            
        logger.info(f"Starting vector search to extract key points from article")
        
        # Initialize embeddings
        embeddings = OllamaEmbeddings(model="llama3.2:latest")
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_text(article_content)
        
        logger.info(f"Split article into {len(chunks)} chunks")
        
        # Create vector store
        vectorstore = FAISS.from_texts(chunks, embeddings)
        
        # Create search queries based on title and keyword
        queries = [
            f"What is the main point of an article titled '{article_title}'?",
            f"What are the key facts about {keyword} in this article?",
            f"What are the most interesting insights about {keyword}?",
            f"What statistics or numbers are mentioned about {keyword}?",
            f"What conclusions or future implications does the article discuss about {keyword}?"
        ]
        
        all_results = []
        for query in queries:
            results = vectorstore.similarity_search(query, k=2)
            all_results.extend([doc.page_content for doc in results])
            
        # Remove duplicates while preserving order
        unique_results = []
        for result in all_results:
            if result not in unique_results:
                unique_results.append(result)
                
        # Limit to num_points most relevant chunks
        key_points = unique_results[:num_points]
        
        logger.info(f"Extracted {len(key_points)} key points using vector search")
        
        # Join key points with separators for readability
        formatted_key_points = "\n\n* ".join([""] + key_points)
        
        return formatted_key_points
        
    except Exception as e:
        logger.error(f"Error extracting key points with vectors: {e}")
        return None
    
def generate_tweet(article_title, article_link, key_points):
    """Generate a human-like tweet using LangChain with extracted key points."""
    try:
        llm = OllamaLLM(model="llama3.2:latest", temperature=0.7, top_p=0.9)

        # Include key points in the prompt if available
        if key_points:
            prompt_template = """
            Generate a tweet (max 260 characters) about this article that sounds natural and conversational, like a human wrote it:
            
            Title: "{article_title}"
            Key Points: {key_points}
            
            The tweet should:
            1. Sound casual and natural, not formal or robotic
            2. Focus on ONE specific insight that would interest people
            3. Include your personal reaction (like "Wow!" or "This is fascinating")
            4. Use contractions (like "I'm" instead of "I am")
            5. Maybe include a brief thought-provoking question
            6. End with the article link
            7. Include 1-2 relevant hashtags that people actually use
            8. Add appropriate emojis for engagement and emotion
            
            Keep it under 260 characters total (not counting the link) to leave room for the URL.
            Just return the tweet text without any explanation.
            """
        else:
            # Fallback to title-only approach if no key points
            prompt_template = """
            Generate a tweet (max 260 characters) about this article that sounds like a real person wrote it:
            
            Title: "{article_title}"
            
            Make it:
            1. Sound casual and conversational, like you're talking to friends
            2. Include a natural reaction (like "Fascinating!" or "Just read this...")
            3. Use contractions and occasional informal language
            4. Maybe pose a question that makes people curious
            5. End with the article link
            6. Include 1-2 relevant hashtags that real people use
            7. Add appropriate emojis for engagement and emotion
            
            Keep it under 260 characters total (not counting the link) to leave room for the URL.
            Just return the tweet text without any explanation.
            """

        prompt = PromptTemplate(
            input_variables=["article_title", "article_link", "key_points"] if key_points else ["article_title", "article_link"],
            template=prompt_template
        )

        formatted_prompt = prompt.format(
            article_title=article_title, 
            article_link=article_link,
            key_points=key_points if key_points else ""
        )
        
        tweet = llm.invoke(formatted_prompt)
        
        # Ensure tweet doesn't exceed 280 characters
        if len(tweet) > 280:
            logger.warning(f"Tweet too long ({len(tweet)} chars), will handle in validation")
            
        # Make sure the link is included
        if article_link not in tweet:
            tweet = tweet.strip() + " " + article_link
            
        return tweet.strip() if tweet else None
        
    except Exception as e:
        logger.error(f"Error generating tweet: {e}")
        return None

def validate_tweet(article_title, article_link, key_points, tweet):
    """Ensure the tweet is accurate, human-like, and within character limit."""
    try:
        llm = OllamaLLM(model="llama3.2:latest", temperature=0.5, top_p=0.8)
        
        # Start with the original tweet
        validated_tweet = tweet
        
        # Check if tweet is within limit
        if len(validated_tweet) <= 280:
            return validated_tweet
        
        # Tweet is too long, need to fix it
        logger.warning(f"Tweet too long ({len(validated_tweet)} chars), trying emergency shortening")
        
        # Emergency shortening as a fail-safe
        emergency_prompt = PromptTemplate(
            input_variables=["article_title", "tweet_link"],
            template="""
            Emergency tweet shortening needed! Create a very short tweet (under 200 chars) about:
            
            "{article_title}"
            
            Include this link at the end: {tweet_link}
            
            Make it extremely concise but still human and engaging.
            Just return the shortened tweet without explanation.
            """
        )
        
        # Extract link from original tweet or use article_link
        if "http" in validated_tweet:
            link_parts = validated_tweet.split("http")
            tweet_link = "http" + link_parts[-1].strip()
        else:
            tweet_link = article_link
            
        emergency_formatted = emergency_prompt.format(
            article_title=article_title,
            tweet_link=tweet_link
        )
        
        validated_tweet = llm.invoke(emergency_formatted)
        
        # Final check and manual truncation as last resort
        if len(validated_tweet) > 280:
            logger.error(f"Emergency shortening failed, manual truncation required")
            
            # Find a sentence break if possible
            sentences = validated_tweet.split('.')
            shortened = ''
            for s in sentences:
                if len(shortened + s + '.') < 240:
                    shortened += s + '.'
                else:
                    break
            
            # Add link back
            if shortened:
                validated_tweet = shortened.strip() + " " + tweet_link
            else:
                # Last resort - hard truncation
                validated_tweet = validated_tweet[:240].strip() + "... " + tweet_link
        
        # Ensure the link is preserved
        if article_link not in validated_tweet and "http" not in validated_tweet:
            logger.warning("Link missing from validated tweet, adding it back")
            validated_tweet = validated_tweet.strip() + " " + article_link
            
        logger.info(f"Final tweet length: {len(validated_tweet)} characters")
        return validated_tweet.strip() if validated_tweet else None
        
    except Exception as e:
        logger.error(f"Error validating tweet: {e}")
        # Return a shortened version of original tweet as failsafe
        if len(tweet) > 280:
            cutoff = tweet.rfind(' ', 0, 240)
            if cutoff == -1:
                cutoff = 240
            return tweet[:cutoff].strip() + "... " + article_link
        return tweet  # Return original tweet if validation fails

def save_tweet(keyword, article_title, article_link, article_content, key_points, tweet, validated):
    """Save the tweet to the database."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        # Insert with the new schema including key_points
        cursor.execute(
            "INSERT INTO tweets (timestamp, keyword, article_title, article_link, article_content, key_points, tweet, validated) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", 
            (timestamp, keyword, article_title, article_link, article_content, key_points, tweet, validated)
        )
        
        conn.commit()
        tweet_id = cursor.lastrowid
        
        # Safe logging to handle any potential Unicode issues
        try:
            logger.info(f"Tweet saved to database with ID {tweet_id}: {tweet}")
        except UnicodeEncodeError:
            ascii_tweet = tweet.encode('ascii', 'replace').decode('ascii')
            logger.info(f"Tweet saved to database with ID {tweet_id}: {ascii_tweet}")
            
        return tweet_id
        
    except Exception as e:
        logger.error(f"Error saving tweet: {e}")
        conn.rollback()
        return None
        
    finally:
        conn.close()

def post_to_twitter(tweet_id):
    """Post a tweet to Twitter using the Twitter API v2 endpoints."""
    try:
        # Get the tweet from the database
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT tweet FROM tweets WHERE id = ?", (tweet_id,))
        result = cursor.fetchone()
        
        if not result:
            logger.error(f"Tweet with ID {tweet_id} not found in database")
            conn.close()
            return False
            
        tweet_text = result[0]
        
        # Use Twitter API v2 with tweepy Client instead of API
        client = tweepy.Client(
            consumer_key=os.getenv("API_KEY"),
            consumer_secret=os.getenv("API_SECRET"),
            access_token=os.getenv("ACCESS_TOKEN"),
            access_token_secret=os.getenv("ACCESS_SECRET")
        )
        
        # Post tweet using v2 endpoint
        response = client.create_tweet(text=tweet_text)
        twitter_id = response.data['id']
        
        # Get Twitter username from environment variable or use a default
        twitter_username = os.getenv("TWITTER_USERNAME", "user")
        
        # Generate the full Twitter URL for the tweet
        twitter_url = f"https://twitter.com/{twitter_username}/status/{twitter_id}"
        logger.info(f"Tweet posted successfully! ID: {tweet_id}, Twitter URL: {twitter_url}")

        # Mark the tweet as posted in the database and save the Twitter ID
        cursor.execute("UPDATE tweets SET posted = 1, twitter_id = ? WHERE id = ?", (twitter_id, tweet_id))
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error posting to Twitter: {e}")
        return False
    
def select_keyword():
    """Select a keyword based on weighted probabilities."""
    items = list(KEYWORDS.items())
    keywords, weights = zip(*items)
    return random.choices(keywords, weights=weights, k=1)[0]

def tweet_job():
    """Core function to be run every 30 minutes."""
    try:
        logger.info("Starting scheduled tweet job")
        
        # Select a keyword based on weighted probabilities
        keyword = select_keyword()
        logger.info(f"Selected keyword: {keyword}")
        
        # Get article from HackerNews
        article_title, article_link, _ = get_hackernews(keyword)
        
        if not article_title or not article_link:
            logger.warning("No suitable article found, exiting")
            return
            
        # Extract article content
        article_content = extract_article_content(article_link)
        
        # Extract key points using vector search
        key_points = None
        if article_content:
            key_points = extract_key_points_with_vectors(article_title, article_content, keyword)
            if key_points:
                logger.info("Successfully extracted key points using vector search")
            else:
                logger.warning("Failed to extract key points, will generate tweet based on title only")
        else:
            logger.warning("Could not extract article content, will generate tweet based on title only")
            
        # Generate tweet
        tweet = generate_tweet(article_title, article_link, key_points)
        
        if not tweet:
            logger.warning("Failed to generate tweet, exiting")
            return
            
        # Validate tweet
        validated_tweet = validate_tweet(article_title, article_link, key_points, tweet)
        
        if validated_tweet:
            # Safe logging to handle any potential Unicode issues
            logger.info(f"Generated tweet: {tweet}")
            logger.info(f"Validated tweet: {validated_tweet}")
            
            # Save tweet to database
            tweet_id = save_tweet(keyword, article_title, article_link, article_content, key_points, validated_tweet, validated=1)
            
            # Post to Twitter
            if tweet_id:
                post_to_twitter(tweet_id)
            
        else:
            logger.warning("Failed to validate tweet, saving original")
            save_tweet(keyword, article_title, article_link, article_content, key_points, tweet, validated=0)
            
    except Exception as e:
        logger.error(f"Error in tweet job: {e}")
        import traceback
        logger.error(traceback.format_exc())

def run_schedule():
    """Run the scheduler continuously."""
    # Schedule the job to run every 30 minutes
    schedule.every(30).minutes.do(tweet_job)
    
    logger.info("Scheduler started - will post tweets every 30 minutes")
    
    # Run the job once immediately
    tweet_job()
    
    # Keep running the scheduler
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    # Setup database
    setup_database()
    
    # Run the scheduler
    run_schedule()
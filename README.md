# AI-Powered Twitter Bot

A sophisticated Twitter bot that automatically discovers trending tech articles from HackerNews, generates insightful tweets, and posts them on a regular schedule.

## Features

- Automatically discovers relevant articles from HackerNews based on weighted keywords
- Extracts and analyzes article content using vector search and embeddings
- Generates natural, human-like tweets with proper hashtags and emoji
- Posts tweets on a scheduled basis (every 30 minutes)
- Implements robust error handling and retry mechanisms
- Maintains a database of all generated and posted tweets

## Requirements

- Python 3.8+
- Twitter API credentials (v2)
- Ollama running locally with llama3.2 model loaded

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ai-twitter-bot.git
cd ai-twitter-bot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup Ollama

Ensure Ollama is installed and running with the llama3.2 model.

```bash
# Install Ollama (if not already installed)
# See: https://ollama.ai/download

# Run the server
ollama serve

# In another terminal, pull the llama3.2 model
ollama pull llama3.2:latest
```

### 4. Create Environment File

Create a `.env` file in the project root with the following:

```
API_KEY=your_twitter_api_key
API_SECRET=your_twitter_api_secret
ACCESS_TOKEN=your_twitter_access_token
ACCESS_SECRET=your_twitter_access_token_secret
TWITTER_USERNAME=your_twitter_username
```

### 5. Run the Bot

```bash
python twitter_bot.py
```

The bot will immediately generate and post the first tweet, then continue posting every 30 minutes.

## Architecture

### Core Components

1. **Data Collection**
   - Fetches articles from HackerNews API
   - Extracts full article content using BeautifulSoup

2. **Content Analysis**
   - Splits article content into chunks
   - Uses vector embeddings to identify key points
   - Performs semantic search based on queries

3. **Tweet Generation**
   - Generates human-like tweets using LLama 3.2
   - Includes emotional reactions, questions, and appropriate hashtags
   - Ensures tweets stay within character limits

4. **Scheduling & Posting**
   - Uses the `schedule` library to run every 30 minutes
   - Posts to Twitter using Tweepy's v2 API client
   - Logs all activities and maintains a database of posts

### Database Schema

The bot maintains an SQLite database (`tweets.db`) with the following schema:

| Column | Description |
|--------|-------------|
| id | Primary key |
| timestamp | When the tweet was generated |
| keyword | The keyword used to find the article |
| article_title | Title of the article |
| article_link | URL to the article |
| article_content | Extracted text content |
| key_points | Key points extracted from the article |
| tweet | The generated tweet text |
| validated | Whether the tweet passed validation (1/0) |
| posted | Whether the tweet was posted (1/0) |
| twitter_id | Twitter's ID for the posted tweet |

## Configuration Options

### Modifying Keywords

Edit the `KEYWORDS` dictionary in the script to change the topics the bot focuses on:

```python
KEYWORDS = {
    "AI": 5,        # Higher weight = more likely to be selected
    "Tech": 4,
    "Space Exploration": 3,
    "Climate Change": 3,
    "Finance": 2
}
```

### Changing the Schedule

To change how often tweets are posted, modify the scheduling line:

```python
# For hourly posts
schedule.every(60).minutes.do(tweet_job)

# For daily posts at specific time
schedule.every().day.at("10:00").do(tweet_job)
```

## Troubleshooting

### Common Issues

1. **API Rate Limiting**: If you encounter Twitter API rate limits, the bot will automatically retry with exponential backoff.

2. **Article Extraction Failures**: Some websites use techniques that make content extraction difficult. The bot will fall back to title-only tweet generation in these cases.

3. **Model Availability**: Ensure Ollama is running and the llama3.2 model is properly loaded.

### Log Files

Logs are written to `twitter_bot.log` and also output to the console. Check these logs to diagnose any issues.

## License

MIT License

## Contribution

Contributions are welcome! Please feel free to submit a Pull Request.
# import time
# import pandas as pd
# import tweepy  
# import os
# from src.predictor import SentimentPredictor  

# BEARER_TOKEN = os.getenv("BEARER_TOKEN")
# client = tweepy.Client(bearer_token=BEARER_TOKEN)

# def research_tweet(keyword="Twitter", max_tweets=100):
#     """ Raccoglie tweet in tempo reale con un determinato keyword """
#     tweets = client.search_recent_tweets(query=keyword, max_results=max_tweets)
#     return [tweet.text for tweet in tweets.data] if tweets.data else []

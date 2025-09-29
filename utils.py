import requests
import pandas as pd

from pyspark.sql import SparkSession

from config import NEWS_API_KEY

def fetch_news():
    url = f"https://newsapi.org/v2/top-headlines?language=en&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    data = response.json()
    articles = data.get('articles', [])
    texts = [(article['title'] + " " + (article.get('description') or "")) for article in articles]
    return texts

def spark_session(app_name="NewsSentimentApp"):
    return SparkSession.builder.appName(app_name).getOrCreate()

def load_training_data(spark, path):
    return spark.read.option("header", True).csv(path)

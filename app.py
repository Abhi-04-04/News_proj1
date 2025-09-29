
import streamlit as st
from predict import predictions
import pandas as pd

# Load training data
train_data_path = 'training_data/labeled_headlines.csv'
train_df = pd.read_csv(train_data_path)

# Display the first few rows of the dataset
train_df.head()

st.title("Real-Time News Sentiment Analysis")

# Convert Spark DataFrame to Pandas for display
pandas_df = predictions.select("text", "predicted_label").toPandas()

st.dataframe(pandas_df)

# Download CSV
st.download_button(
    label="Download Predictions",
    data=pandas_df.to_csv(index=False),
    file_name="news_sentiment_results.csv",
    mime="text/csv"
)


import streamlit as st
from predict import predictions
import pandas as pd

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

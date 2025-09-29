from pyspark.ml.feature import IndexToString
from pyspark.sql import Row
from utils import spark_session, fetch_news
from config import MODEL_PATH

spark = spark_session("NewsSentimentPrediction")
model = Pipeline.load(MODEL_PATH)

# Fetch news
texts = fetch_news()
df_news = spark.createDataFrame([(text,) for text in texts], ["text"])

# Predict
predictions = model.transform(df_news)

# Convert numeric prediction back to string label
label_converter = IndexToString(
    inputCol="prediction",
    outputCol="predicted_label",
    labels=model.stages[3].labels
)
predictions = label_converter.transform(predictions)

predictions.show(truncate=False)

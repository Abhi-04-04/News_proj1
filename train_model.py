# train_model.py

from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
import os

# ----------------------------
# 1️⃣ Set up SparkSession
# ----------------------------
spark = SparkSession.builder \
    .appName("NewsSentimentML") \
    .getOrCreate()

# ----------------------------
# 2️⃣ Load training CSV
# ----------------------------
# Make sure your CSV has columns: 'text' and 'label'
train_path = "C:/Users/abhir/Downloads/labeled_headlines.csv"

if not os.path.exists(train_path):
    raise FileNotFoundError(f"Training CSV not found: {train_path}")

df_train = spark.read.option("header", True).csv(train_path)
df_train.show(5)

# ----------------------------
# 3️⃣ Define ML pipeline
# ----------------------------
tokenizer = Tokenizer(inputCol="text", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
vectorizer = CountVectorizer(inputCol="filtered", outputCol="features")
label_indexer = StringIndexer(inputCol="label", outputCol="labelIndex")
lr = LogisticRegression(featuresCol="features", labelCol="labelIndex")

pipeline = Pipeline(stages=[tokenizer, remover, vectorizer, label_indexer, lr])

# ----------------------------
# 4️⃣ Train the model
# ----------------------------
model = pipeline.fit(df_train)
print("✅ Model training completed!")

# ----------------------------
# 5️⃣ Save the trained model
# ----------------------------
model_path = "news_sentiment_model"
model.write().overwrite().save(model_path)
print(f"✅ Trained model saved at: {model_path}")

# Stop SparkSession
spark.stop()


from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import IndexToString
from utils import spark_session, load_training_data
from config import TRAINING_DATA_PATH, MODEL_PATH

spark = spark_session("NewsSentimentTraining")
df_train = load_training_data(spark, TRAINING_DATA_PATH)

# Build pipeline
tokenizer = Tokenizer(inputCol="text", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
vectorizer = CountVectorizer(inputCol="filtered", outputCol="features")
label_indexer = StringIndexer(inputCol="label", outputCol="labelIndex")
lr = LogisticRegression(featuresCol="features", labelCol="labelIndex")

pipeline = Pipeline(stages=[tokenizer, remover, vectorizer, label_indexer, lr])

# Train and save model
model = pipeline.fit(df_train)
model.write().overwrite().save(MODEL_PATH)
print("Model trained and saved at:", MODEL_PATH)

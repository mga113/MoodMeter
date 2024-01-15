import sys
from pyspark.sql import SparkSession, functions, types

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download(["vader_lexicon", "punkt"])

spark = SparkSession.builder.appName('reddit').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
assert spark.version >= '3.2' # make sure we have Spark 3.2+


comments_schema = types.StructType([
        types.StructField('archived', types.BooleanType()),
    types.StructField('author', types.StringType()),
    types.StructField('author_flair_css_class', types.StringType()),
    types.StructField('author_flair_text', types.StringType()),
    types.StructField('created', types.LongType()),
    types.StructField('created_utc', types.StringType()),
    types.StructField('distinguished', types.StringType()),
    types.StructField('domain', types.StringType()),
    types.StructField('downs', types.LongType()),
    types.StructField('edited', types.BooleanType()),
    types.StructField('from', types.StringType()),
    types.StructField('from_id', types.StringType()),
    types.StructField('from_kind', types.StringType()),
    types.StructField('gilded', types.LongType()),
    types.StructField('hide_score', types.BooleanType()),
    types.StructField('id', types.StringType()),
    types.StructField('is_self', types.BooleanType()),
    types.StructField('link_flair_css_class', types.StringType()),
    types.StructField('link_flair_text', types.StringType()),
    types.StructField('media', types.StringType()),
    types.StructField('name', types.StringType()),
    types.StructField('num_comments', types.LongType()),
    types.StructField('over_18', types.BooleanType()),
    types.StructField('permalink', types.StringType()),
    types.StructField('quarantine', types.BooleanType()),
    types.StructField('retrieved_on', types.LongType()),
    types.StructField('saved', types.BooleanType()),
    types.StructField('score', types.LongType()),
    types.StructField('secure_media', types.StringType()),
    types.StructField('selftext', types.StringType()),
    types.StructField('stickied', types.BooleanType()),
    types.StructField('subreddit', types.StringType()),
    types.StructField('subreddit_id', types.StringType()),
    types.StructField('thumbnail', types.StringType()),
    types.StructField('title', types.StringType()),
    types.StructField('ups', types.LongType()),
    types.StructField('url', types.StringType()),
    types.StructField('year', types.IntegerType()),
    types.StructField('month', types.IntegerType()),
])

#to compute sentiment scores for each submissions
def avg_vader_scores(comment):
	sia = SentimentIntensityAnalyzer()
	sentences = nltk.sent_tokenize(comment)
	
	compound_scores = [sia.polarity_scores(sentence).get("compound", 0.0) for sentence in sentences]
	if len(compound_scores) > 0:
		avg_score = sum(compound_scores)/len(compound_scores)
		return avg_score
	else:
		return 0.0
	
#to classify a submission as positive, negative or neutral	
def sent_word(score):
	if (score > 0.05):
		return "positive"
	if (score <-0.05):
		return "negative"	
	else:
		return "neutral"


def main(in_directory, out_directory):
    comments = spark.read.json(in_directory, schema=comments_schema)
    comments = comments.filter(comments['selftext'] != "[deleted]")
    comments = comments.filter(comments['selftext'] != "[removed]")
    comments = comments.filter(comments['selftext'] != "")
    comments = comments.select("selftext", "subreddit", "created_utc", "score", "month")
    
    print(comments.count())
     
    vader_udf = functions.udf(avg_vader_scores, returnType=types.FloatType())
    sent_udf = functions.udf(sent_word, returnType=types.StringType())
    stats_comments = comments
    stats_comments = stats_comments.withColumn("sentiment_score", vader_udf(comments["selftext"]))
    stats_comments = stats_comments.withColumn("sentiment", sent_udf(stats_comments["sentiment_score"]))
   
    stats_comments.show(5)
    
    stats_comments = stats_comments.select("subreddit", "created_utc", "score", "sentiment_score", "sentiment", "month")
    etl_data = stats_comments.coalesce(1) #under 4500 records
    etl_data.write.json('stats_data-L', compression='gzip', mode="overwrite")
    
    



if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)

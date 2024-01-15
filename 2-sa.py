import pandas as pd
import gzip, glob
import sys
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download(["vader_lexicon", "punkt"])
nltk.download('stopwords')


def avg_vader_scores(comment):
	sia = SentimentIntensityAnalyzer()
	sentences = nltk.sent_tokenize(comment)
	
	compound_scores = [sia.polarity_scores(sentence).get("compound", 0.0) for sentence in sentences]
	if len(compound_scores) > 0:
		return sum(compound_scores)/len(compound_scores)
	else:
		return 0.0	
	
def sent_word(score):
	if (score > 0.05):
		return "positive"
	if (score <-0.05):
		return "negative"	
	else:
		return "neutral"

def filtered_comment_dataset(comments):
	stopwords = nltk.corpus.stopwords.words("english") # defines common stopwords
	
	all_comments_text = " ".join(comments.astype(str).values) # concatente and tokenize comments
	all_words =  nltk.word_tokenize(all_comments_text)
	words = [w for w in all_words if w.lower() not in stopwords and w.isalpha()] #remove stopwords and ensures is alpha numeric
	return words # returns filtered/cleaned comment dataset

def pos_neg_neutral_partition(filtered_comment_dataset):
	pos_words = [] # initialize arrays to store positive, negative, and neutral scoring words
	neg_words = [] 
	neutral_words = [] 
	for w in filtered_comment_dataset: # determine each word's score
		score = avg_vader_scores(w)
		if(sent_word(score)== "positive"):
			pos_words.append(w)
		elif(sent_word(score)== "negative"):
			neg_words.append(w)
		else:
			neutral_words.append(w)
	fd1 = nltk.FreqDist(pos_words) # initialize feature distribution and top 5 common words
	most_common_pos_words = fd1.most_common(5)

	fd2 = nltk.FreqDist(neg_words)
	most_common_neg_words = fd2.most_common(5)

	fd3 = nltk.FreqDist(neutral_words)
	most_common_neutral_words = fd3.most_common(5)
	# return common words for each array set 
	return (most_common_pos_words,most_common_neg_words,most_common_neutral_words)

def top_10_trigrams(comment_data):
	# top trigrams for all words
	finder = nltk.collocations.TrigramCollocationFinder.from_words(comment_data)
	top_trigrams = finder.ngram_fd.most_common(10)
	return top_trigrams

def top_10_pos_neg_neu_trigrams(comment_data):
	# top trigrams for positive, negative and neutral words
	pos_words = [] 
	neg_words = [] 
	neutral_words = [] 
	for w in comment_data: 
		score = avg_vader_scores(w)
		if(sent_word(score) == "positive"):
			pos_words.append(w)
		elif(sent_word(score) == "negative"):
			neg_words.append(w)
		else:
			neutral_words.append(w)
			
	finder2 = nltk.collocations.TrigramCollocationFinder.from_words(pos_words)
	top_pos_trigrams = finder2.ngram_fd.most_common(10)

	finder3 = nltk.collocations.TrigramCollocationFinder.from_words(neg_words)
	top_neg_trigrams = finder3.ngram_fd.most_common(10)

	finder4 = nltk.collocations.TrigramCollocationFinder.from_words(neutral_words)
	top_neutral_trigrams = finder4.ngram_fd.most_common(10)

	return top_pos_trigrams, top_neg_trigrams, top_neutral_trigrams
	

def main():
	output = sys.argv[1]
	etl_filename = glob.glob(output+'/part-*.json.gz')[0]
	etl_data = pd.read_json(etl_filename, lines=True)
	print(etl_data.head(10))
	etl_data["sentiment_score"] = etl_data["selftext"].apply(avg_vader_scores)
	etl_data["sentiment"] = etl_data["sentiment_score"].apply(sent_word)
	etl_data["sentiment_score"] = etl_data["sentiment_score"]+1
	print(etl_data[["id", "sentiment_score", "sentiment"]].head(10))
	
	
	comment_data = filtered_comment_dataset(etl_data["selftext"]) 
	partioned = ((pos_neg_neutral_partition(comment_data)))
	print("\n The top 5 most common POSTIVE words are:", partioned[0], "\n")
	print("The top 5 most common NEGATIVE words are:", partioned[1], "\n")
	print("The top 5 most common NEUTRAL words are:", partioned[2], "\n")


	t10trigrams = top_10_trigrams(comment_data)
	print("Top 10 Trigrams:", t10trigrams)

	t10_pnn_trigrams = top_10_pos_neg_neu_trigrams(comment_data)
	print("\n Top 10 Positive Trigrams:", t10_pnn_trigrams[0], "\n")
	print("Top 10 Negative Trigrams::", t10_pnn_trigrams[1], "\n")
	print("Top 10 Neutral Trigrams: ", t10_pnn_trigrams[2], "\n")

	
	
   
main()   

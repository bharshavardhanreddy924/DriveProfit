from textblob import TextBlob

#CARRYING OUT SENTIMENTAL ANALYSIS ON REVIEWS
def review_analysis(user_review):
	sent = TextBlob(user_review).sentiment
	print(sent)
	polarity = sent[0]
	subjectivity = sent[1]

	return polarity, subjectivity
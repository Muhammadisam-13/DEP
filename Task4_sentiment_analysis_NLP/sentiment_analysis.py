import nltk
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import random
import string
import numpy as np

# This code creates a natural language processing model to predict the sentiment of a provided text
# First we load the data (nltk dataset of movie reviews)
# We then preprocess the data (convert to lower, stem, and remove punctuation, as these are not neccessary for NLP)
# We vectorize the data and represent positive as 1, else 0 for negative
# Finally we train the model with training values and test it using our testing values. The loop tests five samples, prints the text to be tested, and the sentiment
# below it

# nltk.download('movie_reviews') (only to run once)

# function to do all the preprocessing on the text as described earlier
def preprocess_text(text): # to convert text to lowercase, remove punctuation, stem
    stemmer = PorterStemmer()
    transformed = [] # the converted version of the text
    stopwords_set = set(stopwords.words('english'))

    for words, _ in text:
        words = [word.lower() for word in words]
        words = [word.translate(str.maketrans('', '', string.punctuation)) for word in words]
        words = [stemmer.stem(word) for word in words if word not in stopwords_set]
        transformed.append(' '.join(words)) # add them into corpus
    return transformed

text = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]



texts = [item[0] for item in text]
label = [item[1] for item in text] 
transformed = preprocess_text(text)

vectorizer = CountVectorizer() # to vectorize
X = vectorizer.fit_transform(transformed).toarray()
y = [1 if label == 'pos' else 0 for _, label in text] # convert to binary values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # training and testing data

clf = RandomForestClassifier(n_jobs=-1)
clf.fit(x_train, y_train) # train the model

test_samples = np.array([x_test[i] for i in range(5)]) # array of the first 5 test samples
predictions = clf.predict(test_samples) # prediction
inverse_vectorizer = vectorizer.inverse_transform(test_samples) # to convert the numerical values back to the word for

# Print the 5 test samples with the sentiment prediction
for i, (words, prediction) in enumerate(zip(inverse_vectorizer, predictions)):
    original_text = ' '.join(words)
    sentiment = "Positive " if prediction == 1 else "Negative"
    print(f"Test{i + 1}: {original_text}\nThis text is {sentiment}\n")
    
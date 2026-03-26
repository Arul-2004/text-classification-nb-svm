from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

# Sample data
texts = ["I love this", "I hate this", "This is great", "This is bad"]
labels = [1, 0, 1, 0]  # 1=positive, 0=negative

# Convert text to features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)
print("NB Prediction:", nb.predict(X_test))

# SVM
svm = SVC()
svm.fit(X_train, y_train)
print("SVM Prediction:", svm.predict(X_test))
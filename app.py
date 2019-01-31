from flask import Flask, render_template, url_for, request
import pandas as pd
# import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
# from sklearn.metrics import classification_report

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    df = pd.read_csv('./data/spam.csv', encoding='latin-1', usecols=['class', 'message'])
    df['label'] = df['class'].map({'ham': 0, 'spam':1})
    X = df['message']
    y = df['label']

    countVectzer = CountVectorizer()
    X = countVectzer.fit_transform(X)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Naive Bayes
    from sklearn.naive_bayes import MultinomialNB
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    nb.score(X_test, y_test)
    # y_pred = nb.predict(X_test)
    # print(classification_report(y_test, y_pred))
    # persist model  for future use to avoid retraining
    # joblib.dump(nb, 'NB_spam_model.pkl')

    #Alternative Usage of Saved Model
	# joblib.dump(clf, 'NB_spam_model.pkl')
	# NB_spam_model = open('NB_spam_model.pkl','rb')
	# clf = joblib.load(NB_spam_model)

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        messageVectzer = countVectzer.transform(data).toarray()
        new_prediction = nb.predict(messageVectzer)
    return render_template('result.html', prediction = new_prediction, message = message)


if __name__ == '__main__':
    app.run(debug =True)

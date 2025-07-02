import re
import pandas as pd
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess(text):
    tokens = word_tokenize(text)
    tokens = [re.sub(r'\W+', '', token) for token in tokens]
    tokens = [token.lower() for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if not any(word in token for word in ['quot', 'quoti', 'kompas', 'id'])]
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

data = pd.read_csv('datatrainPapua.csv')
data['sentiment'] = data['sentiment'].replace({'Positif ':'Positif', 'Negatif ':'Negatif', 'Netral ':'Netral'})
data.dropna(inplace = True)

data['processed_text'] = data['mentions'].apply(preprocess)
x = data['processed_text']
y = data['sentiment']
original_text = data['mentions']

x_train, x_test, y_train, y_test, original_train, original_test = train_test_split(x, y, original_text, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
x_train_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)

model = LogisticRegression()
model.fit(x_train_vectorized, y_train)

y_pred = model.predict(x_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(cm)
print('Classification Report:')
print(report)

joblib.dump(model, 'lmodel.pkl')
joblib.dump(vectorizer, 'lvectorizer.pkl')

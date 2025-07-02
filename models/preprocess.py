import re
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess(text):
    tokens = word_tokenize(text)
    tokens = [re.sub(r'\W+', '', token) for token in tokens]
    tokens = [token.lower() for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if not any(word in token for word in ['quot', 'quoti', 'kompas', 'id'])]
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

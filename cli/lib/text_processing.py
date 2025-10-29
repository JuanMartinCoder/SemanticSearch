
import string
from nltk.stem import PorterStemmer

from .search_utils import load_stop_words

stemmer = PorterStemmer()


def preprocess_text(text: str) -> str:
    return text.lower().translate(str.maketrans("", "", string.punctuation))

def tokenize(text: str) -> list[str]:
    # STEP 1: PREPROCESS TEXT
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
            if token:
                valid_tokens.append(token)

    # STEP 2: REMOVE STOP WORDS
    stop_words = load_stop_words()
    filtered_words = []
    for word in valid_tokens:        
        if word not in stop_words:
            filtered_words.append(word)    

    # STEP 3: STEM WORDS
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))


    return stemmed_words


    




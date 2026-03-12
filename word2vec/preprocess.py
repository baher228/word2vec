import re
import string

def preprocess(raw_data):
    text = raw_data.lower()  # lowercase
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # optional: normalize spaces
    return text

def map_words(text):
    word_to_id = {}
    id_to_word = {}
    encoded = []
    for word in text.split():
        if word not in word_to_id:
            index = len(word_to_id)
            word_to_id[word] = index
            id_to_word[index] = word
        encoded.append(word_to_id[word])
    return word_to_id, id_to_word, encoded
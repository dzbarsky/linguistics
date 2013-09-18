# David Zbarsky: dzbarsky@wharton.upenn.edu
# Yaou Wang: yaouwang@wharton.upenn.edu

from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import sent_tokenize, word_tokenize
import math

Corpus_root = 'corpus'
Mixed_root = 'mixed'
Starbucks_root = 'corpus/starbucks'
Heinz_root = 'corpus/heinz'

def get_sub_directories(directory):
    subdirs = set()
    for file in PlaintextCorpusReader(directory, '.*').fileids():
        index = file.find('/')
        if index > 0:
            subdirs.add(file[0:index])
    return list(subdirs)

def get_all_files(directory):
    return PlaintextCorpusReader(directory, '.*').fileids()

def load_file_sentences(filepath):
    index = filepath.rfind('/')
    dir = filepath[:index]
    filepath = filepath[index + 1:]
    return sent_tokenize(PlaintextCorpusReader(dir, filepath).raw().lower())

def load_collection_sentences(directory):
    sentences = []
    for file in get_all_files(directory):
        sentences.extend(load_file_sentences(directory + '/' + file))
    return sentences

def load_file_tokens(filepath):
    tokens = []
    for sentence in load_file_sentences(filepath):
        tokens.extend(word_tokenize(sentence))
    return tokens

def load_collection_tokens(directory):
    tokens = []
    for file in get_all_files(directory):
        tokens.extend(load_file_tokens(directory + '/' + file))
    return tokens

def get_tf(path):
    if path.find('/') >= 0:
        tokens = load_collection_tokens(path)
    else:
        tokens = load_file_tokens(path)

    count = 1.0/len(tokens)
    map = dict()
    for token in tokens:
       if token in map:
           map[token] += count
       else:
           map[token] = count
    return map

def get_idf(directory):
    docs = list()
    files = get_all_files(directory)
    for file in files:
        docs.append(load_file_tokens(directory + '/' + file))

    map = dict()
    for doc in docs:
        for token in set(doc):
            if token in map:
                map[token] += 1
            else:
                map[token] = 1.0

    for token in map.keys():
        map[token] = math.log(len(files)/map[token])

    return map

def get_tf_idf_words(dict1, dict2, k):
    tuples = []
    for term in dict1:
        tuples.append((term, dict1[term] * dict2[term]))

    tuples.sort(key=lambda x: x[1], reverse=True)

    terms = []
    for item in tuples[0:k]:
        terms.append(item[0])
    return terms

def get_mi(directory, w):
    dir_tokens = load_collection_tokens(directory)
    num = 0.0
    for token in dir_tokens:
        if token == w:
            num += 1
    num /= len(dir_tokens)
    denom = 0.0
    cor_tokens = load_collection_tokens('corpus')
    for token in cor_tokens:
        if token == w:
            denom += 1
    denom /= len(cor_tokens)
    return math.log((num/denom))
    
def get_mi_words(directory, k):
    pass

def create_feature_space(list):
    map = dict()
    counter = 0
    for sentence in list:
        for token in word_tokenize(sentence):
            if not token in map:
                map[token] = counter
                counter += 1
    return map

def vectorize(feature_space, str):
    vector = []
    for i in feature_space.keys():
        vector.append(0)

    for token in set(word_tokenize(str)):
        vector[feature_space[token]] = 1

    return vector

def cosine_similarity(X, Y):
    numerator = 0.0
    Xsum = 0
    Ysum = 0
    for i in range(len(X)):
        numerator += X[i] * Y[i]
        Xsum += X[i] * X[i]
        Ysum += Y[i] * Y[i]

    return numerator/(math.sqrt(Xsum) * math.sqrt(Ysum))

def main():
    #print get_sub_directories(Corpus_root)
    #print get_all_files(Corpus_root)
    #print load_file_sentences(Starbucks_root + '/118990300.txt')
    #print load_collection_sentences(Starbucks_root)
    #print load_file_tokens(Starbucks_root + '/118990300.txt')
    #print load_collection_tokens(Starbucks_root)
    #dict1 = get_tf(Heinz_root)
    #dict2 = get_idf(Corpus_root)
    #print get_tf_idf_words(dict1, dict2, 10)
    print get_mi(Starbucks_root, 'starbucks')
    sentences = ["this is a test", "this is another test"]
    print create_feature_space(sentences)
    feature_space = create_feature_space(sentences)
    print vectorize(feature_space, "another test")
    print cosine_similarity([1,1,0,1], [2,0,1,1])


if __name__ == "__main__":
    main()

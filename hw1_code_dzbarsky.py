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
    dir = filepath[0:index]
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
    map = dict()
    if path.find('/') >= 0:
        tokens = load_collection_tokens(path)
    else:
        tokens = load_file_tokens(path)
    for token in tokens:
       if token in map:
           map[token] += 1
       else:
           map[token] = 1
    for token in map.keys():
        pass
        map[token] /= float(len(tokens))
    return map

def get_idf(directory):
    docs = list()
    files = get_all_files(directory)
    for file in files:
        docs.append(load_file_tokens(directory + '/' + file))
    print "DONE LOADING"
    map = dict()
    for doc in docs:
        print doc
        for token in doc:
            if token in map:
                continue
            occurences = 0
            for doc2 in docs:
                if token in doc2:
                    occurences += 1
            map[token] = occurences

    for token in map.keys():
        map[token] = math.log(len(files)/map[token])

    return map

def get_tf_idf(dict1, dict2, k):
    tuples = []
    for term in dict1:
        tuples.append((term, dict1[term] * dict2[term]))

    tuples.sort(key=lambda x: x[1], reverse=True)

    terms = []
    for item in tuples[0:k]:
        terms.append(item[0])
    return terms

def main():
    #print get_sub_directories(Corpus_root)
    #print get_all_files(Starbucks_root)
    #print load_file_sentences(Starbucks_root + '/118990300.txt')
    #print load_collection_sentences(Starbucks_root)
    #print load_file_tokens(Starbucks_root + '/118990300.txt')
    #print load_collection_tokens(Starbucks_root)
    dict1 = get_tf(Heinz_root)
    print dict1
    dict2 = get_idf(Corpus_root)
    print dict2
    print get_tf_idf(dict1, dict2, 10)


if __name__ == "__main__":
    main()

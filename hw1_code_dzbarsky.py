# David Zbarsky: dzbarsky@wharton.upenn.edu
# Yaou Wang: yaouwang@wharton.upenn.edu

from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import sent_tokenize, word_tokenize

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
    files = []
    files.extend(PlaintextCorpusReader(directory, '.*').fileids())
    for subdir in get_sub_directories(directory):
        files.extend(get_all_files(directory + "/" + subdir))
    return files

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
    docs = []
    files = get_all_files(directory)
    for file in files:
        docs.append(load_file_tokens(directory + '/' + file))
    map = dict()
    for doc in docs:
        for token in doc:
            if token in map:
                continue
            occurences = 0
            for doc2 in docs:
                if token in doc2:
                    occurences += 1
            map[token] = occurences
            
    for token in map.keys():
        map[token] = Math.log(len(files)/map[token])
    return map

if __name__ == "__main__":
    main()

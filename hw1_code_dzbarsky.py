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
        map[token] /= len(tokens)
    return map

def main():
    print get_sub_directories(Corpus_root)
    print get_all_files(Starbucks_root)
    print load_file_sentences(Starbucks_root + '/118990300.txt')
    print load_collection_sentences(Starbucks_root)
    print load_file_tokens(Starbucks_root + '/118990300.txt')
    print load_collection_tokens(Starbucks_root)
    print get_tf(Heinz_root)

if __name__ == "__main__":
    main()

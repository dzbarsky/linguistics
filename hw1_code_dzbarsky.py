# David Zbarsky: dzbarsky@wharton.upenn.edu
# Yaou Wang: yaouwang@wharton.upenn.edu

from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import sent_tokenize

Corpus_root = 'corpus'
Mixed_root = 'mixed'
Starbucks_root = 'corpus/starbucks'

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

def load_file_sentences(file):
    index = file.rfind('/')
    dir = file[0:index]
    file = file[index + 1:]
    return sent_tokenize(PlaintextCorpusReader(dir, file).raw().lower())

def main():
    print get_sub_directories(Corpus_root)
    print get_all_files(Starbucks_root)
    print load_file_sentences(Starbucks_root + '/118990300.txt')

if __name__ == "__main__":
    main()

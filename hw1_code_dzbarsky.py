# David Zbarsky: dzbarsky@wharton.upenn.edu
# Yaou Wang: yaouwang@wharton.upenn.edu

from nltk.corpus import PlaintextCorpusReader

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

def main():
    print get_sub_directories(Corpus_root)
    print get_all_files(Starbucks_root)

if __name__ == "__main__":
    main()

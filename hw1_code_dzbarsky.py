# David Zbarsky: dzbarsky@wharton.upenn.edu
# Yaou Wang: yaouwang@wharton.upenn.edu

from nltk.corpus import PlaintextCorpusReader

Corpus_root = 'corpus'
Mixed_root = 'mixed'
Starbucks_root = 'corpus/starbucks'

def get_sub_directories(directory):
    subdirs = set()
    for file in PlaintextCorpusReader(directory, '.*').fileids():
        subdirs.add(file[0:file.find('/')])
    return list(subdirs)

def get_all_files(directory):
    files = []
    print "TESTING\n\n"
    #print PlaintextCorpusReader(directory, '.txt').fileids()
    #files.extend(PlaintextCorpusReader(directory, '.txt').fileids())
    print files
    for subdir in get_sub_directories(directory):
        files.extend(get_all_files(directory + "/" + subdir))
    return files

def main():
    print get_sub_directories(Corpus_root)
    print get_all_files(Corpus_root)

if __name__ == "__main__":
    main()

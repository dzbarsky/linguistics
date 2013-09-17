# David Zbarsky: dzbarsky@wharton.upenn.edu
# Yaou Wang: yaouwang@wharton.upenn.edu

from nltk.corpus import PlaintextCorpusReader

Corpus_root = '/home1/c/cis530/hw1/data/corpus'
Mixed_root = '/home1/c/cis530/hw1/data/mixed'

#files_all = PlaintextCorpusReader(Starbucks_root, '.*')

def get_sub_directories(directory):
    files_all = PlaintextCorpusReader(directory, '.*')
    print files_all



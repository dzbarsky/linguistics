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
    # We assume that a filename with a . is always a file rather than a directory
    # IF we were passed a file, just return that file.  This simplifies representing documents because they need to handle single files.
    if directory.find('.') < 0:
        return PlaintextCorpusReader(directory, '.*').fileids()
    #if directory is a file return the file in a list
    return [directory]

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
    if path.find('.') >= 0:
        tokens = load_file_tokens(path)
    else:
        tokens = load_collection_tokens(path)

    map = dict()
    for token in tokens:
       if token in map:
           map[token] += 1
       else:
           map[token] = 1
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

def get_words_freq(directory):
    map = dict()
    tokens = load_collection_tokens(directory)
    for token in tokens:
        if token in map:
            map[token] += 1
        else:
            map[token] = 1.0
    return map

def compute_mi(w, dir_map, corpus_map):
    num = dir_map[w]/sum(dir_map.values())
    denom = corpus_map[w]/sum(corpus_map.values())
    return math.log(num/denom)

def get_mi(directory, w):
    dir_map = get_words_freq(directory)
    corpus_map = get_words_freq('corpus')
    return compute_mi(w, dir_map, corpus_map)

def get_mi_words(directory, k):
    corpus_map = get_words_freq('corpus')
    dir_map = get_words_freq(directory)
    for token in corpus_map.keys():
        if corpus_map[token] < 5:
            del corpus_map[token]
            if token in dir_map:
                del dir_map[token]
    freq_map = dict()
    for token in dir_map.keys():
        freq_map[token] = compute_mi(token, dir_map, corpus_map)
    tokens = freq_map.keys()
    tokens.sort(key=freq_map.__getitem__, reverse=True)
    return tokens[:k-1]

def get_precision(L_1, k, L_2):
    L1_k = L1[:k]
    return len(L1_k.intersect(L_2))/float(len(L1_k))

def get_recall(L_1, k, L_2):
    L1_k = L1[:k]
    return len(L1_k.intersect(L_2))/float(len(L_2))

def get_fmeasure(L_1, k, L_2):
    precision = get_precision(L_1, k, L_2)
    recall = get_recall(L_1, k, L_2)
    return 2 * precision * recall / (precision + recall)

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
    if Xsum is 0 or Ysum is 0:
        return -1
    return numerator/(math.sqrt(Xsum) * math.sqrt(Ysum))

def get_doc_binary_vector(path, W):
    if path.find('.') < 0:
        fileText = load_collection_tokens(path)
    else:
        fileText = load_file_tokens(path)
    vector = []
    for word in W:
        if word in fileText:
            vector.append(1)
        else:
            vector.append(0)
    return vector

def get_tfidf_vector_helper(tf_map, idf_map, fileText, W):
    vector = []
    for word in W:
        if word in fileText:
            vector.append(tf_map[word]*idf_map[word])
        else:
            vector.append(0)
    return vector

def get_doc_tfidf_vector(path, W):
    tf_map = get_tf(path)
    idf_map = get_idf(Corpus_root)
    if path.find('.') < 0:
        fileText = load_collection_tokens(path)
    else:
        fileText = load_file_tokens(path)
    vector = get_tfidf_vector_helper(tf_map, idf_map, fileText, W)
    return vector

def get_tfidf_simdocs(ref_path, k):
    dict1 = get_tf(ref_path)
    dict2 = get_idf(Corpus_root)
    W = get_tf_idf_words(dict1, dict2, k)
    corp = get_doc_tfidf_vector(ref_path, W)
    testFiles = get_all_files(Mixed_root)
    doc_vectors = dict()
    for file in testFiles:
        fileText = load_file_tokens(Mixed_root + '/' + file)
        vector = get_tfidf_vector_helper(dict1, dict2, fileText, W)
        doc_vectors[file] = vector
    similarity = []
    for file in testFiles:
        similarity.append((file, cosine_similarity(doc_vectors[file], corp)))
    list.sort(similarity, key= lambda x: x[1], reverse=True)
    return similarity[:100]

def get_mi_simdocs(ref_path, k):
    corpus_map = get_words_freq('corpus')
    dir_map = get_words_freq(ref_path)
    W = get_mi_words(ref_path, k)
    corp = get_doc_binary_vector(ref_path, W)
    i = 0
    for word in W:
        corp[i] = corp[i]*compute_mi(word, dir_map, corpus_map)
        i += 1
    testFiles = get_all_files(Mixed_root)
    doc_vectors = dict()
    for file in testFiles:
        fileText = load_file_tokens(Mixed_root + '/' + file)
        vector = get_doc_binary_vector(Mixed_root + '/' + file, W)
        doc_vectors[file] = vector
    sim = []
    for file in testFiles:
        sim.append((file, cosine_similarity(doc_vectors[file], corp)))
    list.sort(sim, key= lambda x: x[1], reverse=True)
    return sim[:100]
'''
 We see for fixed path = Starbucks_root and a fixed k = 10, the tf_idf model works
 better and finds more relevant results (all the starbucks###.txt are at the top whereas
 in the MI approach there is a heinz###.txt). We like the tf_idf approach better because
 it stores actual values in the vectors whereas the MI method only has binary vectors
 (which leads to many files with the same cosine similarity). We also think that the
 tf_idf has more information with regards to the weightings of each word.
'''

def get_word_contexts(word, directory):
    context = []
    tokens = load_collection_tokens(directory)
    for i in range(len(tokens)):
	if tokens[i] == word:
	    context.append(tokens[i-1])
	    context.append(tokens[i+1])
    return list(set(context))

def get_common_contexts(word1, word2, directory):
    context1 = get_word_contexts(word1, directory)
    context2 = get_word_contexts(word2, directory)
    return list(set(context1) & set(context2))

def compare_word_sim(path, k):
    

def main():
    #print get_sub_directories(Corpus_root)
    #print get_all_files(Starbucks_root + '/118990300.txt')
    #print load_file_sentences(Starbucks_root + '/118990300.txt')
    #print load_collection_sentences(Starbucks_root)
    #print load_file_tokens(Starbucks_root + '/118990300.txt')
    #print load_collection_tokens(Starbucks_root)
    #dict1 = get_tf(Heinz_root)
    #print dict1
    #dict2 = get_idf(Corpus_root)
    #print get_tf_idf_words(dict1, dict2, 10)
    #print get_mi_words(Starbucks_root, 10)
    #sentences = ["this is a test", "this is another test"]
    #print create_feature_space(sentences)
    #feature_space = create_feature_space(sentences)
    #print vectorize(feature_space, "another test")
    #print cosine_similarity([1,1,0,1], [2,0,1,1])
    #print get_tfidf_simdocs(Starbucks_root, 10)
    #print get_mi_simdocs(Starbucks_root, 10)
    print get_common_contexts('sales', 'earnings', Starbucks_root)

if __name__ == "__main__":
    main()

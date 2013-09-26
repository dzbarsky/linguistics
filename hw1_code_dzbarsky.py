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
    # Find al the files and parse out the directory up to the /, those are the
    # subdirectories.
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
    # If this is a single file, load the tokens.  Otherwise, load all tokens
    # for the collection.
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

def get_tfidf_words(dict1, dict2, k):
    tuples = []
    # Compute tf-idf as tf*idf
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
    corpus_map = get_words_freq(Corpus_root)
    return compute_mi(w, dir_map, corpus_map)

def get_mi_words(directory, k):
    corpus_map = get_words_freq(Corpus_root)
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
        # Tokenize the words because we want to ignore the punctuation.
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
    W = get_tfidf_words(dict1, dict2, k)
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

def feature_space_helper(list):
    map = dict()
    counter = 0
    for token in list:
        if not token in map:
            map[token] = counter
            counter += 1
    return map

def vectorize_helper(feature_space, list):
    vector = []
    for i in feature_space.keys():
        vector.append(0)

    for token in set(list):
        vector[feature_space[token]] = 1

    return vector

def compare_word_sim(path, k):
    W = get_tfidf_words(get_tf(path), get_idf(Corpus_root), k)
    contexts_list = []
    contexts = []
    for word in W:
        con = get_word_contexts(word,path)
        contexts_list.append(con)
        contexts.extend(con)
    feature_space = feature_space_helper(set(contexts))
    word_vectors = []
    for context in contexts_list:
        word_vectors.append(vectorize_helper(feature_space, context))
    matrix = []
    for vector1 in word_vectors:
        sim = []
        for vector2 in word_vectors:
            sim.append(cosine_similarity(vector1, vector2))
        matrix.append(sim)
    return matrix

'''
When printing compare_word_sim(Corpus_root + '/qualcomm', 10), we get:

[[1.0, 0.23124864503144016, 0.18895099718933206, 0.1975029649151584, 0.17431926281167376, 0.1198271198752773, 0.2776795303967047, 0.042811734078171355, 0.19650294347725425, 0.19334007027804562], [0.23124864503144016, 1.0, 0.2136183479338211, 0.24253562503633294, 0.14002800840280097, 0.09378721544644802, 0.18069243800050583, 0.04356068418690321, 0.12065379433217112, 0.17705013195704256], [0.18895099718933206, 0.2136183479338211, 1.0, 0.40892939847550536, 0.24895967608079353, 0.1809381136270079, 0.2386133605551424, 0.029660839433408286, 0.2347262634065101, 0.2589698645587962], [0.1975029649151584, 0.24253562503633294, 0.40892939847550536, 1.0, 0.2680554821237548, 0.1381052127208091, 0.2515631714063358, 0.03207237536192409, 0.19289588622650466, 0.26650662042910983], [0.17431926281167376, 0.14002800840280097, 0.24895967608079353, 0.2680554821237548, 1.0, 0.46047009705340675, 0.2150670189265881, 0.02592379236826063, 0.15044515564140667, 0.1717073987899772], [0.1198271198752773, 0.09378721544644802, 0.1809381136270079, 0.1381052127208091, 0.46047009705340675, 1.0, 0.1702367079689875, 0.013022324932151278, 0.15801683252692647, 0.08625395191157326], [0.2776795303967047, 0.18069243800050583, 0.2386133605551424, 0.2515631714063358, 0.2150670189265881, 0.1702367079689875, 0.9999999999999999, 0.042575420953522417, 0.24066290927412823, 0.23072744115175065], [0.042811734078171355, 0.04356068418690321, 0.029660839433408286, 0.03207237536192409, 0.02592379236826063, 0.013022324932151278, 0.042575420953522417, 1.0, 0.0063819965088956964, 0.05098769784045568], [0.19650294347725425, 0.12065379433217112, 0.2347262634065101, 0.19289588622650466, 0.15044515564140667, 0.15801683252692647, 0.24066290927412823, 0.0063819965088956964, 1.0, 0.12681431837544702], [0.19334007027804562, 0.17705013195704256, 0.2589698645587962, 0.26650662042910983, 0.1717073987899772, 0.08625395191157326, 0.23072744115175065, 0.05098769784045568, 0.12681431837544702, 0.9999999999999998]]

'''

def get_similar_pairs(k, N):
    W = get_tfidf_words(get_tf(Starbucks_root), get_idf(Corpus_root), k)
    matrix = compare_word_sim(Starbucks_root, k)
    map = dict()
    for i in range(len(matrix)):
        for j in range(i):
            map[W[i] + '_' + W[j]] = matrix[i][j]

    l = []
    for w in sorted(map, key=map.get, reverse=True):
        l.append((w, map[w]))
    return l[:N]

'''
When printing get_similar_pairs(30, 10), we get:

[('million_$', 0.6614801056400267), ('%_percent', 0.5501485601695769), ('same_period', 0.5012804118276031), ('corporation_corp.', 0.46861058735073596), ('net_consolidated', 0.3818813079129867), ('earnings_revenues', 0.36666666666666664), ('earnings_consolidated', 0.35856858280031806), ('earnings_sales', 0.3261640365267211), ('sales_revenues', 0.3261640365267211), ('company_corp.', 0.3203704199628461)]

'''

def main():
    #print get_sub_directories(Corpus_root)
    #print get_all_files(Starbucks_root + '/118990300.txt')
    #print load_file_sentences(Starbucks_root + '/118990300.txt')
    #print load_collection_sentences(Starbucks_root)
    #print load_file_tokens(Starbucks_root + '/118990300.txt')
    #print load_collection_tokens(Starbucks_root)
    #dict1 = get_tf(Starbucks_root)
    #dict2 = get_idf(Corpus_root)
    #print get_tfidf_words(dict1, dict2, 30)
    #print get_mi_words(Starbucks_root, 10)
    #sentences = ["this is a test", "this is another test"]
    #print create_feature_space(sentences)
    #feature_space = create_feature_space(sentences)
    #print vectorize(feature_space, "another test")
    #print cosine_similarity([1,1,0,1], [2,0,1,1])
    #print get_tfidf_simdocs(Starbucks_root, 10)
    #print get_mi_simdocs(Starbucks_root, 10)
    #print get_common_contexts('sales', 'earnings', Starbucks_root)
    #print compare_word_sim(Starbucks_root, 10)
    #print compare_word_sim(Corpus_root + '/qualcomm', 10)
    print get_similar_pairs(30, 10)

if __name__ == "__main__":
    main()

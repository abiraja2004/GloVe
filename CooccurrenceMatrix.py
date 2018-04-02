# Perform imports
from keras.preprocessing.text import text_to_word_sequence
import numpy as np
import pickle

'''
Define your training directory and training documents
'''
TRAIN_DIR = "/home/sudeep/Work/Text Summarization/GigaSentSummDist/dev/"
TRAIN_DOCS = ["valid.article.filter.txt", "valid.title.filter.txt"]

'''
tokenise TRAIN_DOCS into a list of words

returns:    docs    -> a list of words representation of the documents    
'''
def get_data():
    docs = []
    for DOC in TRAIN_DOCS:
        f = open(TRAIN_DIR+DOC, "r")
        docs += f.readlines()
        f.close()
    docs = list(map(text_to_word_sequence, docs))
    return docs

'''
parameters:     docs            -> a list of words representation of the document

returns   :     V               -> size of the vocabulary 
                word_to_index   -> dictionary whose keys are words and values are the corresponding index
                index_to_word   -> list containing the vocab words indexed by their indices     
'''
def build_vocab(docs):
    V = 0
    word_to_index = {}
    index_to_word = [None]
    for doc in docs:
        for word in doc:
            if word not in word_to_index:
                V = V + 1
                word_to_index[word] = V
                index_to_word.append(word)
    print("The vocab size is :" + str(V))       
    return V, word_to_index, index_to_word

'''
parameters:     docs            -> a list of words representation of the document
                word_to_index   -> q list containing the vocab words indexed by their indices 
                
returns   :     indexed_docs    -> documents whose words in the articles are replaced by the corresponding indices 
'''
def get_indexed_documents(docs, word_to_index):
    indexed_docs = []
    for doc in docs:
        indexed_doc = []
        for word in doc:
            indexed_doc.append(word_to_index[word])
        assert(len(indexed_doc) == len(doc))
        indexed_docs.append(indexed_doc)
    return indexed_docs

'''
builds the cooccurence matrix for the corpus

parameters:     k -> window size
                V -> vocab_size
                docs -> indexed documents (list of word indices)
                
returns   :     Numpy array contaning the coocurence matrix   
'''
def build_cooccurence(docs, V, k):
    X = [[0 for _ in range(V+1)] for _ in range(V+1)]
    for doc in docs:
        l = len(doc)
        for i in range(l):
            for j in range(i-k, i+k+1):
                if j < 0 or j >= l:
                    continue
                X[doc[i]][doc[j]] += 1
    for i in range(1, V+1):
        for j in range(i, V+1):
            assert(X[i][j] == X[j][i])
    return np.array(X)

'''
stores the cooccurence matrix, word_to_index dictionary, and index_to_word list in the working directory
'''
def main():
    docs = get_data()
    V, word_to_index, index_to_word = build_vocab(docs)
    
    f = open("word_to_index.pkl","wb")
    pickle.dump(word_to_index,f)
    f.close()
    
    f = open("index_to_word.pkl", "wb")
    pickle.dump(index_to_word, f)
    f.close()
    
    docs = get_indexed_documents(docs, word_to_index)
    
    P = build_cooccurence(docs, V, k=5)
    print("Shape of occurrence matrix: " + str(P.shape))
    
    np.save('CoocuurenceMatrix', P)
    
main()


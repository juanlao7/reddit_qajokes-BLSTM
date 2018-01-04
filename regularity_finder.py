import numpy as np
from sklearn.neighbors import KNeighborsClassifier

source = 'glove'
k = 5
embeddings_size = 300

def fileToEmbeddings(filePath):
    f = open(filePath)
    embeddings_word2vector = {}
    
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_word2vector[word] = coefs
    
    f.close()
    return embeddings_word2vector

def askWord(caption, embeddings_word2vector):
    while True:
        word = raw_input(caption)
        
        if word in embeddings_word2vector:
            return embeddings_word2vector[word]
        
        print 'Word not found in the embedding dictionary.'

# Loading embeddings
print 'Loading embeddings...'

if source == 'glove':
    embeddings_word2vector = fileToEmbeddings('glove.6B.' + str(embeddings_size) + 'd.txt')
elif source == 'glove42':
    embeddings_word2vector = fileToEmbeddings('glove.42B.' + str(embeddings_size) + 'd.txt')
elif source == 'lexvec':
    embeddings_word2vector = fileToEmbeddings('lexvec.commoncrawl.' + str(embeddings_size) + 'd.W.pos.vectors')
elif source == 'word2vec':
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative' + str(embeddings_size) + '.bin', binary=True)
    embeddings_word2vector = {}
    
    for word in model.wv.vocab:
        embeddings_word2vector[word] = model.word_vec(word)
else:
    raise Exception('Unknown embedding source')

print 'Preparing KNN for recalling', len(embeddings_word2vector), 'words...'

keys = embeddings_word2vector.keys()
values = embeddings_word2vector.values()

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(values, keys)

while True:
    print 'Let\'s find a new regularity in the form A - B + C = D (e.g. king - man + woman = queen).'
    a = askWord('A? ', embeddings_word2vector)
    b = askWord('B? ', embeddings_word2vector)
    c = askWord('C? ', embeddings_word2vector)
    
    print k, 'possible values for D:'
    
    for neighbor in knn.kneighbors([a - b + c], k, False)[0]:
        print keys[neighbor]

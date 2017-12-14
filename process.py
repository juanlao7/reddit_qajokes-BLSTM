import os
import random
import numpy as np
import cPickle as pickle
import hickle
from keras.models import Sequential
from keras.layers import Dense, Activation, RepeatVector, TimeDistributed, Bidirectional
from keras.layers import LSTM, GRU, Dropout
from keras.optimizers import RMSprop, SGD
import argparse
import json
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.utils import np_utils
import csv
from keras.models import load_model
from sklearn.svm.libsvm import predict
from sklearn.neighbors import KNeighborsClassifier
from sets import Set
import gensim
import re

ACTIVATION_FUNCTION = 'tanh'
EPOCHS = 200
BATCH_SIZE = 100
LEARNING_RATE = 0.001
LSTM_UNITS = 128
MAX_LENGTH = 20
REMOVE_UNEXISTENT = False
SHOW_PADDINGS = True

def plotOptions(results, title, ylabel, keys):
    plt.gca().set_color_cycle(None)
    
    for i in results:
        plt.plot(results[i]['h'][keys[0]])
    
    plt.legend(results.keys(), loc='upper right')
    
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Epoch')
    
    #plt.ylim(ymin=0.024, ymax=0.045)
    
    plt.show()

def load_config_file(nfile, abspath=False):
    ext = '.json' if 'json' not in nfile else ''
    pre = '' if abspath else './'
    fp = open(pre + nfile + ext, 'r')

    s = ''

    for l in fp:
        s += l

    return json.loads(s)

def getText(sequence, knn, config):
    if config['tokens'] == 'extra':
        recalls = []
        
        for embedding in sequence:
            if embedding[-1] > 0.5:
                recalls.append('')
            elif embedding[-1] < -0.5:
                recalls.append('*')
            else:
                recalls.append(knn.predict([embedding[:-1]])[0])
    else:
        recalls = [knn.predict([embedding])[0] for embedding in sequence]
    
    if SHOW_PADDINGS:
        recalls = [word if word else '_' for word in recalls]
    
    return ' '.join(filter(None, recalls))

def getClosest(embedding, knn, k, keys):
    return [keys[i] for i in knn.kneighbors([embedding], k, False)[0]]

def prepareSequence(sequence):
    sequence = ''.join(ch for ch in sequence.lower() if ch.isalnum() or ch == '.' or ch == ' ')
    preparedSequence = sequence.split(' ')
    
    for word in preparedSequence:
        if re.match('(^[0-9]*[a-z]+$)|(^[0-9]+$)', word) is None:
            return None
    
    return preparedSequence

def embedSequence(sequence, sequence_size, datasetWords, embeddings_word2vector, padding, unknown):
    sequenceParsed = []
    
    for j in xrange(len(sequence)):
        word = sequence[j]
        datasetWords.add(word)
        
        if word in embeddings_word2vector:
            sequenceParsed.append(embeddings_word2vector[word])
        elif unknown is None:
            return None
        else:
            sequenceParsed.append(unknown)
    
    return np.array(sequenceParsed + [padding] * (sequence_size - len(sequenceParsed)))

def embedDataset(x, y, x_size, y_size, embeddings_word2vector, padding, unknown):
    datasetWords = Set()
    xParsed = []
    yParsed = []
    
    for i in xrange(len(x)):
        input = embedSequence(x[i], x_size, datasetWords, embeddings_word2vector, padding, unknown)
        output = embedSequence(y[i], y_size, datasetWords, embeddings_word2vector, padding, unknown)
        
        if input is None or output is None:
            continue
        
        xParsed.append(input)
        yParsed.append(output)
    
    return np.array(xParsed), np.array(yParsed), datasetWords

def loadJokes():
    f = open('jokes.csv', 'r')
    reader = csv.reader(f, delimiter=',', quotechar='"')
    x = []
    y = []
    
    next(reader)       # Skip the first line
    
    for joke in reader:
        question = prepareSequence(joke[1])
        answer = prepareSequence(joke[2])
        
        if question is not None and answer is not None and len(question) + len(answer) <= MAX_LENGTH:
            x.append(question)
            y.append(answer)
    
    f.close()
    return x, y

def loadData(config):
    source = config['embedding_source']
    embeddings_size = config['embedding_size']
    remove_unknown = config['remove_unknown']
    
    # Loading embeddings
    print 'Loading embeddings...'
    
    if source == 'glove':
        f = open(os.path.join('.', 'glove.6B.' + str(embeddings_size) + 'd.txt'))
        embeddings_word2vector = {}
        
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_word2vector[word] = coefs
        
        f.close()
    elif source == 'word2vec':
        model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative' + str(embeddings_size) + '.bin', binary=True)
        embeddings_word2vector = {}
        
        for word in model.wv.vocab:
            embeddings_word2vector[word] = model.word_vec(word)
    else:
        raise Exception('Unknown embedding source')
    
    print 'Preprocessing each word of the embeddings dictionary...'
    
    for word in embeddings_word2vector.keys():
        if embeddings_word2vector[word] is None:        # There are some elements with no embeddings. We must remove them in order to represent the embedding space in the K-D tree later.
            del embeddings_word2vector[word]
            continue
        
        processed = prepareSequence(word)
        processed = processed[0] if processed else None
        
        if processed is None:                           # Deleting strange words, such as "str95bb".
            del embeddings_word2vector[word]
        elif processed != word:
            if processed not in embeddings_word2vector: # Only adding the processed word if it didn't exist yet
                embeddings_word2vector[processed] = embeddings_word2vector[word]
            
            del embeddings_word2vector[word]
    
    # Defining padding and unknown words.
    
    if config['tokens'] == '1s-1s':
        padding = np.ones(embeddings_size)
        unknown = np.full(embeddings_size, -1.0)
    elif config['tokens'].startswith('stdevs'):
        multiplier = float(config['tokens'][len('stdevs'):])
        
        mean = np.mean(embeddings_word2vector.values(), axis=0)
        stdev = np.std(embeddings_word2vector.values(), axis=0)
        
        padding = mean + stdev * multiplier
        unknown = mean - stdev * multiplier
    elif config['tokens'] == 'extra':
        embeddings_size += 1
        
        # Adding a new element with value 0 in each embedding
        
        for word in embeddings_word2vector:
            newEmbedding = np.zeros(embeddings_size)
            newEmbedding[:-1] = embeddings_word2vector[word]
            embeddings_word2vector[word] = newEmbedding
        
        padding = np.zeros(embeddings_size)
        padding[-1] = 1.0
        
        unknown = np.zeros(embeddings_size)
        unknown[-1] = -1.0 
    else:
        raise Exception('Unknown token encoding.')
    
    if remove_unknown:
        unknown = None
    
    # Loading jokes data set
    
    print 'Loading jokes...'
    x, y = loadJokes()
    x_size = max([len(i) for i in x])
    y_size = max([len(i) for i in y])
    
    print 'Embedding jokes...'
    x, y, datasetWords = embedDataset(x, y, x_size, y_size, embeddings_word2vector, padding, unknown)
    
    if REMOVE_UNEXISTENT:
        print 'Removing unexistent words from the embeddings dictionary...'
        
        for word in embeddings_word2vector.keys():
            if word not in datasetWords:
                del embeddings_word2vector[word]
    
    embeddings_word2vector[''] = padding
    
    if unknown is not None:
        embeddings_word2vector['*'] = unknown
    
    print len(x), 'jokes loaded; x_size =', x_size, '; y_size =', y_size
    return x, y, embeddings_word2vector

# Initialization
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config', help='Experiment configuration')
args = parser.parse_args()

configs = load_config_file(args.config)
results = {}

for configName in configs:
    print('###    ' + configName + '    ###')
    
    config = configs[configName]
    
    # Preparing the data
    
    x, y, embeddings_word2vector = loadData(config)
    
    # Preparing the model.
    
    resultsDir = 'results/' + args.config
    
    try:
        os.makedirs(resultsDir)
    except:
        pass
    
    resultFileName = resultsDir + '/' + configName + '.result'
    modelFileName = resultsDir + '/' + configName + '.h5'
    
    if os.path.isfile(resultFileName):
        handler = open(resultFileName, 'rb')
        results[configName] = pickle.load(handler)
        handler.close()
        model = load_model(modelFileName)
    else:
        model = Sequential()
        model.add(Bidirectional(LSTM(LSTM_UNITS, activation=ACTIVATION_FUNCTION, implementation=2), input_shape=(x.shape[1], x.shape[2])))
        model.add(RepeatVector(y.shape[1]))
        model.add(Bidirectional(LSTM(LSTM_UNITS, activation=ACTIVATION_FUNCTION, implementation=2, return_sequences=True)))
        model.add(TimeDistributed(Dense(y.shape[2])))
        
        optimizer = RMSprop(lr=LEARNING_RATE)
        model.compile(optimizer=optimizer, loss='mse')
        
        # Training the model
    
        h = model.fit(x, y,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      verbose=1)
    
        # Results
    
        results[configName] = {
            'h': h.history
        }
        
        handler = open(resultFileName, 'wb')
        pickle.dump(results[configName], handler)
        handler.close()
        
        model.save(modelFileName)
    
    # Trying the model
    print 'Preparing KNN for recalling', len(embeddings_word2vector), 'words...'
    
    padding = embeddings_word2vector['']
    unknown = embeddings_word2vector['*']
    
    if config['tokens'] == 'extra':
        # We remove the padding and unknown tokens from the embedding space, since we recall them with a threshold on the extra dimension.
        del embeddings_word2vector['']
        del embeddings_word2vector['*']
        
        # We remove the extra dimension
        values = [embedding[:-1] for embedding in embeddings_word2vector.values()]
        padding = padding[:-1]
        unknown = unknown[:-1]
    else:
        values = embeddings_word2vector.values()
    
    keys = embeddings_word2vector.keys()

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(values, keys)
    
    print 'Closest words to the padding token:', getClosest(padding, knn, 5, keys)
    
    if not config['remove_unknown']:
        print 'Closest words to the unknown token:', getClosest(unknown, knn, 5, keys)
    
    for i in xrange(10):
        print i, ':', getText(x[i], knn, config) + '?'
        print '    Actual answer:    ', getText(y[i], knn, config)
        
        prediction = model.predict(np.reshape(x[i], (1, x.shape[1], x.shape[2])))[0]
        print '    Model answer:     ', getText(prediction, knn, config)
        
        print

print '### FINISH! ###'

for i in results:
    h = results[i]['h']
    print i, '(' + str(len(h['loss'])), 'epochs):'
    result = [str(round(i, 6)) for i in [h['loss'][-1]]]
    print ','.join(result)
    
# Plotting
plotOptions(results, 'Model loss', 'Loss', ['loss'])


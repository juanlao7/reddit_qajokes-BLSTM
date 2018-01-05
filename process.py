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
from sklearn.neighbors import KNeighborsClassifier
from sets import Set
import gensim
import re
import copy

ACTIVATION_FUNCTION = 'tanh'
EPOCHS = 200
BATCH_SIZE = 100
LEARNING_RATE = 0.001
LSTM_UNITS = 128
MAX_LENGTH = 20
REMOVE_UNEXISTENT = False
SHOW_PADDINGS = True
EXTRA_DIMENSION_PADDING_THRESHOLD = 0.2
SAMPLE_JOKES = 0
ANALYSIS_JOKES = 0
FAST = False

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
            if embedding[-1] > EXTRA_DIMENSION_PADDING_THRESHOLD:
                recalls.append('')
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
    sequence = ''.join(ch for ch in sequence.lower() if ch.isalnum() or ch == ' ')
    preparedSequence = sequence.split(' ')
    
    for word in preparedSequence:
        if re.match('(^[0-9]*[a-z]+$)|(^[0-9]+$)', word) is None:
            return None
    
    return preparedSequence

def embedSequence(sequence, sequence_size, datasetWords, embeddings_word2vector, padding):
    sequenceParsed = []
    
    for j in xrange(len(sequence)):
        word = sequence[j]
        datasetWords.add(word)
        
        if word in embeddings_word2vector:
            sequenceParsed.append(embeddings_word2vector[word])
        else:
            return None
    
    return np.array(sequenceParsed + [padding] * (sequence_size - len(sequenceParsed)))

def embedDataset(x, y, x_size, y_size, embeddings_word2vector, padding):
    datasetWords = Set()
    xParsed = []
    yParsed = []
    
    for i in xrange(len(x)):
        input = embedSequence(x[i], x_size, datasetWords, embeddings_word2vector, padding)
        output = embedSequence(y[i], y_size, datasetWords, embeddings_word2vector, padding)
        
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

def loadData(config):
    source = config['embedding_source']
    embeddings_size = config['embedding_size']
    
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
    
    print len(embeddings_word2vector), 'word embeddings.'
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
    
    # Defining the padding token.
    
    if config['tokens'] == '0s':
        padding = np.zeros(embeddings_size)
    elif config['tokens'].startswith('stdevs'):
        multiplier = float(config['tokens'][len('stdevs'):])
        
        mean = np.mean(embeddings_word2vector.values(), axis=0)
        stdev = np.std(embeddings_word2vector.values(), axis=0)
        
        padding = mean + stdev * multiplier
    elif config['tokens'] == 'extra':
        embeddings_size += 1
        
        # Adding a new element with value 0 in each embedding
        
        for word in embeddings_word2vector:
            newEmbedding = np.zeros(embeddings_size)
            newEmbedding[:-1] = embeddings_word2vector[word]
            embeddings_word2vector[word] = newEmbedding
        
        padding = np.zeros(embeddings_size)
        padding[-1] = 1.0
    else:
        raise Exception('Unknown token encoding.')
    
    # Loading jokes data set
    
    print 'Loading jokes...'
    x, y = loadJokes()
    x_size = max([len(i) for i in x])
    y_size = max([len(i) for i in y])
    
    print 'Embedding jokes...'
    x, y, datasetWords = embedDataset(x, y, x_size, y_size, embeddings_word2vector, padding)
    
    if REMOVE_UNEXISTENT:
        print 'Removing unexisting words from the embeddings dictionary...'
        
        for word in embeddings_word2vector.keys():
            if word not in datasetWords:
                del embeddings_word2vector[word]
    
    embeddings_word2vector[''] = padding
    
    print len(x), 'jokes loaded; x_size =', x_size, '; y_size =', y_size
    return x, y, embeddings_word2vector

def analyzeTokens(y, knn, config, predictions):
    totalWrongPaddings = []
    
    for i in xrange(ANALYSIS_JOKES):
        actualAnswer = getText(y[i], knn, config)
        modelAnswer = getText(predictions[i], knn, config)
        
        wrongPaddings = 0
        
        for j in xrange(y.shape[1]):
            if (modelAnswer[j] == '_' or actualAnswer[j] == '_') and modelAnswer[j] != actualAnswer[j]:
                wrongPaddings += 1
                
        totalWrongPaddings.append(wrongPaddings)
    
    return totalWrongPaddings

def vectorInSequence(vector, sequence):
    for elementIndex in xrange(sequence.shape[0]):
        if np.array_equal(vector, sequence[elementIndex, :]):
            return True
    
    return False

def meetsFilter(question, answer, testFilter):
    for embeddedFilter in testFilter:
        if not vectorInSequence(embeddedFilter, question) and not vectorInSequence(embeddedFilter, answer):
            return False
    
    return True

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
    
    x, y, embeddings_word2vector = loadData(config) if not FAST else None, None, None
    
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
    if SAMPLE_JOKES > 0:
        print 'Preparing KNN for recalling', len(embeddings_word2vector), 'words...'
        originalEmbeddings_word2vector = copy.deepcopy(embeddings_word2vector)
        
        padding = embeddings_word2vector['']
        
        if config['tokens'] == 'extra':
            # We remove the padding tokens from the embedding space, since we recall them with a threshold on the extra dimension.
            del embeddings_word2vector['']
            
            # We remove the extra dimension
            values = [embedding[:-1] for embedding in embeddings_word2vector.values()]
            padding = padding[:-1]
        else:
            values = embeddings_word2vector.values()
        
        keys = embeddings_word2vector.keys()
    
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(values, keys)
        
        print 'Closest words to the padding token:', getClosest(padding, knn, 5, keys)
    
        print 'Predicting the first', SAMPLE_JOKES, 'jokes...'
        predictions = model.predict(x[:SAMPLE_JOKES])
        accuracies = []
        testFilter = config['test_filter'] if 'test_filter' in config else []
        
        for i in xrange(len(testFilter)):
            testFilter[i] = embeddings_word2vector[testFilter[i]]
        
        for it in xrange(SAMPLE_JOKES):
            if meetsFilter(x[it], y[it], testFilter):
                print it, ':', getText(x[it], knn, config) + '?'
                
                actualAnswer = getText(y[it], knn, config)
                modelAnswer = getText(predictions[it], knn, config)
                accuracy = len([i for i, j in zip(actualAnswer.split(), modelAnswer.split()) if i == j]) / float(y.shape[1])
                
                print '    Actual answer:    ', actualAnswer
                print '    Model answer:     ', modelAnswer
                print '    Accuracy:         ', round(accuracy * 100, 2), '%'
                print
                
                accuracies.append(accuracy)
        
        print 'Global accuracy: ', np.mean(accuracies), '+-', np.std(accuracies)
    
        if args.config == 'tokens.json' and ANALYSIS_JOKES > 0:
            # This is an analysis only performed when comparing different tokens.
            predictions = model.predict(x[:ANALYSIS_JOKES])
            
            if config['tokens'] == 'extra':
                print 'Optimizing the thresholds for the extra dimension...'
                
                paddingResults = []
                tests = [x / 100.0 for x in xrange(10, 95, 10)]
                
                for i in tests:
                    print 'Testing threshold', i
                    EXTRA_DIMENSION_PADDING_THRESHOLD = i
                    totalWrongPaddings = analyzeTokens(y, knn, config, predictions)
                    paddingResults.append(np.mean(totalWrongPaddings))
                
                EXTRA_DIMENSION_PADDING_THRESHOLD = tests[paddingResults.index(min(paddingResults))]
                
                print 'Optimal padding token threshold:', EXTRA_DIMENSION_PADDING_THRESHOLD
            
            print 'Calculating wrong tokens...'
            totalWrongPaddings = analyzeTokens(y, knn, config, predictions)
            
            print 'Token encoding:', config['tokens']
            print 'Error on paddings:', np.mean(totalWrongPaddings), '+-', np.std(totalWrongPaddings)
    
        if 'interactive' in config and config['interactive']:
            while True:
                question = raw_input('Ask a question: ')
                
                if not question:
                    break
                    
                questionVector = embedSequence(prepareSequence(question), x.shape[1], Set(), originalEmbeddings_word2vector, originalEmbeddings_word2vector[''])
                prediction = model.predict(np.array([questionVector]))[0]
                print getText(prediction, knn, config)

print '### FINISH! ###'

for i in results:
    h = results[i]['h']
    print i, '(' + str(len(h['loss'])), 'epochs):'
    result = [str(round(i, 6)) for i in [h['loss'][-1]]]
    print ','.join(result)
    
# Plotting
plotOptions(results, 'Model loss', 'Loss', ['loss'])


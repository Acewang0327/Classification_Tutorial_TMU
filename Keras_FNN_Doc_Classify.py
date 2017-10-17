
# coding: utf-8

# In[1]:

'''
Trains an FNN on the IMDB sentiment classification task.
'''
from __future__ import print_function
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import Flatten
from keras.layers import Dropout
from keras.datasets import imdb
from keras import backend as K
from keras import regularizers
import six


# In[2]:

'''
Settings
'''
# maximum size of vocabulary
max_vocabulary = 20000
# maximum length of each document
maxlen = 100
# size of word embeddings
embedding_size = 32
# size of each mini-batch
batch_size = 16
# epochs
epochs = 20


# In[14]:

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_vocabulary,skip_top=10)
imdb_w2id = imdb.get_word_index()
imdb_id2w = dict([(i,w) for (w,i) in six.iteritems(imdb_w2id)])
print(len(x_train), 'training documents')
print(len(x_test), 'test documents')
print('Example data:')
print('=====Word=====')
print(' '.join(imdb_id2w[w] for w in x_train[5]))
print('======ID======')
print(x_train[5])
print('=====Label====')
print(y_train[5])


# In[13]:

print('Padding documents (samples x length)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('Example padded data:')
print('=====Word=====')
print(' '.join(imdb_id2w.get(w,'-') for w in x_train[5]))
print('======ID======')
print(x_train[5])


# In[ ]:

print('x_train shape: {} documents x {} words'.format(x_train.shape[0], x_train.shape[1]))
print('y_train shape: {} labels'.format(y_train.shape[0]))
print('x_test shape: {} documents x {} words'.format(x_test.shape[0], x_test.shape[1]))
print('y_test shape: {} labels'.format(y_test.shape[0]))


# In[ ]:

K.clear_session()
print('Building model...')
model = Sequential()
model.add(Embedding(max_vocabulary, embedding_size, input_shape=(maxlen,)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(4, kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print('Start training...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))


# In[ ]:

print('Start testing...')
score, acc = model.evaluate(x_test, y_test, verbose=2,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

K.clear_session()


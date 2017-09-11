# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 11:37:26 2017

@author: User
"""

import glob
import pickle
import codecs
import numpy as np
import tensorflow as tf
import helpers


####PREPROCESSING START

book_filenames =sorted(glob.glob("C:/Users/User/Desktop/work/data1/firstintro10.txt"))
print("Found {} books".format(len(book_filenames)))

corpus_raw=u""
for filename in book_filenames:
    with codecs.open(filename, 'r', 'utf-8') as book_file:
        corpus_raw+=book_file.read()
        
print("Corpus is {} characters long".format(len(corpus_raw)))

#corpus_raw: one string


corpus_splitlines = corpus_raw.splitlines()
#corpus_splitlines: list of all sentence

corpus=[]
for sentence in corpus_splitlines:
    sentence_wo_dot=sentence.replace('.', '')
    word = sentence_wo_dot.split(' ')
    corpus.append(word)
#corpus: list of sentence which is list of words
    

corpus_set=set()
for sentence in corpus:
    for word in sentence:
        corpus_set.add(word)
#corpus_set: set of all word        
        

voca_size=len(corpus_set)

voca_to_int=dict(zip(corpus_set, range(voca_size)))
int_to_voca=dict(zip(range(voca_size), corpus_set))


corpus_int=[]
for sentence in corpus:
    tmp=[]
    for word in sentence:
       word_int = voca_to_int[word]
       tmp.append(word_int)
    corpus_int.append(tmp)
#corpus_int: list of sentence which is list of word_int

####PREPROCESSING END

EOS=1
encoder_input_data = corpus_int
decoder_input_data = []
decoder_target_data = []

for sentence in corpus_int:
    tmp=[]
    tmp=[EOS]+sentence
    tmp1=[]
    tmp1=sentence+[EOS]
    decoder_input_data.append(tmp)
    decoder_target_data.append(tmp1)
    
    
#batch, batch_len=helpers.batch(corpus_int[batch_all])


#####GRAPH START
tf.reset_default_graph()
sess=tf.InteractiveSession()

PAD=0
EOS=1
vocab_size=voca_size
input_embedding_size=20
encoder_hidden_units=20
decoder_hidden_units=encoder_hidden_units

encoder_inputs=tf.placeholder(shape=(None,None),dtype=tf.int32, name='encoder_inputs')
decoder_inputs=tf.placeholder(shape=(None,None),dtype=tf.int32, name='decoder_inputs')
decoder_targets=tf.placeholder(shape=(None,None),dtype=tf.int32, name='decoder_targets')

embeddings=tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0,1.0), dtype=tf.float32)

encoder_inputs_embedded=tf.nn.embedding_lookup(embeddings, encoder_inputs)
decoder_inputs_embedded=tf.nn.embedding_lookup(embeddings, decoder_inputs)

encoder_cell=tf.contrib.rnn.LSTMCell(encoder_hidden_units)
encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
encoder_cell, encoder_inputs_embedded,
dtype=tf.float32, time_major=True,
)
del encoder_outputs

decoder_cell=tf.contrib.rnn.LSTMCell(decoder_hidden_units)
decoder_outputs, decoder_final_state=tf.nn.dynamic_rnn(
decoder_cell, decoder_inputs_embedded,
initial_state=encoder_final_state,
dtype=tf.float32, time_major=True, scope='plane_decoder',
)

decoder_logits=tf.contrib.layers.linear(decoder_outputs, vocab_size)
decoder_prediction=tf.argmax(decoder_logits,2)

stepwise_cross_entropy=tf.nn.softmax_cross_entropy_with_logits(
labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
logits=decoder_logits,
)
loss=tf.reduce_mean(stepwise_cross_entropy)
train_op=tf.train.AdamOptimizer().minimize(loss)

sess.run(tf.global_variables_initializer())
#####GRAPH END

def int2voc(sentence_int):
    sentence_list=[]
    for word_int in sentence_int:
        word=int_to_voca[word_int]
        sentence_list.append(word)
        sentence = ' '.join(sentence_list)
    return sentence

##try
"""
batch_=corpus_int[0:4]
batch_, batch_length_=helpers.batch(batch_)
print('batch_encoded: \n'+str(batch_))
print(type(batch_))
print(batch_.shape)

din_, dlen_ =helpers.batch(np.ones(shape=(4,1),dtype=np.int32),
                           max_sequence_length=4)
print('decoder inputs: \n'+str(din_))

pred_=sess.run(decoder_prediction,
              feed_dict={
                  encoder_inputs: batch_,
                  decoder_inputs: din_,
              })
print('decoder predictions:\n'+str(pred_))
"""

def next_feed(batches,from_idx, to_idx):###input: ? -> output: dict
    batch=batches[from_idx:to_idx]
    encoder_inputs_, _ = helpers.batch(batch)
    batch1=batches[from_idx+1:to_idx+1]
    decoder_targets_, _ = helpers.batch(
    [(sequence)+[EOS] for sequence in batch1]
    )
    decoder_inputs_, _ =helpers.batch(
    [[EOS]+(sequence) for sequence in batch1]
    )
    return{
        encoder_inputs: encoder_inputs_,
        decoder_inputs: decoder_inputs_,
        decoder_targets: decoder_targets_,
    }

batch_size=512
loss_track=[]
max_batches=31
batches_in_epoch =5
try:
    for batch in range(max_batches):
        from_idx = batch*batch_size
        to_idx = from_idx + batch_size
        fd=next_feed(corpus_int, from_idx, to_idx)
        _, l =sess.run([train_op,loss], fd)
        loss_track.append(l)
        
        if batch==0 or batch%batches_in_epoch ==0:
            print('batch {}'.format(batch))
            print('minibatch loss: {}'.format(sess.run(loss,fd)))
            predict_=sess.run(decoder_prediction, fd)
            
            for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                print('sample{}:'.format(i+1))
                inp.tolist
                pred.tolist
                inp_sentence = int2voc(inp)
                print('INPUT   >   {}'.format(inp_sentence))
                pred_sentence = int2voc(pred)
                print('PREDICT >   {}'.format(pred_sentence))
                if i>+2:
                    break
            print()
except KeyboardInterrupt:
    print('training interrupted')







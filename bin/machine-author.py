import sys
import os
import time
import re
import argparse
from subprocess import call

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__))+"/../lib")
from helper import *

version = "1.1"
date = "2017-05-30"

# Definieren der Kommandozeilenparameter
parser = argparse.ArgumentParser(description='machine author trains on existent books and generates sample text',
                                 epilog='author: alexander.vogel@prozesskraft.de | version: ' + version + ' | date: ' + date)
parser.add_argument('--book', action='store', metavar='STRING', default=False,
                   help='book to train on or sample from')
parser.add_argument('--action', metavar='train|sample|list', action='store', default='sample', required=True,
                   help='choose an action. list: lists all available book files. train: training the network (RNN) on books. sample: network creates text samples')

args = parser.parse_args()

# location of books
bookDir = os.path.dirname(os.path.realpath(__file__)) + '/../books' 
log('info', 'using book dir ' + bookDir)
#bookFiles = ['goethe_faust-teil1.txt', 'goethe_faust-teil2.txt']

# location of trained models
modelDir = os.path.dirname(os.path.realpath(__file__)) + '/../trained'
log('info', 'using model dir ' + modelDir)

if(args.action == 'list'):
    print("books:")
    print("------")
    print("\n".join(os.listdir(bookDir)))
    print("======")
    print("pretrained models:")
    print("------")
    print("\n".join(os.listdir(modelDir)))
    sys.exit(0)

if(args.action == 'train' or args.action == 'sample'):
    if not args.book:
        print("sample and train need a --book")
        sys.exit(1)

## temp file for concatenated books
#textfile = 'traintext.txt' 
#textfileabs = os.getcwd() + "/" + textfile

## Merge all books in 1 file
#with open(textfile, 'w') as outfile:
#    log('info', 'concatinating books in a text file: ' + textfile)
#    for fname in args.book:
#        with open(bookDir+"/"+fname, "r", encoding='utf-8', errors='ignore') as infile:
#            log('info', '   - ' + fname)
#            outfile.write(infile.read())

# define checkpoints directory for current book selection


# load text file
log('info', 'loading text file: ' + bookDir+"/"+args.book)
with open(bookDir+"/"+args.book, 'r') as f:
    text=f.read()
    
# convert text into integers
log('info', 'converting text into integers: ' + bookDir+"/"+args.book)
vocab = set(text)
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
chars = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

log('info', 'defining hyperparameters')
batch_size = 100
num_steps = 100 
lstm_size = 1024
num_layers = 2
learning_rate = 0.001
keep_prob = 0.5

if args.action == 'sample':
    log('info', "loading checkpoint for book " + args.book)
    tf.train.get_checkpoint_state('trained/'+args.book)
#    checkpoint = "checkpoints/i960_l512_v1.599.ckpt"

    checkpoint = "trained/"+args.book+"/checkpoint.ckpt"
    
    myPrime = None
    m = re.search('^d_', args.book)
    if m:
        myPrime = "Die "
    else:
    	myPrime = "The True"
    
    log('info', 'using prime: ' + myPrime)
    
    samp = sample(checkpoint, 2000, lstm_size, vocab_to_int, int_to_vocab, vocab, prime='The True')
    print(samp)
    sys.exit(0)
    
print("example snippet as text:\n", text[5500:5600])
print("example snippet as integers:\n", chars[5500:5600])

log('info', 'split text in train and validation data')
train_x, train_y, val_x, val_y = split_data(chars, 10, 50)

log('info', 'creating model (RNN)')
model = build_rnn(len(vocab), 
                  batch_size=batch_size,
                  num_steps=num_steps,
                  learning_rate=learning_rate,
                  lstm_size=lstm_size,
                  num_layers=num_layers)

log('info', 'training')
epochs = 20
# Save every N iterations
save_every_n = 200
train_x, train_y, val_x, val_y = split_data(chars, batch_size, num_steps)

saver = tf.train.Saver(max_to_keep=100)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # Use the line below to load a checkpoint and resume training
    #saver.restore(sess, 'checkpoints/______.ckpt')
    
    n_batches = int(train_x.shape[1]/num_steps)
    iterations = n_batches * epochs
    for e in range(epochs):
        
        # Train network
        new_state = sess.run(model.initial_state)
        loss = 0
        for b, (x, y) in enumerate(get_batch([train_x, train_y], num_steps), 1):
            iteration = e*n_batches + b
            start = time.time()
            feed = {model.inputs: x,
                    model.targets: y,
                    model.keep_prob: keep_prob,
                    model.initial_state: new_state}
            batch_loss, new_state, _ = sess.run([model.cost, model.final_state, model.optimizer], 
                                                 feed_dict=feed)
            loss += batch_loss
            end = time.time()
            print('Epoch {}/{} '.format(e+1, epochs),
                  'Iteration {}/{}'.format(iteration, iterations),
                  'Training loss: {:.4f}'.format(loss/b),
                  '{:.4f} sec/batch'.format((end-start)))
        
            
            if (iteration%save_every_n == 0) or (iteration == iterations):
                # Check performance, notice dropout has been set to 1
                val_loss = []
                new_state = sess.run(model.initial_state)
                for x, y in get_batch([val_x, val_y], num_steps):
                    feed = {model.inputs: x,
                            model.targets: y,
                            model.keep_prob: 1.,
                            model.initial_state: new_state}
                    batch_loss, new_state = sess.run([model.cost, model.final_state], feed_dict=feed)
                    val_loss.append(batch_loss)

                print('Validation loss:', np.mean(val_loss),
                      'Saving checkpoint!')
                saver.save(sess, "checkpoints/i{}_l{}_v{:.3f}.ckpt".format(iteration, lstm_size, np.mean(val_loss)))



"""
Midterm assignment
Gavri Kepets and Ayden Shankman

The goal of this project is to recreate figure 2 in the paper "Distributed Representations of Words and Phrases
and their Compositionality" by Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean

In this code we used the c4 dataset which is Google's cleaned up version of the Web Crawl corpus.

Tomas Mikolov et al. “Distributed Representations of Words and Phrases and
their Compositionality”. In: Advances in Neural Information Processing Systems.
Ed. by C.J. Burges et al. Vol. 26. Curran Associates, Inc., 2013. URL:
https://proceedings.neurips.cc/paper/2013/file/
9aa42b31882ec039965f3c4923ce901b-Paper.pdf
"""

from __future__ import print_function
import tqdm
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
import os
import pickle
import json
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
from os.path import exists
import matplotlib.pyplot as plt
from tensorflow import keras  
from sklearn.decomposition import PCA

tokenizer = Tokenizer()
sentences = []
num_ns = 4 # negative sampling size
window_size = 2
# Words to test embedding
countries_and_cities = ["France","Paris", "Russia", "Moscow", "Italy", "Rome", "England", "America", "London", "Japan", "Tokyo", "Canada", "Toronto", "Spain", 'Switzerland', "Warsaw", "Ireland"]
topics = ['oil', 'gas','fuel', 'city', 'town', 'village', 'country', 'angry', 'sad', 'happy', 'cry']


# Creating/Loading Sentences and Tokenizer
if exists('c4_sentences.pickle'):
    with open('c4_sentences.pickle', 'rb') as handle:
        sentences = pickle.load(handle)
    with open('c4_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
else:
    directory = 'data/'
    tokenizer = Tokenizer()

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            with open (f, encoding="utf8") as fin:
                print("Apending Sentences...")
                count = 0
                for line in tqdm.tqdm(fin):
                    if(count == 20000): # When we use more than 20000 sentences we run out of ram (32GB)
                        break
                    j = line[9:line.find("\"timestamp\":")-2]
                    tokens = word_tokenize(j)
                    sentences.append(tokens)
                    count += 1
   
    print("Tokenizing Words...")
    tokenizer.fit_on_texts(tqdm.tqdm(sentences))

    print("Dumping tokenizer into pickle...")
    with open('c4_tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Dumping sentences into pickle...")
    with open('c4_sentences.pickle', 'wb') as handle:
        pickle.dump(sentences, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

sentenceCount = len(sentences)

print("Number of words in Big Tokenizer: " + str(len(tokenizer.word_index))) # over 300,000 unique words
print("Number of sentences trained on: " + str(sentenceCount)) # 20,000 samples
vocab_size = len(tokenizer.word_index) + 1

# Vectorizing Sentences
sequences = []
for i in tqdm.tqdm(range(sentenceCount)):
    sequence_unprocessed = np.array(tokenizer.texts_to_sequences(sentences[i]))
    sequence = []
    for i in range(len(sequence_unprocessed)):
        sequence.append(sequence_unprocessed[i][0] if len(sequence_unprocessed[i]) else 0)
    sequences.append(sequence)

# Function to get PCA projection graphs
def get_PCA(words, tokenizer, model_name, graph_title, file_name):

    word2vec = keras.models.load_model(model_name)
    weights = word2vec.get_layer('w2v_embedding').get_weights()[0] # get trainable weights
    allwords_i = np.arange(len(tokenizer.word_index)+1)
    seqs = tokenizer.texts_to_sequences(words)
    word_i = np.array(seqs).flatten()
    
    pca = PCA().fit_transform(weights[allwords_i])[:, :2] # Project embedding size dimenions down to 2 dimensions
    
    # Plot the words and their locations on the PCA graph
    plt.figure()
    plt.scatter(np.array(pca)[word_i, 0], np.array(pca)[word_i, 1])
    plt.title(graph_title)
    for i, word in enumerate(words):
        plt.text(np.array(pca)[word_i[i], 0], np.array(pca)[word_i[i], 1], word)
    plt.savefig(file_name)


def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
  print("GENERATING NEW DATA...")

  # Elements of each training example are appended to these lists.
  targets, contexts, labels = [], [], []

  # Build the sampling table for `vocab_size` tokens.
  sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

  # Iterate over all sequences (sentences) in the dataset.
  for sequence in tqdm.tqdm(sequences):

    # Generate positive skip-gram pairs for a sequence (sentence).
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          sequence,
          vocabulary_size=vocab_size,
          sampling_table=sampling_table,
          window_size=window_size,
          negative_samples=0)

    # Iterate over each positive skip-gram pair to produce training examples
    # with a positive context word and negative samples.
    for target_word, context_word in positive_skip_grams:
      context_class = tf.expand_dims(
          tf.constant([context_word], dtype="int64"), 1)

      negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
          true_classes=context_class,
          num_true=1,
          num_sampled=num_ns,
          unique=True,
          range_max=vocab_size,
          seed=seed,
          name="negative_sampling")

      # Build context and label vectors (for one target word)
      negative_sampling_candidates = tf.expand_dims(
          negative_sampling_candidates, 1)

      context = tf.concat([context_class, negative_sampling_candidates], 0)
      label = tf.constant([1] + [0]*num_ns, dtype="int64")

      # Append each element from the training example to global lists.
      targets.append(target_word)
      contexts.append(context)
      labels.append(label)

  return targets, contexts, labels

class Word2Vec(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim):
    super(Word2Vec, self).__init__()
    self.target_embedding = layers.Embedding(vocab_size,
                                      embedding_dim,
                                      input_length=1,
                                      name="w2v_embedding")
    self.context_embedding = layers.Embedding(vocab_size,
                                       embedding_dim,
                                       input_length=num_ns+1)

  def call(self, pair):
    target, context = pair
    # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+
    # context: (batch, context)
    if len(target.shape) == 2:
      target = tf.squeeze(target, axis=1)
    # target: (batch,)
    word_emb = self.target_embedding(target)
    # word_emb: (batch, embed)
    context_emb = self.context_embedding(context)
    # context_emb: (batch, context, embed)
    dots = tf.einsum('be,bce->bc', word_emb, context_emb)
    # dots: (batch, context)
    return dots

def custom_loss(x_logit, y_true):
      return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)

# Generate training data if it doesn't already exist
targets, contexts, labels = [], [], []
if not exists("./data_gen_c4/targets.pickle"):
    targets, contexts, labels = generate_training_data(sequences, window_size, num_ns, vocab_size, 42)
    with open("./data_gen_c4/targets.pickle", 'wb') as handle:
        pickle.dump(targets, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("./data_gen_c4/contexts.pickle", 'wb') as handle:
        pickle.dump(contexts, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("./data_gen_c4/labels.pickle", 'wb') as handle:
        pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open("./data_gen_c4/targets.pickle", 'rb') as handle:
        targets = pickle.load(handle)
    with open("./data_gen_c4/contexts.pickle", 'rb') as handle:
        contexts = pickle.load(handle)
    with open("./data_gen_c4/labels.pickle", 'rb') as handle:
        labels = pickle.load(handle)
    

targets = np.array(targets)
contexts = np.array(contexts)[:,:,0]
labels = np.array(labels)

BATCH_SIZE = 1024
BUFFER_SIZE = 100000
dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

word2vec = 0
emb_size = 32

if not exists("model_c4_32"):
    word2vec = Word2Vec(vocab_size, emb_size)
    word2vec.compile(optimizer='adam',
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    history = word2vec.fit(dataset, epochs=30)
    word2vec.save('model_c4_32')

    plt.figure()
    plt.plot(history.history['loss'])
    plt.title("Loss for C4 Corpus with Embedding Size 32")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig('loss_c4.png')

    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.title("Training Accuracy for C4 Corpus with Embedding Size 32")
    plt.ylabel('Acc')
    plt.xlabel('Epoch')
    plt.savefig('acc_c4.png')

get_PCA(countries_and_cities, tokenizer, "model_c4_32", ("PCA of Countries/Cities for C4 Corpus with Embedding Size " + str(emb_size)), ('PCA_model_c4_32_cities.png'))
get_PCA(topics, tokenizer, "model_c4_32", ("PCA of Various Topics for C4 Corpus with Embedding Size " + str(emb_size)), ('PCA_model_c4_32_topics.png'))



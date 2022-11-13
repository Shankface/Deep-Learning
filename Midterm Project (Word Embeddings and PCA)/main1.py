#!/bin/env python3.8

"""
Midterm assignment
Gavri Kepets and Ayden Shankman

The goal of this project is to recreate figure 2 in the paper "Distributed Representations of Words and Phrases
and their Compositionality" by Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean

In this code we used the Brown Corpus from NLTK which consists of one million words of American English texts printed in 1961.



Tomas Mikolov et al. “Distributed Representations of Words and Phrases and
their Compositionality”. In: Advances in Neural Information Processing Systems.
Ed. by C.J. Burges et al. Vol. 26. Curran Associates, Inc., 2013. URL:
https://proceedings.neurips.cc/paper/2013/file/
9aa42b31882ec039965f3c4923ce901b-Paper.pdf
"""

# import nltk
from tensorflow.keras import layers
import tensorflow as tf
from nltk.corpus import brown
from tensorflow.keras.preprocessing.text import Tokenizer
import tqdm
import numpy as np
import pickle
from os.path import exists
import matplotlib.pyplot as plt
from tensorflow import keras  
from sklearn.decomposition import PCA

sentences = brown.sents()
tokenizer = Tokenizer()
sentenceCount = len(sentences)
num_ns = 10 # number of negative samples per word
window_size = 2
# Words to test embedding
countries_and_cities = ["France","Paris", "Russia", "Moscow", "Italy", "Rome", "England", "America", "London", "Japan", "Tokyo", "Canada", "Toronto", "Spain", 'Switzerland', "Warsaw", "Ireland"]
topics = ['oil', 'gas','fuel', 'city', 'town', 'village', 'country', 'angry', 'sad', 'happy', 'cry']
    

if exists('brown_tokenizer.pickle'):
    with open('brown_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
else:
    for sentence in tqdm.tqdm(sentences[0:sentenceCount]):
        tokenizer.fit_on_texts(sentence)

    with open('brown_tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Number of words in Brown Tokenizer: " + str(len(tokenizer.word_index))) # over 47,000 unique words
print("Number of sentences trained on: " + str(sentenceCount)) # 57,000 sentences
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

# Function to generate skip_grams with negative sampling
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
if not exists("./data_gen_brown/targets.pickle"):
    targets, contexts, labels = generate_training_data(sequences, window_size, num_ns, vocab_size, 42)
    with open("./data_gen_brown/targets.pickle", 'wb') as handle:
        pickle.dump(targets, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("./data_gen_brown/contexts.pickle", 'wb') as handle:
        pickle.dump(contexts, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("./data_gen_brown/labels.pickle", 'wb') as handle:
        pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open("./data_gen_brown/targets.pickle", 'rb') as handle:
        targets = pickle.load(handle)
    with open("./data_gen_brown/contexts.pickle", 'rb') as handle:
        contexts = pickle.load(handle)
    with open("./data_gen_brown/labels.pickle", 'rb') as handle:
        labels = pickle.load(handle)
    
targets = np.array(targets)
contexts = np.array(contexts)[:,:,0]
labels = np.array(labels)

BATCH_SIZE = 1024
BUFFER_SIZE = 100000
dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

embedding_layers = [8, 16, 32, 64, 128, 256, 512]
word2vec = 0
#epochs = [200, 150, 100, 50, 50, 40, 40]
epochs = [1,1,1,1,1,1,1]

# Training models with increasing embedding size and decreasing epochs
for i, emb_size in enumerate(embedding_layers):

    print("Training Model with embedding size " + str(emb_size))
    word2vec = Word2Vec(vocab_size, emb_size)
    word2vec.compile(optimizer='adam',
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
        
    word2vec.fit(dataset, epochs=epochs[i])
    history = word2vec.fit(dataset, epochs=epochs[i])
    model_name = 'model2_' + str(emb_size)
    word2vec.save(model_name)

    plt.figure()
    plt.plot(history.history['loss'])
    plt.title("Loss for Brown Corpus with Embedding Size " + str(emb_size))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig('loss2_' + str(emb_size) + '.png')

    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.title("Training Accuracy for Brown Corpus with Embedding Size " + str(emb_size))
    plt.ylabel('Acc')
    plt.xlabel('Epoch')
    plt.savefig('acc2_' + str(emb_size) + '.png')

    get_PCA(countries_and_cities, tokenizer, model_name, ("PCA of Countries/Cities for Brown Corpus with Embedding Size " + str(emb_size)), ('PCA_' + model_name + '_cities.png'))
    get_PCA(topics, tokenizer, model_name, ("PCA of Various Topics for Brown Corpus with Embedding Size " + str(emb_size)), ('PCA_' + model_name + '_topics.png'))


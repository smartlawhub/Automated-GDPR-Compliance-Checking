# UTIL FUNCTIONS FOR

import numpy as np
from os.path import join

def get_emb_weights(embeddings=None, vocab=None, emb_dim=300, oov_random=True, **kwargs):
    """

    This function returns a matrix containing the weights that will be used as pretrained embeddings. It will read 
    weights_matrix.pkl file as long as it exists. This will make the code much faster. 

    Args:
        dictionary: dictionary, dictionary containing a word2idx of all the words present in the dataset.
        word2vector: dictionary, the keys are the words and the values are the embeddings.
        dims: integer, dimensionality of the embeddings.
        read: boolean, variable that allows us to decide wether to read from pre-processed files or not.
    Returns:
        weights_matrix: np.array, matrix containing all the embeddings.

    """
    
    print("Generating embedding matrix ...")
    # We add 1 to include the "None" value in the word2vector
    matrix_len = len(vocab) + 1

    # Instantiate our weights matrix
    weights_matrix = np.zeros((matrix_len, emb_dim))
    oov_words = 0
    for word, i in vocab.items():
        try:
            weights_matrix[i] = embeddings[word]
        except KeyError:
            # If OOV, generate random vector for that word
            if oov_random: weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim,))
            oov_words += 1
    if oov_words != 0:
        print(f"Some words were missing in the pretrained embedding. {oov_words} words were not found.")

    return weights_matrix

from torch import nn
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def extract_from_model_file(words, word2idx, vectors, path, file):
    '''
    Extract the Word2Vec model
    '''
    idx=1
    with open(join(path, file), encoding="utf8") as fast_text_file:
        for line in fast_text_file:
            """if idx == 158090:"""
            split_line = line.split()
            word = split_line[0]
            words.append(word)
            word2idx[word] = idx
            vector = np.array(split_line[1:]).astype(np.float)
            vectors.append(vector)
            idx += 1
    return words, word2idx, vectors

def get_pretrained_embs(path, emb_dim):
    """
    This functions gets the pretrained embedding vectors.
    
    Args:
        path: string, path to the folder containing the glove embeddings
        emb_dim: integer, embeddings dimensionality to use.
    Returns:
        word2vector: dictionary, the keys are the words an the values are the embeddings associated with that word.
    """
    
    print("Extracting pretrained embeddings ...")
    words = [None]
    word2idx = {None: 0}
    vectors = [np.zeros(emb_dim)]
    words, word2idx, vectors = extract_from_model_file(
        words, word2idx, vectors, path, '.vec')

    word2vector = {w: vectors[word2idx[w]] for w in words}
    return word2vector
# DATA PROCESSING FUNCTIONS

import codecs
import glob
import os
from os import listdir
from os.path import isfile, join
import nltk
import numpy as np
import pandas as pd
import torch

def get_vocab(data_path):
    """
    Checks all the files in filespath and returns a set of all the words found in the files. 
    The function will ignore all the folders inside filespath automatically.
    Tokenizes and lowercases each line in the data files
    
    Args:
        data_path: string, path to the folder with all the files containing the words that we want to extract.
    Returns:
        dictionary: set, set containing all the different words found in the files. 
    """

    print("Generating vocab ...")
    dictionary = set()

    # Gather CSVs in the data folder
    files = [join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f))]
    files = glob.glob(os.path.join(data_path/'**/', '*.csv'))
    files = [file for file in files if ".keep" not in file]

    # Lower case and tokenize each line of the csv
    for f in files:
        opened_file = codecs.open(f, 'r', encoding='utf-8', errors='ignore')
        for i, line in enumerate(opened_file):
            a = line.split('","')
            a[1] = map(str.lower, set(nltk.word_tokenize(a[1])))

            # Append to dictionary
            dictionary = dictionary.union(a[1])

    # sort dictionary
    dictionary = sorted(dictionary)
    
    # Append everything to our vocab
    vocab = {None: 0}
    for i, word in enumerate(dictionary, 1):
        vocab[word] = i
    
    return vocab


def tokenize_sentence(sentence, vocab, lower_case=True):
    """ 
    Transforms a sentence into a list of integers. No integer will be appended if the token is not present in vocab.
    
    Args:
        sentence: string, sentence that we want to serialize.
        vocab: dictionary, dictionary with words as keys and indexes as values.
        lower_case: boolean, turns all words in the sentence to lower case. Useful if vocab 
        doesn't support upper case words.
    Returns: 
        s_sentence: list, list containing the indexes of the words present in the sentence. 
        s_sentence stands for serialized sentence.
        
    """
    s_sentence = []
    not_found = 0
    
    # Tokenize & Lowercase
    if lower_case: tokens = map(str.lower, nltk.word_tokenize(sentence))
    else: tokens = nltk.word_tokenize(sentence)

    for token in tokens:
        try:
            s_sentence.append(vocab[token])
        except KeyError:
            not_found += 1
            print("Warning: At least one token is not present in the vocab dict. For instance: " + token +
                ". Not found: " + str(not_found))
    return s_sentence


def process_dataset(df, vocab, include_parent=True, attr_model=False):
    """
    1. This function process all the privacy policy data and tokenizes all the segments 
    2. It also transforms all the labels into a list of 0s except in the positions associated with the labels in which we will find 1s
    where we will find a 1. 
    
    Args:
        df: dataframe generated from privacy policy CSV files in data_folder
        vocab: dictionary the keys are the words and the values the index where we can find the vector in 
        weights_matrix.

    Returns:
        sentence_matrices: list, a list of lists of lists containing tokenized segments  
        sentence_matrices[i][j][k] -> "i" is for the file, "j" for the line and "k" for the token. 
        labels_matrices: list, a list of lists of lists containing the labels of the dataset. labels_matrices[i][j][k] ->
        "i" is for the file, "j" for the line and "k" for the boolean variable specifying 
        the presence of the a label.
        
    """

    print("Processing dataset ...")
        
    num_records = len(df)
    num_labels = len(df["label"].iloc[0])
    print(f'Num of unique segments: {num_records}')

    # Define empty arrays for sentences and labels
    sentence_matrices = np.zeros(num_records, dtype='object')
    labels_matrices = np.zeros((num_records, num_labels))

    if include_parent:
        num_labels_parent = len(df["label_parent"].iloc[0])
        parent_labels_matrices = np.zeros((num_records, num_labels_parent))
    else: 
        parent_labels_matrices = np.zeros((num_records, num_labels))

    for index, row in df.iterrows():
        sentence = row["segment"]
        label = row["label"]

        # Tokenize the sentence
        sentence_matrices[index] = tokenize_sentence(sentence, vocab)

        # Add label
        labels_matrices[index] = label
        if include_parent:
            label_parent = row["label_parent"]
            parent_labels_matrices[index] = label_parent
        
    if include_parent or attr_model:
         return sentence_matrices, labels_matrices, parent_labels_matrices
    else: return sentence_matrices, labels_matrices


def create_multi_labels(single_attr_df, child_labels_dict, categories_dict, has_parent=True):
    '''
    Create multi labels for child, with parent labels
    '''
    
    child_grouped =  single_attr_df.groupby(['segment'])['child_label'].unique()

    # TURN INTO DATAFRAME
    tmp_single_attr_df = pd.DataFrame(child_grouped)
    tmp_single_attr_df.reset_index(inplace=True)
    
    # TURN INTO DATAFRAME
    if has_parent:
        parent_grouped =  single_attr_df.groupby(['segment'])['parent_label'].unique()
        parent_df = pd.DataFrame(parent_grouped)
        parent_df.reset_index(inplace=True)

        single_attr_df = pd.concat([tmp_single_attr_df,parent_df],1)
        single_attr_df = single_attr_df.iloc[:,[0,1,3]].copy()

    # ADD CHILD LABELS
    labels_ls=[]
    for child_label in single_attr_df.child_label:
        idx_ls=[]
        if isinstance(child_label, list) or isinstance(child_label,np.ndarray):
            for label in child_label:
                idx_ls.append(child_labels_dict[label])   
        else:
            idx_ls.append(child_labels_dict[child_label])   

        target = [1] * len(idx_ls)
        labels = [0] * len(child_labels_dict.keys())
        for x,y in zip(idx_ls,target):
            labels[x] = y
        labels_ls.append(labels)

    single_attr_df['label'] = labels_ls

    # ADD PARENT LABELS
    if has_parent:
        labels_ls=[]
        for parent_label in single_attr_df.parent_label:
            idx_ls=[]
            if isinstance(parent_label, list) or isinstance(parent_label,np.ndarray):
                for label in parent_label:
                    idx_ls.append(categories_dict[label])   
            else:
                idx_ls.append(child_labels_dict[parent_label])   
            target = [1] * len(idx_ls)
            labels = [0] * len(categories_dict.keys())
            for x,y in zip(idx_ls,target):
                labels[x] = y
            labels_ls.append(labels)

        single_attr_df['label_parent'] = labels_ls
    
    return single_attr_df


def stack_segments(segments, clearance = 2):
    segments_len = map(len, segments)
    max_len = max(segments_len)
    segments_list = []
    output_len = max_len + clearance * 2
    for i, segment in enumerate(segments):
        segment_array = np.array(segment)

        zeros_to_prepend = int((output_len - len(segment_array))/2)

        zeros_to_append = output_len - len(segment_array) - zeros_to_prepend
        resized_array = np.append(np.zeros(zeros_to_prepend), segment_array)
        resized_array = np.append(resized_array, np.zeros(zeros_to_append))
        segments_list.append(torch.tensor(resized_array, dtype = torch.int64))
        segments_tensor = torch.stack(segments_list).unsqueeze(1)
    return segments_tensor 

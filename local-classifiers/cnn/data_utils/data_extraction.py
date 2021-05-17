# DATA EXTRACTION FUNCTIONS, EXTRACTS DATA FROM DATA FILES

import pandas as pd
import numpy as np
import glob
import os
from os import listdir
from os.path import isfile, join
import pickle 

def unpickle_dataset(path):
    
    with open(path, "rb") as dataset_file:

        dataset = pickle.load(dataset_file)

        return dataset


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def label_to_vector(label, dictionary, length):
    """
    Returns a vector representing the label passed as an input, e.g. [0,1,0,0]
    
    Args:
        label: string, label that we want to transform into a vector.
        dictionary: dictionary, dictionary with the labels as the keys and indexes as the values.
    Returns:
        vector: np.array, 1-D array of lenght 12.
        
    """

    vector = np.zeros((length))
    try:
        index = dictionary[label]
        vector[index] = 1
    except KeyError:
        vector = np.zeros((length))
    return vector

def load_category_data(raw_data_path):
    files = [join(raw_data_path, f) for f in listdir(raw_data_path) if isfile(join(raw_data_path, f))]
    files = glob.glob(os.path.join(raw_data_path, '*.csv'))
    files = [file for file in files if ".keep" not in file]
    
    names=['idx', 'segment', 'category']
    raw_df = pd.DataFrame(columns=names)
    for f in files:
        h = pd.read_csv(f, header=None, names=names)
        raw_df = pd.concat([raw_df, h], axis=0)
    
    categories = sorted(raw_df.category.unique().tolist())
    categories_dict = {v:i for i,v in enumerate(categories)}
    #len(categories), categories_dict
    
    data = raw_df.copy()

    all_results = pd.DataFrame({'label': [], 'segment': []})
    colnames = ["idx", "segment", "label"]

    # Add label to dataframe
    data['label'] = data['category'].apply(lambda x: label_to_vector(
        x, categories_dict, len(categories_dict)))

    all_results = data.drop_duplicates(subset=['segment','category'])

    all_results = all_results.fillna("None")
    all_results = all_results[all_results["segment"] != "None"]

    all_results.reset_index(drop=True, inplace=True)
    return all_results, categories_dict


def load_attr_cat_data(data_path, attribute_folders, categories_dict=None, include_parent=True,model=True):
    '''
        data_path : Path to a folder of attribute folders
        categories_dict : dictionary of parent labels
    '''


    # Preprare dataframe
    attr_cat_df = pd.DataFrame({'child_label': [], 'segment': []})
    
    # Read all CSVs as dataframes and append to master dataframe
    for folder in attribute_folders:
        print(f'extracting from {folder} ...')
        # Read all CSVs in data_folder
        if model:
            files = glob.glob(os.path.join(data_path/folder, '*.csv'))
        else:
            files = glob.glob(os.path.join(data_path/folder/'**/', '*.csv'))
        files = [file for file in files if ".keep" not in file]
        
        for f in files:
            # Open datafile as dataframe
            data = pd.read_csv(f, header=None) #, names=colnames)
            if isinstance(data.iloc[0,0].item(), int):
                colnames = ["idx", "segment", "child_label"]
                if len(data.columns)==4:
                    colnames.append("parent_label")
            else:
                colnames = ["segment", "child_label"]
                if len(data.columns)==3:
                    colnames.append("parent_label")
            
            data.columns = colnames
            
            data['attribute'] = str(folder)
            attr_cat_df = pd.concat([attr_cat_df, data])

    child_labels = sorted(attr_cat_df.child_label.unique().tolist())
    child_labels_dict = {v:i for i,v in enumerate(child_labels)}

    # Add label to dataframe
    attr_cat_df['label'] = attr_cat_df['child_label'].apply(lambda x: label_to_vector(
        x, child_labels_dict, len(child_labels_dict)))

    if include_parent:
        assert("parent_label" in attr_cat_df.columns), "'parent_information' set to True but parent_label not found, make sure your attribute data has a parent label column"
               
        attr_cat_df['label_parent'] = attr_cat_df['parent_label'].apply(
            lambda x: label_to_vector(x, categories_dict, len(categories_dict)))

    attr_cat_df = attr_cat_df.fillna("None")
    attr_cat_df = attr_cat_df[attr_cat_df["segment"] != "None"]
    if include_parent:
        attr_cat_df.drop_duplicates(subset=['segment','child_label','parent_label'])
    else:
        attr_cat_df.drop_duplicates(subset=['segment','child_label'])
    attr_cat_df.reset_index(drop=True, inplace=True)

    return attr_cat_df, child_labels_dict

def load_data(data_folder, child_labels_dict, parent_labels_dict=None, include_parent=True):
    
    # Read all CSVs in data_folder
    files = glob.glob(os.path.join(data_folder, '*.csv'))
    files = [file for file in files if ".keep" not in file]

    # Preprare dataframe
    all_results = pd.DataFrame({'label': [], 'segment': []})
    colnames = ["idx", "segment", "label"]
    if include_parent:
        colnames.append("label_parent")
    
    for f in files:
        # Open datafile as dataframe
        data = pd.read_csv(f, names=colnames)
        
        # Add label to dataframe
        data['label'] = data['label'].apply(lambda x: label_to_vector(
            x, child_labels_dict, len(child_labels_dict)))

        if include_parent:
            data['label_parent'] = data['label_parent'].apply(
                lambda x: label_to_vector(x, parent_labels_dict, len(parent_labels_dict)))

        if include_parent:
            labels_data = data[['idx', 'label', 'label_parent']]
        else:
            labels_data = data[['idx', 'label']]

        labels = labels_data.groupby("idx").sum()

        segments = data[['idx', 'segment']].set_index(
            'idx').drop_duplicates()

        result = pd.merge(labels, segments, left_index=True, right_index=True)

        all_results = pd.concat([all_results, result])
        all_results = all_results.fillna("None")
        all_results = all_results[all_results["segment"] != "None"]

    all_results.reset_index(drop=True, inplace=True)

    return all_results

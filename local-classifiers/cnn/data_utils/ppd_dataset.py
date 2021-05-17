### DATESET FILE
import numpy as np
import torch
from torch import tensor
from torch.utils.data import Dataset
from random import sample

class PrivacyPoliciesDataset(Dataset):
    def __init__(self, segments_array, labels_list, labels, parent_information=None):
        self.segments_array = segments_array
        self.labels_tensor = tensor(labels_list)
        self.labels = labels
        self.is_parent = False
        if parent_information is not None:
            if not isinstance(parent_information, dict):
                raise ValueError('The argument parent_information must be a dictionary.')
            self.parent_labels_tensor = tensor(parent_information['labels_list'])
            self.parent_labels = parent_information['labels']
            self.is_parent = True
        
    def __len__(self):
        if self.segments_array.shape[0] == self.labels_tensor.shape[0]: 
            return self.segments_array.shape[0]
        else:
            print("Warning: number of segments don't match number of annotations")
            return self.segments_array.shape[0]     
    
    def __getitem__(self, idx):
        segment = self.segments_array[idx]
        label = self.labels_tensor[idx]
        if self.is_parent:
            label_parent = self.parent_labels_tensor[idx]
            return (segment, label, label_parent)
        return (segment, label)
    
    
    def train_test_split(self, ratio = 0.2):
        """
        This function randomly splits the dataset in two parts using the split ratio provided

        Args:
            dataset: torch.utils.data.Dataset, dataset containing the data to split
            ratio: double, test data percentage 
        Returns:
            train_dataset: torch.utils.data.Dataset, dataset with length = len(dataset) * (1 - ratio)
            test_dataset: torch.utils.data.Dataset, dataset with length = len(dataset) * ratio 

        """
        
        # Get labels from PrivacyPoliciesDataset class
        labels = self.labels
        
        # Define the number of samples in our test set
        num_test_samples = int(ratio * len(self))
        
        # get random samples for train and test idxes
        test_dataset_idx_set = set(sample(range(len(self)), num_test_samples))
        test_dataset_idx_tensor = tensor(list(test_dataset_idx_set))
        train_dataset_idx_tensor = tensor(list(set(range(len(self))).difference(test_dataset_idx_set)))
    
        # Return data for these indxes
        test_dataset_data = self[test_dataset_idx_tensor]
        train_dataset_data = self[train_dataset_idx_tensor]
        
        if self.is_parent:
            # Return Datasets with Parent Information also
            test_dataset_parent = dict(labels_list=test_dataset_data[2], labels=self.parent_labels)
            test_dataset = PrivacyPoliciesDataset(
                test_dataset_data[0], test_dataset_data[1], labels, test_dataset_parent)
            
            train_dataset_parent = dict(labels_list=train_dataset_data[2], labels=self.parent_labels)
            train_dataset = PrivacyPoliciesDataset(
                train_dataset_data[0], train_dataset_data[1], labels, train_dataset_parent)
        else:
            # Return Datasets without parent information
            test_dataset = PrivacyPoliciesDataset(test_dataset_data[0], test_dataset_data[1], labels)
            train_dataset = PrivacyPoliciesDataset(train_dataset_data[0], train_dataset_data[1], labels)

        return train_dataset, test_dataset
    

    def pickle_dataset(self, path):

        import pickle

        with open(path, "wb") as dataset_file:

            pickle.dump(self, dataset_file)
            
    def labels_stats(self):
        
        p_labels =  self.labels_tensor.sum(0)
        
        total_labels = int(p_labels.sum())
        
        num_segments = len(self)
        
        print('Num of segments: {}'.format(num_segments))
        
        print('Num of labels: {}'.format(total_labels))
        
        print('Percentages with respect to number of labels ... ')
        
        for label, idx in self.labels.items():
            
            num_p = int(p_labels[idx])
            
            pct = 100 * p_labels[idx] / total_labels
            print('{}. {} : {} ({}%)'.format(idx, label, num_p, pct))
    
    @staticmethod
    def unpickle_dataset(path):
    
        import pickle

        with open(path, "rb") as dataset_file:

            dataset = pickle.load(dataset_file)

            return dataset

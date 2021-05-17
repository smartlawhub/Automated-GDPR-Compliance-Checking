import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

class AttributeValueModel(nn.Module):
    """
    Convolutional Neural Model used for training the models. The total number of kernels that will be used in this
    CNN is Co * len(Ks).

    Args:
        weights_matrix: numpy.ndarray, the shape of this n-dimensional array must be (words, dims) were words is
        the number of words in the vocabulary and dims is the dimensionality of the word embeddings.
        Co (number of filters): integer, stands for channels out and it is the number of kernels of the same size that will be used.
        Hu: integer, stands for number of hidden units in the hidden layer.
        num_classes: integer, number of units in the last layer (number of classes)
        Ks: list, list of integers specifying the size of the kernels to be used.

    """

    def __init__(self, weights_matrix, num_classes, num_categories, Co, Hu, Ks, drop=0.5,name='generic'):
        super(AttributeValueModel, self).__init__()
        
        self.weights_matrix = weights_matrix
        self.emb_dim = weights_matrix.size()[1]
        self.num_categories = num_categories

        self.Co, self.Hu, self.Ks = Co, Hu, Ks
        self.name, self.padding_index, self.output, self.cnn_name = name, 0, dict(), name
        self.num_classes = num_classes
        self.concat_length = 2 * self.Hu[0]
        units = [(self.Co * len(self.Ks))+num_categories] + self.Hu

        self.relu = nn.ReLU() 
        self.dropout = nn.Dropout(p=drop) 
        self.sigmoid = nn.Sigmoid()
        
        self.embedding = nn.Embedding.from_pretrained(weights_matrix, padding_idx=0, freeze=True)
#         sz = weights_matrix.size()
#         self.embedding = nn.Embedding(sz[0], sz[1], padding_idx=0)
        
        self.convolutions_child = nn.ModuleList(
                [nn.Conv2d(1, self.Co, (k, int(self.emb_dim))) for k in self.Ks]) 

        self.linear_layers_child = nn.ModuleList([
            nn.Linear(units[k], units[k + 1]) for k in range(len(units) - 1)])

        self.linear_last_child = nn.Linear(self.Hu[-1], self.num_classes)

    def forward(self, x, parent_labels=None):
        x_emb = self.embedding(x)
                   
        x_child = [self.relu(conv(x_emb)).squeeze(3) for conv in self.convolutions_child]
        x_child = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x_child]
        
        if parent_labels is not None:
            parent_labels = parent_labels/len(parent_labels)  
            x_child = torch.cat((x_child[0].float(), parent_labels.float()),dim=1)
        else:
            zs = torch.zeros([x_child[0].size()[0], self.num_categories]).float()
            zs = zs.to(x_child[0].device)
            x_child = torch.cat((x_child[0].float(), zs), dim=1)
            
        for i,linear in enumerate(self.linear_layers_child):
            x_child = linear(x_child)
            x_child = self.relu(x_child)
        x_child = self.dropout(x_child)
        x_child_last = self.linear_last_child(x_child)
        return self.sigmoid(x_child_last)


class ParentCategoryModel(nn.Module):
    """
    Convolutional Neural Model used for training the models. The total number of kernels that will be used in this
    CNN is Co * len(Ks).

    Args:
        weights_matrix: numpy.ndarray, the shape of this n-dimensional array must be (words, dims) were words is
        the number of words in the vocabulary and dims is the dimensionality of the word embeddings.
        Co (number of filters): integer, stands for channels out and it is the number of kernels of the same size that will be used.
        Hu: integer, stands for number of hidden units in the hidden layer.
        num_classes: integer, number of units in the last layer (number of classes)
        Ks: list, list of integers specifying the size of the kernels to be used.

    """

    def __init__(self, weights_matrix, num_classes, Co, Hu, Ks, drop=0.5,name='generic'):
        super(ParentCategoryModel, self).__init__()
        
        self.weights_matrix = weights_matrix
        self.emb_dim = weights_matrix.size()[1]

        self.Co, self.Hu, self.Ks = Co, Hu, Ks
        self.name, self.padding_index, self.output, self.cnn_name = name, 0, dict(), name
        self.num_classes = num_classes
        self.concat_length = 2 * self.Hu[0]
        units = [self.Co * len(self.Ks)] + self.Hu

        self.relu = nn.ReLU() 
        self.dropout = nn.Dropout(p=drop) 
        self.sigmoid = nn.Sigmoid()
        
        self.embedding = nn.Embedding.from_pretrained(weights_matrix, padding_idx=0, freeze=True)
    
        self.convolutions_parent = nn.ModuleList(
                [nn.Conv2d(1, self.Co, (k, int(self.emb_dim))) for k in self.Ks]) 

        self.linear_layers_parent = nn.ModuleList([
            nn.Linear(units[k], units[k + 1]) for k in range(len(units) - 1)])

        self.linear_last_parent = nn.Linear(self.Hu[-1], self.num_classes)

    def forward(self, x):
        x_emb = self.embedding(x)
                   
        x_parent = [self.relu(conv(x_emb)).squeeze(3) for conv in self.convolutions_parent]
        x_parent = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x_parent]
        x_parent = torch.cat(x_parent, 1)

        for i,linear in enumerate(self.linear_layers_parent):
            x_parent = linear(x_parent)
            x_parent = self.relu(x_parent)
            
        x_parent = self.dropout(x_parent)
        x_parent_last = self.linear_last_parent(x_parent)
        return self.sigmoid(x_parent_last)

class HierarchicalModel(nn.Module):
    """
    Convolutional Neural Model used for training the models. The total number of kernels that will be used in this
    CNN is Co * len(Ks).

    Args:
        weights_matrix: numpy.ndarray, the shape of this n-dimensional array must be (words, dims) were words is
        the number of words in the vocabulary and dims is the dimensionality of the word embeddings.
        Co (number of filters): integer, stands for channels out and it is the number of kernels of the same size that will be used.
        Hu: integer, stands for number of hidden units in the hidden layer.
        num_classes: integer, number of units in the last layer (number of classes)
        Ks: list, list of integers specifying the size of the kernels to be used.

    """

    def __init__(self, training_mode, child_information, Co, Hu, Ks, parent_information=None, name='generic'):
        super(HierarchicalModel, self).__init__()
        
#         # set device
#         if torch.cuda.is_available(): self.device = 'cuda'
#         else: self.device = 'cpu'

        self.Co, self.Hu, self.Ks = Co, Hu, Ks
        self.name, self.padding_index, self.output, self.cnn_name = name, 0, dict(), name
            
        if parent_information is not None:
            if not isinstance(parent_information, dict):
                raise ValueError('The argument parent_information must be a dictionary.')
            self.parent_information = True
            
            self.vocab_size_parent = parent_information['vocab_size']
            self.emb_dim_parent = parent_information['emb_dim']
            self.C_parent = parent_information['C']
        else: self.parent_information = False

        if not isinstance(child_information, dict):
            raise ValueError('The argument child_information must be a dictionary.')
        
        self.vocab_size_child = child_information['vocab_size_child']
        self.emb_dim_child = child_information['emb_dim']
        self.C_child = child_information['C']
        
        if self.parent_information: self.concat_length = 2 * self.Hu[0]
        else: self.concat_length = self.Hu[0]
        
        units = [self.Co * len(self.Ks)] + self.Hu
        units_combination = [self.concat_length] + self.Hu

        self.relu = nn.ReLU() 
        self.dropout = nn.Dropout(p=0.5) 
        self.sigmoid = nn.Sigmoid()
        
        
        self.embedding = nn.Embedding(self.vocab_size_child, self.emb_dim_child, self.padding_index)
        
        # If parent information provided at Init, create Parent branch
        if self.parent_information:
#             self.embedding_parent = nn.Embedding(self.vocab_size_parent, self.emb_dim_parent, self.padding_index)
    
            self.convolutions_parent = nn.ModuleList(
                    [nn.Conv2d(1, self.Co, (k, self.emb_dim_parent)) for k in self.Ks]) 
    
            self.linear_layers_parent = nn.ModuleList([
                nn.Linear(units[k], units[k + 1]) for k in range(len(units) - 1)])
            
            self.linear_last_parent = nn.Linear(self.Hu[-1], self.C_parent)
            
        # Create the Child Branch
        #self.embedding_child = nn.Embedding(self.vocab_size_child, self.emb_dim_child, self.padding_index) 
        
        self.convolutions_child = nn.ModuleList(
            [nn.Conv2d(1, self.Co, (k, self.emb_dim_child)) for k in self.Ks]) 
        
        self.linear_layers_child = nn.ModuleList([
                nn.Linear(units[k], units[k + 1]) for k in range(len(units) - 1)])

        self.linear_layers_combination = nn.ModuleList([nn.Linear(
            units_combination[k],units_combination[k + 1]) for k in range(len(units_combination) - 1)]) 
        
        self.linear_last_child = nn.Linear(units_combination[-1], self.C_child)
        
        self.linear_last_combined = nn.Linear(units_combination[-1], self.C_child)

        # Below we get embeddings from parent and child inputs

        # We concatenate the features of linear units
        # for both parent and child .
        # We use this to finally get the class of
        # the child because this way it has access to
        # the parent category information as well.

        # We also construct another branch of linear layers
        # so that during training we are able to compute the
        # output class of the parent category.
        # We do the following only during training
        # because during testing we are not concerned
        # with predicting the parent class.

    def forward(self, x):
        '''
            Takes the tokenized segment/sentence.
            If self.parent_information is True then the parent brach will also be trained
        '''
        
#         import pdb
#         pdb.set_trace()
        # Parent Branch
        x_emb = self.embedding(x)
        if self.parent_information:
            #x_parent = self.embedding_parent(x)
#             x_parent = [self.relu(conv(x_parent)).squeeze(3) for conv in self.convolutions_pare            
            x_parent = [self.relu(conv(x_emb)).squeeze(3) for conv in self.convolutions_parent]
            x_parent = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x_parent]
            x_parent = torch.cat(x_parent, 1)

            for i,linear in enumerate(self.linear_layers_parent):
                x_parent = linear(x_parent)
                x_parent = self.relu(x_parent)

            x_parent = self.dropout(x_parent)

            # if training:
            x_parent_last = self.linear_last_parent(x_parent)
            self.output['parent'] = self.sigmoid(x_parent_last)
        else: x_parent = torch.tensor([])

        # Child Branch
#         x_child = self.embedding_child(x)
#         x_child = [self.relu(conv(x_child)).squeeze(3) for conv in self.convolutions_child]
        x_child = [self.relu(conv(x_emb)).squeeze(3) for conv in self.convolutions_child]
        x_child = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x_child]
        x_child = torch.cat(x_child, 1)

        for linear in self.linear_layers_child:
            x_child = linear(x_child)
            x_child = self.relu(x_child)

        x_child = self.dropout(x_child)

        # if training:
        x_child_last = self.linear_last_child(x_child)
        self.output['child'] = self.sigmoid(x_child_last)
        
        # Combined 
        x_comb = torch.cat((x_parent, x_child), 1)
        
        for linear in self.linear_layers_combination:
            x_comb = linear(x_comb)
            x_comb = self.relu(x_comb)

        x_comb = self.linear_last_combined(x_comb)
        self.output['combined'] = self.sigmoid(x_comb)

        return self.output
# TRAINER CLASS TO TRAIN AND EVALUATE THE MODEL
import time
import sys
from math import ceil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from fastprogress.fastprogress import master_bar, progress_bar
from evaluation.f1_score import f1_score

# +
# TRAINER CLASS TO TRAIN AND EVALUATE THE MODEL
import time
import sys
from math import ceil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from fastprogress.fastprogress import master_bar, progress_bar
from evaluation.f1_score import f1_score

class Trainer():
    def __init__(self):
        '''
        Class to run training and evaluation
        '''
        # Set the device to use
        if torch.cuda.is_available(): self.device = 'cuda'
        else: self.device = 'cpu'

        # Set loss function
        self.criterion = nn.BCELoss()

        # Init stats and metrics
        self.train_losses, self.validation_losses, self.epochs = [],[],[]
        self.f1_scores_validations, self.precisions_validations, self.recalls_validations = [],[],[]
        self.f1_scores_validations_parent, self.precisions_validations_parent, self.recalls_validations_parent = [],[],[]
        self.best_f1score_validation = 0
        self.patience = 0

    def train(self, model, epochs_num=1, train_dataset=None, validation_dataset=None, data_collator=None, 
              parent_information=None, lr=0.01, batch_size=64, weight_decay=0.01, 
              betas=(0.9,0.999), evaluate_steps=40, has_parent=True, verbose=False): 
        '''
        Train the model given with the dataset provided. Will run evaluation on the validation set every
        `evaluate_steps` training steps, and at the end of each epoch.

        Args:
          model: instantiated model to train
          epochs_num: Number of epochs to train
          train_dataset: Train dataset
          validation_dataset: Validation dataset
          data_collator: A data collator function that when called will collate the data, passed to Dataloader
          parent_information:
          lr: Learning rate to use in the Opimizer
          batch_size: Batch size to use
          weight_decay: Optimizer wieght decay
          betas: Betas used in the Optimizer
          evaluate_steps: How many training steps
          verbose: If true the training loss and addition f1 scores will be printed every step
        Returns:
          f1: double, the resulting mean f1 score of all the labels (it will be a number between 0 and 1)
          precision: double, the resulting mean precision of all the labels (it will be a number between 0 and 1)
          recall:
        '''
        
        self.model = model

        # Prints additional loss and metrics information during training if set to true
        self.verbose = verbose
        
        # Set timers
        start = time.time()
        remaining_time = 0
        
        # Get dataloader
        self.data_collator = data_collator
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
                                      collate_fn=self.data_collator, shuffle=True)  
        # Default optimizer
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)

        mb = master_bar(range(epochs_num))
        pb = progress_bar(train_dataloader, parent=mb)
        for epoch in mb:
            for i_batch, sample_batched in enumerate(pb):
                self.model.train()

                # Get input
                x = sample_batched[0].to(self.device)
                
                #if i_batch == 0: 
                    #print()
                    #print(x.size())
                 
                # Get targets (labels)
                target = sample_batched[1].float().to(self.device)
                
                if has_parent and (len(sample_batched) == 3):
                    parent_labels = sample_batched[2].float().to(self.device)
                    
                    if self.device == 'cuda':
                        self.model.cuda(0)
                        x = x.cuda(0)
                        parent_labels = parent_labels.cuda(0)
                        target = target.cuda(0)
                    else:
                        self.model.cpu()
                        x = x.cpu()
                        parent_labels = parent_labels.cpu()
                        target = target.cpu()

                    # Pass input to model
                    output = self.model(x, parent_labels)
                else:  
                    if self.device == 'cuda':
                        self.model.cuda(0)
                        x = x.cuda(0)
                        target = target.cuda(0)
                    else:
                        self.model.cpu()
                        x = x.cpu()
                        target = target.cpu()
                    
                    # Pass input to model
                    output = self.model(x)
                
                # Loss
                train_loss = self.criterion(output, target)

                if self.verbose: 
                    print(f'train_loss: {train_loss}')
                
                # Do backward, do step and zero gradients
                train_loss.backward()
                optimizer.step()
                model.zero_grad()
                optimizer.zero_grad()

                # Evaluate
                if (i_batch > 0) and (i_batch % evaluate_steps) == 0: 
                    #print('\nevaluating...')
                    _ = self.evaluate(self.model, validation_dataset)
                
                self.train_losses.append(train_loss.item())
                
            # Run evaluation at the end of each epoch and return validation outputs
            #print('\nEnd of epoch evaluation results:')
            validation_outputs = self.evaluate(model, validation_dataset)
            y_hat_validation, validation_labels_child, validation_labels_parent = validation_outputs
            
            # Print out progress stats
            end = time.time()
            remaining_time = remaining_time * 0.90 + (
            (end - start) * (epochs_num - epoch + 1) / (epoch + 1)) * 0.1
            remaining_time_corrected = remaining_time / (1 - (0.9 ** (epoch + 1)))
            epoch_str = "last epoch finished: " + str(epoch+1)
            progress_str = "progress: " + str((epoch + 1) * 100 / epochs_num) + "%" 
            time_str = "time: " + str(remaining_time_corrected / 60) + " mins"
            sys.stdout.write("\r" + epoch_str + " -- " + progress_str + " -- " + time_str)
            sys.stdout.flush()

            self.epochs.append(epoch)

        print("\n" + "Training completed. Total training time: " + str(
            round((end - start) / 60, 2)) + " mins")
        return (y_hat_validation, validation_labels_child, validation_labels_parent,
                self.train_losses, self.validation_losses, 
                self.f1_scores_validations, self.precisions_validations, self.recalls_validations)

    def evaluate(self, model, validation_dataset, threshold=0.5, batch_size=32):
        # Set model to evaluation mode
        model.eval()

        # Get Validation Data
        val_data = self.data_collator(validation_dataset)
        
        # If no parent information provided in the validation set
        if len(val_data) == 2: 
            validation_segments, validation_labels = val_data
            validation_segments = validation_segments.long() #.to(self.device)
            validation_labels = validation_labels #.to(self.device)
            validation_labels_parent = None
        #If parent information provided in the validation set
        elif len(val_data) == 3:
            validation_segments, validation_labels, validation_labels_parent = val_data
            validation_segments = validation_segments.long() #.to(self.device)
            validation_labels = validation_labels #.to(self.device)
            validation_labels_parent = validation_labels_parent #.to(self.device)
        else:
            print('Warning, validation dataset should have 2 or 3 items!')
        
        # Get Correct Device
        if self.device == 'cuda':
            self.model.cuda(0)
            validation_segments = validation_segments.cuda(0)
            validation_labels = validation_labels.cuda(0)
            if len(val_data) == 3:
                validation_labels_parent.cuda(0)
        else:
            self.model.cpu()
            validation_segments = validation_segments.cpu()
            validation_labels = validation_labels.cpu()
            if len(val_data) == 3:
                validation_labels_parent.cpu()
        
        # Get Predictions from model
        with torch.no_grad():
            y_hat_validation =  self.model(validation_segments)
            
        validation_loss = self.criterion(y_hat_validation, validation_labels.float())
        #print(f'validation_loss: {validation_loss}')
        
        # Get COMBINED MODEL f1 score on CHILD LABELS, precision & recall for COMBINED predictions
        f1_scores_validation, precisions_validation, recalls_validation = f1_score(
            validation_labels.float(), y_hat_validation.float(), threshold)
        # Print output F1 score of combined model
        #print(f'Validation F1 Scores : {f1_scores_validation} \n')
        
        f1_scores_validation_parent, precisions_validation_parent, recalls_validation_parent = None,None,None
        
        # Do patience
        if (ceil(f1_scores_validation * 100) / 100) <= (ceil(self.best_f1score_validation * 100) / 100):
            self.patience = self.patience + 1
        else:
            self.best_f1score_validation = f1_scores_validation
            self.patience = 0
    
        # Set the metrics to the Trainer object
        self.validation_losses.append(validation_loss.item())
        self.f1_scores_validations.append(f1_scores_validation)
        self.precisions_validations.append(precisions_validation)
        self.recalls_validations.append(recalls_validation)
        
        validation_labels_parent = None
        return y_hat_validation, validation_labels.float(), validation_labels_parent

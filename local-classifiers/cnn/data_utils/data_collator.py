### DATA FILE

import numpy as np
import torch

class DataCollator():
    def __init__(self): pass
    
    def stack_segments(self, segments, clearance = 2):
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

#         tts = [torch.tensor(seg) for seg in segments]
#         segments_tensor = torch.tensor(nn.utils.rnn.pad_sequence(tts)).transpose(1,0)

        return segments_tensor 
    
    def __call__(self, batch):
        segments = [item[0] for item in batch]
        labels = [item[1] for item in batch]
#         segments = batch[0]
#         labels = batch[1]
#         x = batch[0]
#         if len(x) == 3:
        if len(batch[0]) == 3:
            parent_labels = [item[2] for item in batch]
#             parent_labels = batch[2]
            #parent_labels_tensor = batch[2]
            parent_labels_tensor = torch.stack(parent_labels)
            segments_tensor = self.stack_segments(segments)
            labels_tensor = torch.stack(labels)
#             labels_tensor = labels
            return [segments_tensor, labels_tensor, parent_labels_tensor]
        else:
            segments_tensor = self.stack_segments(segments)
            labels_tensor = torch.stack(labels)
            return [segments_tensor, labels_tensor]
        
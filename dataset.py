import pickle
import tqdm
from collections import Counter
from torch.utils.data import Dataset
import torch
import random

class EHRDataset(Dataset):
    def __init__(self, visit_info,  label, time):
        self.x_lines = visit_info
        self.y_lines = label
        self.time_lines = time
    def __len__(self):
        return len(self.x_lines)

    def __getitem__(self, index):
        visit_diag_code = self.x_lines[index]
        visit_label = self.y_lines[index]
        visit_time = self.time_lines[index]
        return visit_diag_code, visit_label, visit_time

    @staticmethod
    def collate_fn(batch):

        maxVisitTime = 460
        maxCodeLengh = 50
        padding_idx = 8692

        x_result = []
        y_result = []

        time_result = []
        len_result = []
        batch_visit =  torch.zeros(0, maxVisitTime, padding_idx)
        for visit, label, time in batch:

            visit.reverse()
            time.reverse()

            x_result.append(visit)
            y_result.append(label)
            len_result.append(len(visit))

            time.extend([time[-1]] * (maxVisitTime-len(time)))
            time_result.append(time)

            visit_sparse_set = torch.zeros(0,padding_idx)
            for v_j in visit:
                visit_sparse = torch.zeros(padding_idx)
                visit_sparse[v_j] = 1
                visit_sparse_set = torch.cat((visit_sparse_set, visit_sparse.unsqueeze(0)))

            visit_special = torch.zeros(1, padding_idx)
            visit_special[0][-1] = 1
            visit_special = visit_special.repeat(maxVisitTime-len(visit),1)
            visit_sparse_set = torch.cat((visit_sparse_set, visit_special))

            batch_visit = torch.cat((batch_visit, visit_sparse_set.unsqueeze(0)))

        return (batch_visit, torch.tensor(y_result), torch.tensor(time_result), len_result, x_result)

import cv2
import torch
from torch.utils.data import Dataset


class OktoberfestDataset(Dataset):
    def __init__(self, lines, path):
        self.lines = lines
        self.path = path
    
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        row = self.lines[idx][:-1].split(' ')
        
        fname = f'{self.path}/{row[0]}'
        img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
        
        num = int(row[1])
        boxes, labels = [], []
        for i in range(2, 2+5*num, 5):
            labels.append(int(row[i]))
            xmin, ymin = [float(n) for n in row[i+1: i+3]]
            xmax, ymax = [p + float(l) for p, l in zip([xmin,ymin], row[i+3:i+5])]
            boxes.append([xmin, ymin, xmax, ymax])
        
        boxes = torch.tensor(boxes)
        labels = torch.tensor(labels)
        
        target ={}
        target['boxes'] = boxes
        target['labels'] = labels
        
        return img, target
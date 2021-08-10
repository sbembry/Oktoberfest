import cv2
import numpy as np
import albumentations as A
import torch
from torch.utils.data import Dataset


class OktoberfestDataset(Dataset):
    def __init__(self, lines, path, augment=False):
        self.lines = lines
        self.path = path
        self.transform = A.Compose([A.HorizontalFlip(p=.5),
                                    A.Rotate(limit = 30, 
                                             border_mode = cv2.BORDER_CONSTANT, 
                                             value = 0.0, p = 0)
                                    ], bbox_params=A.BboxParams(format='coco'))
        self.augment = augment
        self.normalize = A.Normalize()
    
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        row = self.lines[idx][:-1].split(' ')
        
        fname = f'{self.path}/{row[0]}'
        img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
        
        num = int(row[1])
        boxes = [[float(n) for n in row[i+1:i+5]] + [int(row[i])] for i in range(2, 2+5*num, 5)]
        if self.augment:
            transformed = self.transform(image=img, bboxes=boxes)
            out = self.normalize(**transformed)
        else:
            out = self.normalize(image=img, bboxes=boxes)
        
        img = out['image']
        bboxes = []
        labels = []
        for i in range(len(boxes)):
            bboxes.append(out['bboxes'][i][:4])
            labels.append(out['bboxes'][i][-1] + 1) # Make 0 be background
        img = np.float32(np.transpose(img, [2,0,1]))
        
        target ={}
        target['boxes'] = torch.tensor(bboxes)
        target['labels'] = torch.tensor(labels)
        
        return torch.tensor(img), target
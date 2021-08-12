import cv2
import numpy as np
import albumentations as A
import torch
from torch.utils.data import Dataset


class OktoberfestDataset(Dataset):
    def __init__(self, lines, path, augment=False):
        self.max_boxes = 10
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
        for box in boxes:
            box[2] += box[0]
            box[3] += box[1]
            box = [min(box[0], box[2]), min(box[1],box[3]), max(box[0], box[2]), max(box[1], box[3])]
        if self.augment:
            transformed = self.transform(image=img, bboxes=boxes)
            out = self.normalize(**transformed)
        else:
            out = self.normalize(image=img, bboxes=boxes)
        
        img = out['image']
        bboxes = []
        labels = []
        for i in range(len(boxes)):
            #out['bboxes'][i][-1] = out['bboxes'][i][-1] + 1
            bboxes.append(out['bboxes'][i][:-1])
            labels.append(out['bboxes'][i][-1] + 1) # Make 0 be background
        img = np.float32(np.transpose(img, [2,0,1]))
        
        target = dict()
        input_annots = np.ones((self.max_boxes, 4), dtype=np.float32) * (0)
        input_annots[:,2] = 0.1
        input_annots[:,3] = 0.1
        #input_annots[:,4] = 0
        input_annots[:len(bboxes), :] = bboxes
        #target = input_annots
        target['boxes'] = torch.tensor(input_annots)
        input_labels = np.ones((self.max_boxes), dtype=np.float32) * (0)
        input_labels[:len(bboxes)] = labels
        target['labels'] = torch.tensor(input_labels).long()
        target['iscrowd'] = torch.zeros((10,), dtype=torch.int64)
        target['area'] = (target['boxes'][:,3] - target['boxes'][:,1]) * (target['boxes'][:,2] - target['boxes'][:,0])
        target['image_id'] = torch.tensor([idx])
        target['num_objs'] = torch.tensor(num)
        
        return torch.tensor(img), target
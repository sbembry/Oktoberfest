import cv2
import numpy as np
import albumentations as A
import torch
from torch.utils.data import Dataset


class OktoberfestDataset(Dataset):
    max_boxes = 10

    def __init__(self, lines, path, resize=(540, 960), augment=False, inference=False, include_orig=False):
        """
        Creates Dataset with bboxes and labels
        lines: list of lines in csv containing img path and bboxes
        path: path to images
        resize: Resize image. Currently it is cutting image in half
        augment: boolean on whether to do augmentations or not
        inference: boolean on whether this is for inference or not. True returns normalized image
                   as model input and resized image, False returns processed image + bboxes.
        """
        self.lines = lines
        self.path = path
        self.transform = A.Compose([A.HorizontalFlip(p=.5),
                                    A.Rotate(limit = 30, 
                                             border_mode = cv2.BORDER_CONSTANT, 
                                             value = 0.0, p = 0)
                                    ], bbox_params=A.BboxParams(format='pascal_voc'))
        if not inference:
            self.resize = A.Compose([A.Resize(resize[0], resize[1], interpolation=cv2.INTER_AREA)],
                                     bbox_params=A.BboxParams(format='pascal_voc'))
        else:
            self.resize = A.Resize(resize[0], resize[1], interpolation=cv2.INTER_AREA)
        self.augment = augment
        self.normalize = A.Normalize()
        self.inference = inference
        self.include_orig = include_orig
    
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        row = self.lines[idx].strip().split(' ')
        fname = f'{self.path}/{row[0]}'
        img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
        if self.inference:
            img = self.resize(image=img)#['image']
            normalized_image = self.normalize(**img)['image']
            normalized_image = np.float32(np.transpose(normalized_image, [2,0,1]))
            return torch.tensor(normalized_image), img['image']
        
        num = int(row[1])
        boxes = []
        boxes = [[float(n) for n in row[i+1:i+5]] + [int(row[i])] for i in range(2, 2+5*num, 5)]
        for box in boxes:
            box[2] += box[0]
            box[3] += box[1]
        b = boxes[0]
        transformed = self.resize(image=img, bboxes=boxes)
        orig = transformed['image']
        if self.augment:
            transformed = self.transform(**transformed)
        out = self.normalize(**transformed)
        
        img = out['image']
        bboxes = torch.tensor(out['bboxes'])
        input_annots = np.ones((self.max_boxes, 5), dtype=np.float32) * (-1)
        input_annots[:bboxes.shape[0], :] = bboxes
        img = np.float32(np.transpose(img, [2,0,1]))
        if self.include_orig:
            return torch.tensor(img), torch.tensor(input_annots), torch.tensor(orig)
        else:
            return torch.tensor(img), torch.tensor(input_annots)

    

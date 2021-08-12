import torch

def remove_extra_boxes(boxes, classes):
    good_boxes, good_classes = [], []
    for box, c in zip(boxes, classes):
        if not_repeat(box, good_boxes):
            good_boxes.append(box), good_classes.append(c)
    return torch.stack(good_boxes), torch.stack(good_classes)
        
def not_repeat(box, boxes):
    for b in boxes:
        if iou_pytorch(box, b) > .8:
            return False
    return True
        
def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    outputs_area = (outputs[2] - outputs[0]) * (outputs[3] - outputs[1])
    labels_area = (labels[2] - labels[0]) * (labels[3] - labels[1])
    
    # get area of intersection
    mins = torch.maximum(outputs[:2], labels[:2])
    maxs = torch.minimum(outputs[2:], labels[2:])
    intersection = (maxs[0] - mins[0]) * (maxs[1] - mins[1])
    union = outputs_area + labels_area - intersection
    return intersection / union
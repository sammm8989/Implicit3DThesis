#made by Sam Winant based on https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
#add .. to the path to import the lib folder
import torch
import torchvision
import pickle
import lib.utils as utils
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset
from lib.engine import train_one_epoch,evaluate
import torchvision.transforms as transforms
from torchvision import transforms as T
from PIL import Image
import numpy as np
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import json
import wandb

wandb.init(project="2D_detector")

batch_size = 6
amount_of_epochs = 21


NYU40CLASSES = ['void',
                'wall', 'floor', 'cabinet', 'bed', 'chair',
                'sofa', 'table', 'door', 'window', 'bookshelf',
                'picture', 'counter', 'blinds', 'desk', 'shelves',
                'curtain', 'dresser', 'pillow', 'mirror', 'floor_mat',
                'clothes', 'ceiling', 'books', 'refridgerator', 'television',
                'paper', 'towel', 'shower_curtain', 'box', 'whiteboard',
                'person', 'night_stand', 'toilet', 'sink', 'lamp',
                'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop']

#custom dataset with information about boundingboxes,image_path,labels and iscrowd
class CustomDataset(Dataset):
    def __init__(self, transform_target, data_list):
        self.data_list = data_list
        self.transform_target = transform_target

    def __getitem__(self, idx):
        data = self.data_list[idx]

        img = Image.open(data['image_path'])
        convert_tensor = transforms.ToTensor()
        img = convert_tensor(img)

        boxes = torch.tensor(data['boxes'], dtype=torch.float32)
        labels = torch.tensor(data['labels'], dtype=torch.int64)
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(data['labels']),), dtype=torch.int64)
        area = []
        for i in data['boxes']:
            area.append((i[2]-i[0])*(i[3]-i[1]))

        area = torch.tensor(area,dtype=torch.float32)    

        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id' : idx,
            'iscrowd' : iscrowd,
            'area' : area
        }
        
        target = self.transform_target(target)

        return img, target
    
    def __len__(self):
        return len(self.data_list)    


#get pretrained model
def make_model(num_class):
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_class)

    return model

def arrayFromSUNRGBD(pathToJSON):
    cocoDict = json.load(open(pathToJSON))
    
    listDict = []
    # Create a dictionary mapping image IDs to file names
    image_paths = {image["id"]: image["file_name"] for image in cocoDict["images"]}
    
    # Group annotations by image ID
    annotations_by_image = {}
    for annotation in cocoDict["annotations"]:
        if annotation["image_id"] not in annotations_by_image:
            annotations_by_image[annotation["image_id"]] = []
        annotations_by_image[annotation["image_id"]].append(annotation)
    
    for i in image_paths.keys():
        if i in annotations_by_image:
            data_sample = {}
            data_sample["image_path"] = image_paths[i]
            data_sample["boxes"] = [[annotation["bbox"][0], annotation["bbox"][1], annotation["bbox"][2] + annotation["bbox"][0], annotation["bbox"][3] + annotation["bbox"][1]] for annotation in annotations_by_image[i]]
            data_sample["labels"] = [annotation["category_id"] for annotation in annotations_by_image[i]]
            listDict.append(data_sample)
    
    return listDict   

def get_transform(train=True):
    image_transforms = []
    target_transforms = []
    if train:
        image_transforms.append(T.RandomHorizontalFlip(p=0.5))
    image_transforms.append(T.ConvertImageDtype(torch.float32))
    image_transforms.append(T.ToTensor())
    return T.Compose(image_transforms), T.Compose(target_transforms)


def calculatemAP(coco_evaluation):
    list_eval = coco_evaluation.coco_eval['bbox'].stats.tolist()
    ap_values = []
    ap_values.append(list_eval[0])
    ap_values.append(list_eval[3])
    ap_values.append(list_eval[4])
    ap_values.append(list_eval[5])
    mAP = np.average(ap_values)
    return mAP

def calculatemAR(coco_evaluation):
    list_eval = coco_evaluation.coco_eval['bbox'].stats.tolist()
    ap_values = []
    ap_values.append(list_eval[6])
    ap_values.append(list_eval[7])
    ap_values.append(list_eval[8])
    ap_values.append(list_eval[9])
    ap_values.append(list_eval[10])
    ap_values.append(list_eval[11])
    mAR = np.average(ap_values)
    return mAR

num_classes = len(NYU40CLASSES) 
model = make_model(num_classes)
train_dict = arrayFromSUNRGBD("../data/coco_dataset_train.json")
val_dict = arrayFromSUNRGBD("../data/coco_dataset_val.json")

transform_image, transform_target = get_transform(train=True)
train_dataset = CustomDataset(transform_target, train_dict)
val_dataset = CustomDataset(transform_target, val_dict)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=utils.collate_fn
)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=utils.collate_fn
)

model.to(device)
print(model)

params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

epoch = 0
wandb.watch(model)

best_val_loss = float('inf')
for epoch in range(amount_of_epochs):
    # Train for one epoch
    train_logger = train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=10)
    
    # Evaluate on the validation set
    val_evaluator = evaluate(model, val_dataloader, device)
    
    # Update the learning rate
    lr_scheduler.step()
    
    # Save the model if it's the best so far
    mAP = calculatemAP(val_evaluator)
    mAR = calculatemAR(val_evaluator)
    
    
    
    if mAR + mAP < best_val_loss:
        best_val_loss = mAR + mAP
        torch.save(model.state_dict(), 'best_model.pth')
        
    if epoch % 5 == 0:
        torch.save(model.state_dict(), f'epoch_{epoch}.pth')
    
    wandb.log({"epoch": epoch, "mAR": mAR, "mAP" : mAP, "lr": optimizer.param_groups[0]['lr']})
torch.save(model.state_dict(), 'Implicit3D/fine_tuned.pth')
print("Succesfully transfer learned 2D detector")
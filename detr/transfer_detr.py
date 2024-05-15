#made by Sam Winant based on https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
#add .. to the path to import the lib folder
import sys
sys.path.append("..")

import torch
from torch.utils.data import Dataset
from engine import train_one_epoch,evaluate
from util.misc import collate_fn
import torchvision.transforms as transforms
from models.matcher import HungarianMatcher
from models.detr import SetCriterion, PostProcess
from torchvision import transforms as T
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import json
import wandb

wandb.init(project="2D_detector")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

output_dir = 'output'
BATCH_SIZE = 12
num_epochs = 50

annotations_directory = '../data/annotations/'
images_directory = '../data/images/'

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

        img = Image.open(images_directory + data['image_path'])
        orig_size = img.size
        convert_tensor = transforms.ToTensor()
        img = convert_tensor(img)

        boxes = torch.tensor(data['boxes'], dtype=torch.float32)
        labels = torch.tensor(data['labels'], dtype=torch.int64)
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(data['labels']),), dtype=torch.int64)
        area = []
        for i in data['boxes']:
            area.append((i[2])*(i[3]))

        area = torch.tensor(area,dtype=torch.float32)    

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id' : torch.tensor(idx, dtype=torch.int64),
            'iscrowd' : iscrowd,
            'area' : area,
            'orig_size': torch.tensor(orig_size, dtype=torch.int64)
        }
        
        target = self.transform_target(target)

        return img, target
    
    def __len__(self):
        return len(self.data_list)   


#get pretrained model
def make_model():
    
    #Load model and remove the linear layer weights
    # Load the pre-trained model with the original number of classes
    model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)

    # Define a new classification head
    num_in_features = model.class_embed.in_features
    num_output_features = len(NYU40CLASSES) + 1 # Define your number of output features here

    # Replace the classification head
    model.class_embed = torch.nn.Linear(num_in_features, num_output_features)
    model.to(device)
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
            data_sample["boxes"] = [[annotation["bbox"][0], annotation["bbox"][1], annotation["bbox"][2], annotation["bbox"][3]] for annotation in annotations_by_image[i]]
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
model = make_model()
train_dict = arrayFromSUNRGBD(annotations_directory + "train_labels.json")
val_dict = arrayFromSUNRGBD(annotations_directory + "val_labels.json")

transform_image, transform_target = get_transform(train=True)
train_dataset = CustomDataset(transform_target, train_dict)
val_dataset = CustomDataset(transform_target, val_dict)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

sampler_train = torch.utils.data.RandomSampler(train_dataset)
sampler_val = torch.utils.data.SequentialSampler(val_dataset)

batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, BATCH_SIZE, drop_last=True)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=batch_sampler_train, collate_fn=collate_fn, num_workers=4)
val_dataloader = torch.utils.data.DataLoader(val_dataset, BATCH_SIZE, sampler=sampler_val, drop_last=False, collate_fn=collate_fn, num_workers=4)

model.to(device)
print(model)

# Get the parameters of the classification head to retrain
# parameters = list(model.parameters())
parameters = list(model.class_embed.parameters())

#optimizer and lr_scheduler
optimizer = torch.optim.AdamW(parameters, lr=0.001, weight_decay=0.0001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 0.0001)

matcher = HungarianMatcher()

weight_dict = {
    'loss_ce': 1, 
    'loss_bbox': 1,
    'loss_giou': 1
}

losses = ['labels', 'boxes', 'cardinality']

eos_coef = 0.1
criterion = SetCriterion(num_classes, matcher, weight_dict, eos_coef, losses).to(device)

#Load the postprocessor
postprocessors = {'bbox': PostProcess()}

val_ds = COCO(annotations_directory + "val_labels.json")

epoch = 0
wandb.watch(model)

#Train the model
model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_parameters = sum(p.numel() for p in parameters if p.requires_grad)
num_epochs = 250

print("Number of model parameters: ", model_parameters)
print("Number of training parameters: ", n_parameters)
print("Number of epochs: ", num_epochs)
print("Starting the actual training process...")

best = float('inf')

while(True):
    train_stats  = train_one_epoch(model, criterion, train_dataloader, optimizer, device, epoch)
    lr_scheduler.step()
    test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, val_dataloader, val_ds, device, output_dir)
    
    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'test_{k}': v for k, v in test_stats.items()},
                    'epoch': epoch,
                    'n_parameters': n_parameters,
                    'model_parameters' : model_parameters}
    
    wandb.log(log_stats)
    
    try:
        with open('output/log.txt', 'a') as f:
            f.write(json.dumps(log_stats) + "\n")
    except:
        print('Failed to open log.Txt')
    
    #extract test_loss_ce from test_stats, this is the class error
    score = test_stats['loss_ce']
    if score < best:
        print(f'New best model at Epoch {epoch}')
        best = score
        torch.save(model.state_dict(), 'pth/best_transfer.pth')
        wandb.log({"best_model_test_loss": best})
        
    
    if(epoch%10 == 0):
            torch.save(model.state_dict(), 'pth/sunrgbTransfer_epoch_'+str(epoch)+'.pth')
    
    if(epoch == num_epochs):
        break
    
    epoch += 1

print("Training finished!")
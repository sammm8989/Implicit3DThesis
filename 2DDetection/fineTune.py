#made by Sam Winant based on https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
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


datasetSplit = 0.70
datasetVal = 0.15
datasetSize = 10336
amount_of_epochs = 20


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
    
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features

    backbone.out_channels = 1280

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_class + 1,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

    return model

#get information out of the preprocessed data
def arrayFromSUNRGBD(pathToPickle, pathToImage , startIdDataset , stopIdDataset):
    listDict = []
    for i in range(startIdDataset, stopIdDataset):
        try:
            dataSample = {}
            with (open(pathToPickle + str(i) + ".pkl", "rb")) as openfile:
                dataFromPickle = pickle.load(openfile)


            dataSample["image_path"] = pathToImage + "/" + str(i) + "/img.jpg"
            dataSample["boxes"] = dataFromPickle["boxes"]["bdb2D_pos"]
            dataSample["labels"] = dataFromPickle["boxes"]["size_cls"]
            listDict.append(dataSample)

        except:
            pass   
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

if __name__ == "__main__":
    num_classes = len(NYU40CLASSES) 
    model = make_model(num_classes)
    dictionaries = arrayFromSUNRGBD("./Implicit3D/data/sunrgbd/sunrgbd_train_test_data/","./Implicit3D/data/time_data",1,int(datasetSize*datasetSplit))
    dictionaries1 = arrayFromSUNRGBD("./Implicit3D/data/sunrgbd/sunrgbd_train_test_data/","./Implicit3D/data/time_data",int(datasetSize*datasetSplit),int((datasetSplit+datasetVal)*datasetSplit))


    transform_image, transform_target = get_transform(train=True)
    dataset = CustomDataset(transform_target, dictionaries)
    dataset1 = CustomDataset(transform_target, dictionaries)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn
    )


    data_loader_val = torch.utils.data.DataLoader(
        dataset1,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        collate_fn=utils.collate_fn
    )

    model = make_model(num_classes)

    model.to(device)

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

    while(epoch < amount_of_epochs):
        eval = train_one_epoch(model, optimizer, data_loader,device, epoch, print_freq=100)
        lr_scheduler.step()
        epoch += 1

    torch.save(model.state_dict(), 'Implicit3D/fine_tuned.pth')
  

    print("Succesfully transfered learned 2D detector")

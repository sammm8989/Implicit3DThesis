import torch
import torchvision
import pickle
import lib.utils as utils

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset
from lib.engine import evaluate
import torchvision.transforms as transforms
from torchvision import transforms as T
from PIL import Image

datasetSplit = 0.85
datasetSize = 10336


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
    image_transforms.append(T.ConvertImageDtype(torch.float32))  # Convert to float32
    image_transforms.append(T.ToTensor())
    return T.Compose(image_transforms), T.Compose(target_transforms)

def make_model(num_class):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_class)
    model.load_state_dict(torch.load('./Implicit3D/fine_tuned.pth'))
    model.eval()
    return model


if __name__ == "__main__":
    dictionaries = arrayFromSUNRGBD("./Implicit3D/data/sunrgbd/sunrgbd_train_test_data/","./Implicit3D/data/time_data",int(datasetSize*datasetSplit),datasetSize)

    
    transform_image, transform_target = get_transform(train=True)
    dataset_test = CustomDataset(transform_target, dictionaries)

    data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=5,
    shuffle=False,
    num_workers=4,
    collate_fn=utils.collate_fn
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = make_model(41)
    model.to(device)
    evaluate(model, data_loader_test, device, False)

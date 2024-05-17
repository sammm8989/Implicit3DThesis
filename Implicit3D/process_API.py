# Demo script
# author: ynie updated by Sam Winant
# date: April, 2020
# updated by Sam Winant to make faster response time

from net_utils.utils import load_device, load_model
import json
from net_utils.utils import CheckpointIO
from configs.config_utils import mount_external_config
import numpy as np
import torch
import math
from configs.data_config import Relation_Config, NYU40CLASSES, NYU37_TO_PIX3D_CLS_MAPPING
rel_cfg = Relation_Config()
d_model = int(rel_cfg.d_g/4)
from models.total3d.dataloader import collate_fn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from net_utils.libs import get_layout_bdb_sunrgbd, get_rotation_matix_result, get_bdb_evaluation
from time import time
import gzip


HEIGHT_PATCH = 256
WIDTH_PATCH = 256

checkpoint = 0
device = 0
net = 0
cfg = 0
objectDetector = 0

twodet = []
implicit = []
zip_time = []

data_transforms = transforms.Compose([
    transforms.Resize((HEIGHT_PATCH, WIDTH_PATCH)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

     
def load_2D_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 41)
    model.load_state_dict(torch.load("fine_tuned.pth"))
    model.eval()
    return model


def get_bounding_boxes(photo, threshold):

    image_tensor = transforms.ToTensor()(photo)
    image_tensor = image_tensor.unsqueeze(0) 
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        prediction = objectDetector(image_tensor)

    bdb2D_pos = []
    size_cls = []
    for i in range(len(prediction[0]["boxes"])):
        if prediction[0]["scores"][i] >= threshold:
            box = prediction[0]["boxes"][i].tolist()
            label = prediction[0]["labels"][i].item()

            bdb2D_pos.append(box)
            size_cls.append(label)

    return bdb2D_pos, size_cls

def get_g_features(bdb2D_pos):
    n_objects = len(bdb2D_pos)
    g_feature = [[((loc2[0] + loc2[2]) / 2. - (loc1[0] + loc1[2]) / 2.) / (loc1[2] - loc1[0]),
                  ((loc2[1] + loc2[3]) / 2. - (loc1[1] + loc1[3]) / 2.) / (loc1[3] - loc1[1]),
                  math.log((loc2[2] - loc2[0]) / (loc1[2] - loc1[0])),
                  math.log((loc2[3] - loc2[1]) / (loc1[3] - loc1[1]))] \
                 for id1, loc1 in enumerate(bdb2D_pos)
                 for id2, loc2 in enumerate(bdb2D_pos)]

    locs = [num for loc in g_feature for num in loc]

    pe = torch.zeros(len(locs), d_model)
    position = torch.from_numpy(np.array(locs)).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe.view(n_objects * n_objects, rel_cfg.d_g)

def load_demo_data(device,image,cam_K):

    boxes = dict()

    bdb2D_pos, size_cls = get_bounding_boxes(image, 0.7)
    if not size_cls:
        return False
    # obtain geometric features
    boxes['g_feature'] = get_g_features(bdb2D_pos)

    # encode class
    cls_codes = torch.zeros([len(size_cls), len(NYU40CLASSES)])
    cls_codes[range(len(size_cls)), size_cls] = 1
    boxes['size_cls'] = cls_codes


    # get object images
    patch = []
    for bdb in bdb2D_pos:
        img = image.crop((bdb[0], bdb[1], bdb[2], bdb[3]))
        img = data_transforms(img)
        patch.append(img)


   
    boxes['patch'] = torch.stack(patch)
    image = data_transforms(image)



    camera = dict()
    camera['K'] = cam_K




    boxes['bdb2D_pos'] = np.array(bdb2D_pos)

    """assemble data"""
    data = collate_fn([{'image':image, 'boxes_batch':boxes, 'camera':camera}])
    image = data['image'].to(device)
    K = data['camera']['K'].float().to(device)
    patch = data['boxes_batch']['patch'].to(device)
    size_cls = data['boxes_batch']['size_cls'].float().to(device)
    g_features = data['boxes_batch']['g_feature'].float().to(device)
    split = data['obj_split']
    rel_pair_counts = torch.cat([torch.tensor([0]), torch.cumsum(
        torch.pow(data['obj_split'][:, 1] - data['obj_split'][:, 0], 2), 0)], 0)
    cls_codes = torch.zeros([size_cls.size(0), 9]).to(device)
    cls_codes[range(size_cls.size(0)), [NYU37_TO_PIX3D_CLS_MAPPING[cls.item()] for cls in
                                        torch.argmax(size_cls, dim=1)]] = 1
    bdb2D_pos = data['boxes_batch']['bdb2D_pos'].float().to(device)

    input_data = {'image':image, 'K':K, 'patch':patch, 'patch_for_mesh':patch, 'g_features':g_features,
                  'size_cls':size_cls, 'split':split, 'rel_pair_counts':rel_pair_counts,
                  'cls_codes':cls_codes, 'bdb2D_pos':bdb2D_pos}

    return input_data


def initiate(cfg_begin):
    '''Begin to run network.'''
    global checkpoint 
    global device
    global net
    global cfg
    global objectDetector

    cfg = cfg_begin

    checkpoint = CheckpointIO(cfg)

    '''Mount external config data'''
    cfg = mount_external_config(cfg)

    device = load_device(cfg)

    net = load_model(cfg, device=device)

    checkpoint.register_modules(net=net)


    checkpoint.parse_checkpoint()

    objectDetector = load_2D_model()
    
    net.train(cfg.config['mode'] == 'train')
    objectDetector.to(device)
    print("All the things are ok")

def transform_dict(og_dictioniary):
    new_dictioniary = {}


    new_dictioniary["layout"] = og_dictioniary["layout"].tolist()
    new_dictioniary["cam_R"] = og_dictioniary["cam_R"].tolist()
    new_dictioniary["class_id"] = og_dictioniary["class_id"]

    bigList = []
    for i in og_dictioniary["bdb"]:
        smallDict = {}
        smallDict["basis"] = i["basis"].tolist()
        smallDict["coeffs"] = i["coeffs"].tolist()
        smallDict["centroid"] = i["centroid"].tolist()
        smallDict["classid"] = int(i["classid"])
        bigList.append(smallDict)
    new_dictioniary["bdb"] = bigList

    masterList = []
    for i in og_dictioniary["objects"]:
        slaveDict = {}
        slaveDict["nameObject"] = str(i["nameObject"])
        slaveDict["v"] = i["v"].tolist()
        slaveDict["f"] = i["f"].tolist()
        masterList.append(slaveDict)
    new_dictioniary["objects"] = masterList

    return new_dictioniary

def run(image, camera):
    global implicit
    global twodet
    global zip_time
    start = time()
    data = load_demo_data(device,image,camera)
    if(data == False):
        return False
    twodet.append(time() - start)
    #print data to asses response time
    #print("TwoDetections = " + str(twodet))

    start = time()
    with torch.no_grad():
        est_data = net(data)

    implicit.append(time()-start)
    #print data to asses response time
    #print("Implicit calculations =" + str(implicit))
    start = time()

    lo_bdb3D_out = get_layout_bdb_sunrgbd(cfg.bins_tensor, est_data['lo_ori_reg_result'],
                                          torch.argmax(est_data['lo_ori_cls_result'], 1),
                                          est_data['lo_centroid_result'],
                                          est_data['lo_coeffs_result'])

    # camera orientation for evaluation
    cam_R_out = get_rotation_matix_result(cfg.bins_tensor,
                                          torch.argmax(est_data['pitch_cls_result'], 1), est_data['pitch_reg_result'],
                                          torch.argmax(est_data['roll_cls_result'], 1), est_data['roll_reg_result'])


    # projected center
    P_result = torch.stack(((data['bdb2D_pos'][:, 0] + data['bdb2D_pos'][:, 2]) / 2 -
                            (data['bdb2D_pos'][:, 2] - data['bdb2D_pos'][:, 0]) * est_data['offset_2D_result'][:, 0],
                            (data['bdb2D_pos'][:, 1] + data['bdb2D_pos'][:, 3]) / 2 -
                            (data['bdb2D_pos'][:, 3] - data['bdb2D_pos'][:, 1]) * est_data['offset_2D_result'][:,1]), 1)
    
    bdb3D_out_form_cpu, bdb3D_out = get_bdb_evaluation(cfg.bins_tensor,
                                                       torch.argmax(est_data['ori_cls_result'], 1),
                                                       est_data['ori_reg_result'],
                                                       torch.argmax(est_data['centroid_cls_result'], 1),
                                                       est_data['centroid_reg_result'],
                                                       data['size_cls'], est_data['size_reg_result'], P_result,
                                                       data['K'], cam_R_out, data['split'], return_bdb=True)

    
    


    # save results
    nyu40class_ids = [int(evaluate_bdb['classid']) for evaluate_bdb in bdb3D_out_form_cpu]

    return_dict = {}

    # save layout
    return_dict["layout"] = lo_bdb3D_out[0, :, :].cpu().numpy()
    # save bounding boxes and camera poses
    interval = data['split'][0].cpu().tolist()
    current_cls = nyu40class_ids[interval[0]:interval[1]]


    return_dict["bdb"] = bdb3D_out_form_cpu[interval[0]:interval[1]]
    return_dict['class_id'] = current_cls
    return_dict["cam_R"] = cam_R_out[0, :, :].cpu().numpy()


    # save meshes
    current_faces = est_data['out_faces'][interval[0]:interval[1]]
    current_coordinates = est_data['meshes'][interval[0]:interval[1]]

    objects = []
    mesh_obj = {}

    for obj_id, obj_cls in enumerate(current_cls):
        file_name = '%s_%s.obj' % (obj_id, obj_cls)

        mesh_obj = {"nameObject" : file_name ,'v': current_coordinates[obj_id].transpose(-1, -2).cpu().numpy(),
                    'f': current_faces[obj_id].cpu().numpy()}

        objects.append(mesh_obj)

    return_dict["objects"] = objects

    json_data = json.dumps(transform_dict(return_dict))
    compressed = gzip.compress(json_data.encode(),compresslevel=1)


    zip_time.append(time() - start)
    #print data to asses response time
    #print("zip_time = " + str(zip_time))

    return compressed

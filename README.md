
# Thesis Fiten-Winant implementation of Implicit3D

This is the github page of the master thesis of Fiten-Winant, where we propose an server implentation of the Implicit3D [https://arxiv.org/pdf/2103.06422] 3D scene understanding.

In this github repo we find two branches. One is the server implementation. This code can be used to deploy the server with pretrained models in it, but can also be used to train and evaluate all the models. The other one is an unity project for a meta quest 3 that is our example of a client that will be used as our proof of concept. 





## Setting up environment

Start by installing some dependencies we need:
```
sudo apt install xvfb ninja-build freeglut3-dev libglew-dev meshlab
```

Install CUDA toolkit (https://developer.nvidia.com/cuda-12-2-0-download-archive). To check if it is installed properly try:

```
nvidia-smi
nvcc -V
```
Both commands should give the same CUDA version. Al code is made with CUDA 12.2 on an ubuntu 22.04.

To set up the conda environment go the root folder and run:

```
conda env create -f environment.yml
conda activate Im3D

```




## Demo

Find the Compute Capability on (https://developer.nvidia.com/cuda-gpus) of your GPU. Set this values in Implicit3D/external/mesh_fusion/libfusiongpu/CMakeLists.txt. Then go to Implicit3D/external/ldif/ldif2mesh/get_cuda_version.py the method should return major and minor version of you Cuda Toolkit.

Go to the /Implicit3D and run:
```
python project.py build
```
Download the pretrained checkpoint (https://kuleuven-my.sharepoint.com/:u:/g/personal/sam_winant_student_kuleuven_be/EdjmhkpQfPZDrKlTIVtfwTEBdMgVhvHAWdpAM_MCbfPc_w?e=7FmZmz) and unzip it into out/total3d/20110611514267/

To check if everything works fine try to run the demo:
```
CUDA_VISIBLE_DEVICES=0 python main.py out/total3d/20110611514267/out_config.yaml --mode demo --demo_path demo/inputs/1
```

In case of errors please visit (https://github.com/chengzhag/Implicit3DUnderstanding/tree/main)



## Data preparation

In the Implicit3D directory do following:

    1. Follow Total3DUnderstanding to download the raw data.(https://github.com/GAP-LAB-CUHK-SZ/Total3DUnderstanding)
    2. According to issue #6(https://github.com/GAP-LAB-CUHK-SZ/Total3DUnderstanding/issues/6) of Total3DUnderstanding, there are a few typos in json files of SUNRGBD dataset, which is mostly solved by the json loader. However, one typo still needs to be fixed by hand. Please find {"name":""propulsion"tool"} in data/sunrgbd/Dataset/SUNRGBD/kv2/kinect2data/002922_2014-06-26_15-43-16_094959634447_rgbf000089-resize/annotation2Dfinal/index.json and remove ""propulsion.
    3. Process the data by
```
python -m utils.generate_data
```

To generate the timing data (being the SUNRGBD dataset but transformed to only pictures and camera intrinstic values matrix) run following command in the root of the project.

```
python generate_time_data.py
```
## Training and Testing

To train the 2D detector with finetunring run the following command in the root directory (the parameters can be tuned in the pyhon file). The pretrained model can be found at https://kuleuven-my.sharepoint.com/:u:/g/personal/sam_winant_student_kuleuven_be/EdjmhkpQfPZDrKlTIVtfwTEBdMgVhvHAWdpAM_MCbfPc_w?e=3QiEBT
```
python 2DDetection/fineTune.py 
```


To test the accuracy of the 2D detector we run in the root folder:
```
python 2DDetection/test_accuracy.py
```
To assess the average time the Implicit3D takes to do predictions you can run (the 95 can be changed to the amount of samples you take to calculate the average):
```
CUDA_VISIBLE_DEVICES=0 python main.py out/total3d/20110611514267/out_config.yaml --mode demo_with_time --avg_amount 95
```

## Server

Go to Implicit3D/server.py and insert the hostname of the server at the last line then run this command in the Implicit3D directory:
```
python3 server.py out/total3d/20110611514267/out_config.yaml
```

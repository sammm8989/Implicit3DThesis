# 2D detectors
## Setting up

Follow the fullproject branch to make the pickle data and the time_data. You can copy or use a shortcut to place the pickle data in a directory named pickle in data. The directory time_data needs to be put in photos. Start to train with:
```
python train_resnet.py
```
When the fine_tuned.pth is made you can load it in the Implicit3D directory as described in Implicit3D.

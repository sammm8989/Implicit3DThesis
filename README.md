
# 2D object detectors

Three ([resnet50](https://drive.google.com/file/d/1Z_9QHP-3vpYd13l8iTnC33Qqh7XVm-Eo/view?usp=sharing), [mobilenet](https://drive.google.com/file/d/1q07IdfKaBZvmzjJgspviQk9vCr67mYg2/view?usp=sharing) and [DETR](https://drive.google.com/file/d/1odzu96fTtsv5NwU4Nq8XDDjQ_sjsDcXS/view?usp=sharing)) different 2D object detectors are tested. These detectors are trained with transfer learning. In the directory data add an directory called pickles. In here the data that is made in the FullProject (Implicit3D/data/sunrgbd/sunrgbd_train_test_data) repo can be pasted or an shortcut can be made. In the data directory a directory named images should also be present. To generate this from the Fullproject repo paste the path to time_data and run:
```
python imagesCopier.py
```

In each directory is a python script to transfer learn a 2D detector. The resnet50 gave the best results so this is the only one that can be added to the server implementation without code change on the server.



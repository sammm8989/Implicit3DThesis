#Sam Winant

from PIL import Image as im 
import os
import numpy as np
import pickle
from Implicit3D.configs.data_config import Config
from multiprocessing import Pool


def makeFolder(id):
    try:
        pickle_path = os.path.join("Implicit3D/data/sunrgbd/sunrgbd_train_test_data", str(id) + '.pkl')
        dataFromPickle = {}
        with (open(pickle_path, "rb")) as file:
            dataFromPickle = pickle.load(file)

        time_data_path = os.path.join("Implicit3D/data/time_data", str(dataFromPickle["sequence_id"]))
        os.makedirs(time_data_path, exist_ok=True)

        dataIm = im.fromarray(np.array(dataFromPickle["rgb_img"])) 
        image_path = os.path.join(time_data_path, "img.jpg" )
        dataIm.save(image_path) 


        K = dataFromPickle["camera"]["K"]
        formatted_matrix = np.array([
        [f'{entry:.1f}' for entry in row] for row in K
        ])

        camera_path = os.path.join(time_data_path, "cam_K.txt" )
        np.savetxt(camera_path, formatted_matrix, fmt='%s', delimiter='\t')
    except:
        pass   


if __name__ == '__main__':
    sunrgbd_config = Config
    os.makedirs("Implicit3D/data/time_data", exist_ok=True)
    num_list = list(range(1, 10336))

    p = Pool(processes=12)
    save_outputs = p.map(makeFolder, num_list)
    p.close()
    p.join()


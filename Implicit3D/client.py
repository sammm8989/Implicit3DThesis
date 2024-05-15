import requests
from time import time
import json
import gzip
import os

listTime = []

for i in range(1,100):
    pathIm = './data/time_data/'+str(i)+'/img.jpg'
    pathText = "./data/time_data/"+str(i)+"/cam_K.txt"
    if(os.path.isfile(pathIm) and os.path.isfile(pathText) ):
        files = {'photo': open(pathIm, 'rb'), 'cam' : open(pathText,"rb")}
        types_set = {type(value) for value in files.values()}
        start = time()
        response = requests.post(url = "http://www.gijsfiten.com:5000/upload", files=files)

        if response.status_code == 200:
            decompressed_data = json.loads(gzip.decompress(response.content).decode())
            listTime.append(time() - start) 
            #uncomment this line to print the totaltime (needed to make plots of response time)
            #print("TotalTime = " + str(listTime))  
            print(decompressed_data["bdb"][0]["centroid"])   

        else:
            print('Error:', response.status_code)



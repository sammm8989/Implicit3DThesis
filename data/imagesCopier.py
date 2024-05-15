import os
import shutil

path = "/home/sam/Desktop/Thesis/giter/Implicit3DUnderstanding/Implicit3D/data/time_data"
try:  
    os.mkdir("images")  
except OSError as error:  
    print("Directory already exists")   

    
for i in range(0,10336):  

    photo = path+"/"+str(i)+"/img.jpg"
    if(os.path.isfile(photo)):
        shutil.copyfile(photo, 'images/'+str(i)+".jpg")

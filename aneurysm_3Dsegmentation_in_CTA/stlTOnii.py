import subprocess
import os
import shutil

source      = "patients"
source_file = os.listdir(source)
source_file.remove(".DS_Store") # for mac systems
cnt = 0
for p in source_file :
    stlFileList = []
    pFile = os.listdir(os.path.join(source , p))
    for file in pFile :
        if ".nii.gz" in file :
            refFile = file
        elif "artery" not in file and ".stl" in file :
            stlFileList.append(file)
    else :
        for stlFile in stlFileList :
            command = [
                "stl2nii" , 
                "-i" , 
                os.path.join(source , p , stlFile) ,
                "-r" ,
                os.path.join(source , p , refFile) ,
                "-o" ,
                os.path.join(source , "converted" , p)
                ]
            print(command)
            subprocess.run(command)
        cnt+=1
        print("number of patient : " , cnt)
        shutil.copy(os.path.join(source , p , refFile) , os.path.join(source , "converted" , p))
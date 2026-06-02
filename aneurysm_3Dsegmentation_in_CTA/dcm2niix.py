import shutil
import os
import subprocess

src = "dicom cases"
dcmzip = os.listdir(src)

try : 
    os.mkdir("unzip")
except : 
    pass

for name in dcmzip: 
    inputDir  = os.path.join(src , name)
    outputDir   = "unzip"
    unzipfolder = os.path.join(outputDir , name).replace(".zip" , "")
    print("./ + {unzipfolder}")
    shutil.unpack_archive(inputDir , unzipfolder)
    res = subprocess.run(["./dcm2niix" , f"{unzipfolder}"])

    break



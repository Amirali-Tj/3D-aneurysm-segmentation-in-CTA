import os
import shutil

src = "unzip"

cnt = 0
for patient in os.listdir(src) :
    if patient != ".DS_Store" : # not for other systems
        patient_study = os.listdir(os.path.join(src , patient))
        selector = {
            "fileSize" : 0 , 
            "fileAdd" : "" ,
        }
        for file in patient_study :
            if "CTA" in file.upper() :
                fileSize = os.path.getsize(os.path.join(src , patient , file))
                if selector["fileSize"] <= fileSize : 
                    selector["fileAdd"] = os.path.join(src , patient , file)
                    selector["fileSize"] = fileSize
        else :
            try : 
                os.makedirs(os.path.join("CTA nii" , patient)) 
                shutil.move(selector["fileAdd"] , os.path.join("CTA nii" , patient))
                cnt+=1
            except :
                pass
else :
    print("################" , f"patient conveted files : {cnt}" , sep="\n")
                
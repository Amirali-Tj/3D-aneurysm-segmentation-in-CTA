import os
from pprint import pprint
import numpy as np
import nibabel as nib

def dataSplitPerSize(dataPath) :
    source = os.listdir(dataPath)
    smallAneurysm    = []
    mediumAneurysm   = []
    largeAneurysm    = []

    for p in source :
        files = os.listdir(os.path.join(dataPath , p))
        aneurysmList = []

        for file in files : 
            if "label" in file :
                file = file.rstrip("-label.nii")
                nameSplit = file.split("_@")
                aneurysmSize = int(nameSplit[1])
                aneurysmList.append(aneurysmSize)
        
                try : # it will be added
                    location     = nameSplit[2]
                except IndexError :
                    pass
            
        averageSize = sum(aneurysmList)/len(aneurysmList)
        if averageSize <= 50 :
            smallAneurysm.append(os.path.join(dataPath , p))
        if averageSize >50 and averageSize <=200 :
            mediumAneurysm.append(os.path.join(dataPath , p))
        if averageSize > 200 :
            largeAneurysm.append(os.path.join(dataPath , p))

    
    return smallAneurysm , mediumAneurysm , largeAneurysm



def dataSplitPerSample(sets , testRatio , seed=None) :
    sets = list(sets)
    if seed != None :
        np.random.seed(seed)

    fullTrainSet = []
    fullTestSet  = []

    for set in sets :
        np.random.shuffle(set)
        setLen = len(set)
        testLen  = np.round(testRatio*setLen)
        trainLen = int(setLen - testLen)
        
        trainSet = set[0:trainLen]
        testSet  = set[trainLen:]

        fullTrainSet.extend(trainSet)
        fullTestSet.extend(testSet)
    
    return fullTrainSet , fullTestSet



def dataTensorLoading(dataset) :
    images = []
    labels = []
    mergeds = []

    for p in dataset :
        files = os.listdir(p)
        label  = []
        image  = []
        merged = []

        for file in files :
            if "label" not in file and "merged" not in file :
                image.append(os.path.join(p , file))
            elif "merge" in file :
                merged.append(os.path.join(p , file))
            else :
                label.append(os.path.join(p , file))
        else :
            if len(merged) == 0 :
                merged.append("NoMerge")
        
        image  = image*len(label)
        merged = merged*len(label)

        images.extend(image)
        labels.extend(label)
        mergeds.extend(merged)

    return images , labels , mergeds


        
def merge(filePath) :
    pFiles = os.listdir(filePath)
    if len(pFiles) > 2 :
        image  = None
        labels = []
        for name in pFiles :
            if "label" in name :
                label = nib.as_closest_canonical(nib.load(os.path.join(filePath , name)))
                label_array = label.get_fdata()
                labels.append(label_array)
            elif "label" not in name :
                image = name
        label_tensor  = np.array(labels)
        merged_tensor = np.sum(label_tensor , axis=0)
        print(label_tensor.shape)

        imgObj = nib.Nifti1Image(
            merged_tensor ,
            affine = label.affine ,
            header = label.header ,
        )

        nib.save(
            imgObj ,
            os.path.join(filePath ,  image.replace(".nii" , "_merged") + ".nii")
        )
        return 1
    else :
        return 0

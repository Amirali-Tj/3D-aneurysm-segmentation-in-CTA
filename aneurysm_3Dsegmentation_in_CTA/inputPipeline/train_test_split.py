import os
from pprint import pprint
import numpy as np


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

    for p in dataset :
        files = os.listdir(p)
        label = []
        image = []
        for file in files :
            if "label" not in file :
                image.append(os.path.join(p , file))
            else :
                label.append(os.path.join(p , file))
        image = image*len(label)

        images.extend(image)
        labels.extend(label)

    return images , labels


        
def merge(*label) :
    label_tensor  = np.array(label)
    merged_tensor = np.sum(label_tensor , sum=0)

    return merged_tensor
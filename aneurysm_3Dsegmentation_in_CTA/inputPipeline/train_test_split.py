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

    
    return {
        "S" : smallAneurysm ,
        "M" : mediumAneurysm ,
        "L" : largeAneurysm 
    }


def dataSplitPerlocation(dataPath) :
    source    = os.listdir(dataPath)
    container = {}
    # adding files to their locations
    for p in source :
        files = os.listdir(os.path.join(dataPath , p))
        locList = []
        for file in files : 
            if "label" in file :
                file      = file.rstrip("-label.nii")
                nameSplit = file.split("_@")
                loc       = nameSplit[2]

                exst = container.get(loc , 0)
                if exst == 0 :
                    container[loc] = []
            
                locList.append(loc)
        locUnq = set(locList)  # variation source

        # finding group
        cntMax  = 0
        nameMax = "" 
        for loc in locUnq :
            cnt = locList.count(loc)
            if cnt > cntMax :
                cntMax  = cnt
                nameMax = loc
        
        # add to container
        container[nameMax].append(os.path.join(dataPath , p))

    return container

def dataSpiltPerConfiguration(dataPath) :
    source    = os.listdir(dataPath)
    container = {}
    # adding files to their locations
    for p in source :
        files = os.listdir(os.path.join(dataPath , p))
        confList = []
        for file in files : 
            if "label" in file :
                file      = file.rstrip("-label.nii")
                nameSplit = file.split("_@")
                conf       = nameSplit[3]

                exst = container.get(conf , 0)
                if exst == 0 :
                    container[conf] = []
            
                confList.append(conf)
        confUnq = set(confList)

        # finding group
        cntMax  = 0
        nameMax = "" 
        for conf in confUnq :
            cnt = confList.count(conf)
            if cnt > cntMax :
                cntMax  = cnt
                nameMax = conf
        
        # add to container
        container[nameMax].append(os.path.join(dataPath , p))

    return container

def findCommon(containers) :
    if len(containers) > 2 :
        raise Exception("container length must be two")

    # selecting first container
    container_first  = containers[0]
    container_second = containers[1]
    commonContainer  = {}
    
    # iterating over containers
    for key in container_first.keys() :
        key_vals = container_first[key]
        for key_2 in container_second.keys() :
            key_2_vals = container_second[key_2]
            for name in key_vals :
                for name_2 in key_2_vals :
                    if name == name_2 :
                        exst = commonContainer.get(f"{key}_{key_2}" , 0)
                        if exst == 0 :
                            commonContainer[f"{key}_{key_2}"] = []
                            commonContainer[f"{key}_{key_2}"].append(name)
                            break
                            # ----
                        else :
                            commonContainer[f"{key}_{key_2}"].append(name)
                            break
            
    return commonContainer


def dataSplitPerSample(sets , testRatio , perSet=False , seed=None) :
    sets = list(sets)
    if seed != None :
        np.random.seed(seed)

    fullTrainSet = []
    fullTestSet  = []
    perSetCollection = {
        "train_set" : {} ,
        "test_set"  : {}
    }

    set_ix = 0
    for set in sets :
        np.random.shuffle(set)
        setLen = len(set)
        testLen  = np.round(testRatio*setLen)
        trainLen = int(setLen - testLen)
        
        trainSet = set[0:trainLen]
        testSet  = set[trainLen:]

        fullTrainSet.extend(trainSet)
        fullTestSet.extend(testSet)

        if perSet :
            perSetCollection["train_set"][set_ix] = trainSet
            perSetCollection["test_set"][set_ix]  = testSet
            set_ix+=1
        else :
            perSetCollection = {}
        
    
    return fullTrainSet , fullTestSet , perSetCollection



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

def dataTensorloadingPerSize(dataset) :
    # plotting aneurysm sizes

    smallImgSet    = []
    mediumImgSet    = []
    largeImgSet    = []

    smallLabelSet  = []
    mediumLabelSet = []
    largeLabelSet  = []

    smallMergeSet  = []
    mediumMergeSet = []
    largeMergeSet  = []

    for p in dataset :
        p_dir = os.listdir(p)
        labCntS = 0
        labCntM = 0
        labCntL = 0
        mergeFile= None

        for file in p_dir :
            if "label" in file :
                nameSplit = file.split("_@")
                aneurysmSize = int(nameSplit[1])
                if aneurysmSize <= 50 :
                    smallLabelSet.append(os.path.join(p , file))
                    labCntS+=1
                elif aneurysmSize >50 and aneurysmSize <=200 :
                    mediumLabelSet.append(os.path.join(p , file))
                    labCntM+=1
                elif aneurysmSize > 200 :
                    largeLabelSet.append(os.path.join(p , file))
                    labCntL+=1
            elif "merged" in file :
                mergeFile = os.path.join(p , file)
            else :
                imgFile   = os.path.join(p , file)
        else :
            if mergeFile == None :
                mergeFile = "NoMerge"

            smallMergeSet.extend([mergeFile]*labCntS)
            mediumMergeSet.extend([mergeFile]*labCntM)
            largeMergeSet.extend([mergeFile]*labCntL)

            smallImgSet.extend([imgFile]*labCntS)
            mediumImgSet.extend([imgFile]*labCntM)
            largeImgSet.extend([imgFile]*labCntL)

    return {
        "S" : [smallImgSet  , smallLabelSet  , smallMergeSet] ,
        "M" : [mediumImgSet , mediumLabelSet , mediumMergeSet] ,
        "L" : [largeImgSet  , largeLabelSet  , largeMergeSet]
    }
        
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


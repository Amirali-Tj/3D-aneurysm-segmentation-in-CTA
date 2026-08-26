import tensorflow as tf
import os
import nibabel as nib
import numpy as np


class multiWindowStacking : # cache
    def __init__(self  , ranges):
        self.ranges = tf.convert_to_tensor(list(ranges) , dtype=tf.float64)
    def WindowStacking(self , img_arr , label_arr) : # graph compatibale
        def cond(rng , i , img_arr , stackedWindows) :
            return i < tf.shape(rng)[0]
        def body(rng , i , img_arr , stackedWindows) :
            wl = rng[i][0]
            ww = rng[i][1]
            minHU = wl - (ww/2)
            maxHU = wl + (ww/2)
            windowChannel  = tf.expand_dims(tf.clip_by_value(img_arr , minHU , maxHU) , axis=0)
            stackedWindows = tf.concat([stackedWindows , windowChannel] , axis=0)
            i+=1
            return rng , i , img_arr , stackedWindows
        _ , _ , _ , windows = tf.while_loop(
            cond ,
            body ,
            loop_vars=[
                self.ranges ,
                0 , 
                img_arr ,
                tf.expand_dims(tf.zeros(tf.shape(img_arr) , dtype=tf.float64) , axis=0)
            ] , 
            shape_invariants = [
                self.ranges.get_shape(),
                tf.TensorShape(()) ,
                img_arr.get_shape() , 
                tf.TensorShape([None , None , None , None])
            ]
        )
        #---
        img_arr   = windows[1: , : , : , :]
        label_arr = tf.tile(tf.expand_dims(label_arr , axis=0) , [tf.shape(self.ranges)[0] , 1 , 1 , 1])
        #---
        return img_arr , label_arr


class quartileWindowStacking(multiWindowStacking) :
    def __init__(self) :
        pass
    def _quaritleWindowFinder(image_arr , label_arr) : 
        pass

class randomMultiWindowStackig() : # on-fly and cache
    def __init__(self , default , wwRange , wlRange , p_ww , p_wl): # default as tf.float32
        self.default = tf.expand_dims(tf.convert_to_tensor(default , dtype=tf.float64) , axis=0)
        self.wwRange = tf.expand_dims(tf.convert_to_tensor(wwRange , dtype=tf.float64) , axis=0)
        self.wlRange = tf.expand_dims(tf.convert_to_tensor(wlRange , dtype=tf.float64) , axis=0)
        self.p_ww    = tf.convert_to_tensor(p_ww , dtype=tf.float64)
        self.p_wl    = tf.convert_to_tensor(p_wl , dtype=tf.float64)
    def WindowStacking(self , img_arr , label_arr) : # graph compatibale
        rGenWL = tf.cond(
            tf.random.uniform(shape=() , dtype=tf.float64) <= self.p_wl ,
            lambda : tf.random.uniform(
                shape=() ,
                minval=self.wlRange[0][0] ,
                maxval=self.wlRange[0][1] ,
                dtype=tf.float64
            ),
            lambda : self.default[0][0]
            )
        rGenWW = tf.cond(
            tf.random.uniform(shape=() , dtype=tf.float64) <= self.p_ww ,
            lambda : tf.random.uniform(
                shape=() ,
                minval=self.wwRange[0][0] ,
                maxval=self.wwRange[0][1] ,
                dtype=tf.float64
            ) ,
            lambda : self.default[0][1]
            )
        self.ranges = tf.convert_to_tensor([(rGenWL , rGenWW)])
        #-----
        def cond(rng , i , img_arr , stackedWindows) :
            return i < tf.shape(rng)[0]
        def body(rng , i , img_arr , stackedWindows) :
            wl = rng[i][0]
            ww = rng[i][1]
            minHU = wl - (ww/2)
            maxHU = wl + (ww/2)
            windowChannel  = tf.expand_dims(tf.clip_by_value(img_arr , minHU , maxHU) , axis=0)
            stackedWindows = tf.concat([stackedWindows , windowChannel] , axis=0)
            i+=1
            return rng , i , img_arr , stackedWindows
        _ , _ , _ , windows = tf.while_loop(
            cond ,
            body ,
            loop_vars=[
                self.ranges ,
                0 , 
                img_arr ,
                tf.expand_dims(tf.zeros(tf.shape(img_arr) , dtype=tf.float64) , axis=0)
            ] , 
            shape_invariants = [
                self.ranges.get_shape(),
                tf.TensorShape(()) ,
                img_arr.get_shape() , 
                tf.TensorShape([None , None , None , None])
            ]
        )
        #---
        img_arr   = windows[1: , : , : , :]
        label_arr = tf.tile(tf.expand_dims(label_arr , axis=0) , [tf.shape(self.ranges)[0] , 1 , 1 , 1]) # redundant line
        #---
        return img_arr , label_arr
    def apply_default(self , img , label) :
        defaultWL = self.default[0][0]
        defaultWW = self.default[0][1]

        minHU = defaultWL - (defaultWW/2)
        maxHU = defaultWL + (defaultWW/2)

        img = tf.clip_by_value(
            img ,
            minHU ,
            maxHU
        )

        return img , label



def optimumWindowFinder(dataPath) :
    source = os.listdir(dataPath)
    allLevel  = []
    allWidth  = []

    for p in source :
        files = os.listdir(os.path.join(dataPath , p))

        # finding image and labels
        image = None
        labels = []

        for file in files :
            if "label" in file :
                labels.append(os.path.join(dataPath , p , file))
            if "merged" not in file and "label" not in file : 
                image = os.path.join(dataPath , p , file)

        # calculating window
        img = nib.as_closest_canonical(nib.load(image))
        img_arr = img.get_fdata().astype(np.float32)

        for label in labels :
            lbl     = nib.as_closest_canonical(nib.load(label))
            lbl_arr = lbl.get_fdata().astype(np.float32)

            # add hu of voi
            voi   = np.extract(lbl_arr == 1 , img_arr)
            # calculate upper bound and lower bound
            Q1   = np.percentile(voi , q=25)
            Q3   = np.percentile(voi , q=75)
            IQRT = Q3 - Q1 
            upperBound = Q3 + 1.5*IQRT
            lowerBound = Q1 - 1.5*IQRT
            meanLevel  = (upperBound + lowerBound)/2 
            allLevel.append(meanLevel)
            allWidth.append(upperBound - lowerBound)
        else :
            print(f"patinet {p} analyzed")
    
    return {
        "levelMin" : np.min(allLevel) , 
        "levelMax" : np.max(allLevel) ,
        "widthMin" : np.min(allWidth) , 
        "widthMax" : np.min(allWidth)  
    }


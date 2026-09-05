import tensorflow as tf
import tensorflow_probability as tfp
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
    def _quaritleWindowFinder(self , image_arr , label_arr) : 
        voi = tf.boolean_mask(
            image_arr , 
            label_arr == 1
        )

        min , Q1 , Q3 , max = tfp.stats.percentile(voi , q=[0. , 25. , 75. , 100.] , interpolation="nearest")

        self.ranges = [
            [min , Q1] , # low  enhance
            [Q1  , Q3] , # mid  enhance
            [Q3 , max]   # high enhance
        ]
    
    def WindowStacking(self , image_arr , label_arr) : 
        self._quaritleWindowFinder(image_arr , label_arr)
        image_arr , label_arr = super().WindowStacking(image_arr , label_arr)
        return image_arr , label_arr

class windowing :
    def __init__(self , ww , wl):
        self.wl = wl
        self.ww = ww
    def apply(self , img , label) :
        minHu = self.wl - (self.ww/2)
        maxHu = self.wl + (self.ww/2)
        
        img = tf.clip_by_value(img , minHu , maxHu)
        return img , label


class randomWindowing(windowing) : # on-fly
    def __init__(self , default , wwRange , wlRange , p_ww , p_wl): # default as tf.float32
        self.default = tf.convert_to_tensor(default , dtype=tf.float32)
        super().__init__(self.default[0] , self.default[1])

        self.wwRange = tf.convert_to_tensor(wwRange , dtype=tf.float32)
        self.wlRange = tf.convert_to_tensor(wlRange , dtype=tf.float32)
        self.p_ww    = tf.convert_to_tensor(p_ww , dtype=tf.float32)
        self.p_wl    = tf.convert_to_tensor(p_wl , dtype=tf.float32)
    def WindowStacking(self , img_arr , label_arr) : # graph compatibale
        rGenWL = tf.cond(
            tf.random.uniform(shape=() , dtype=tf.float64) <= self.p_wl ,
            lambda : tf.random.uniform(
                shape=() ,
                minval=self.wlRange[0] ,
                maxval=self.wlRange[1] ,
                dtype=tf.float32
            ),
            lambda : self.default[0]
            )
        rGenWW = tf.cond(
            tf.random.uniform(shape=() , dtype=tf.float64) <= self.p_ww ,
            lambda : tf.random.uniform(
                shape=() ,
                minval=self.wwRange[0] ,
                maxval=self.wwRange[1] ,
                dtype=tf.float32
            ) ,
            lambda : self.default[1]
            )
        #-----
        minHu = rGenWL - (rGenWW/2)
        maxHu = rGenWL + (rGenWW/2)

        img_arr   = tf.expand_dims(img_arr , axis=0)   # add channel dim
        label_arr = tf.expand_dims(label_arr , axis=0) # add channel dim

        img_arr = tf.clip_by_value(img_arr , minHu , maxHu)

        return img_arr , label_arr
    def apply_default(self , img , label) :

        img , label = self.apply(img , label)

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
        "widthMax" : np.max(allWidth)  
    }


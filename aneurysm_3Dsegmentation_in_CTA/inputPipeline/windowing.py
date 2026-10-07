import tensorflow as tf
import tensorflow_probability as tfp
import os
import nibabel as nib
import numpy as np


class multiWindowStacking : # cache
    def __init__(self  , ranges):
        if len(ranges) != 3 :
            raise Exception("just three widnow is acceptable !!!")
        self.ranges = tf.convert_to_tensor(list(ranges) , dtype=tf.float64)
    def WindowStacking(self , img_arr , label_arr) : # graph compatibale
        w1_param = self.ranges[0]
        w2_param = self.ranges[1]
        w3_param = self.ranges[2]

        w1_min = w1_param[0]
        w1_max = w1_param[1]

        w2_min = w2_param[0]
        w2_max = w2_param[1]

        w3_min = w3_param[0]
        w3_max = w3_param[1]

        img_w1 = tf.clip_by_value(
            img_arr ,
            w1_min ,
            w1_max
        )
        img_w2 = tf.clip_by_value(
            img_arr ,
            w2_min ,
            w2_max
        )
        img_w3 = tf.clip_by_value(
            img_arr ,
            w3_min ,
            w3_max
        )

        img_arr = tf.stack(
            [img_w1 , img_w2 , img_w3] ,
            axis=0
        ) # img_arr will be channelized here

        
        label_arr = tf.expand_dims(
            label_arr ,
            axis=0
        ) # label_arr will be channelized here
        

        return img_arr , label_arr


class quartileWindowStacking(multiWindowStacking) :
    def __init__(self) :
        pass
    def _quaritleWindowFinder(self , image_arr , label_arr) : 
        label_arr.set_shape([None , None , None])
        
        voi = tf.boolean_mask(
            image_arr , 
            label_arr == 1
        )

        Q_stats = tfp.stats.percentile(voi , q=[0. , 25. , 75. , 100.] , interpolation="nearest")
        min = Q_stats[0]
        Q1  = Q_stats[1]
        Q3  = Q_stats[2] 
        max = Q_stats[3] 

        ranges = [
            [min , Q1] , # low  enhance
            [Q1  , Q3] , # mid  enhance
            [Q3  , max]  # high enhance
        ]
        
        return ranges
    
    def WindowStacking(self , image_arr , label_arr) : 
        ranges = self._quaritleWindowFinder(image_arr , label_arr)
        super().__init__(ranges)
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


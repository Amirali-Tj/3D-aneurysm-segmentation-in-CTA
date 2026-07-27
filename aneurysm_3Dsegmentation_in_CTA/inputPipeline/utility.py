import nibabel as nib
import numpy as np
from scipy import ndimage
import tensorflow as tf


def totalProbabilityForAug(total_proba , n_stage) : # return per stage proba when you want to set total augmentation probability
    perStageProba = 1 - (tf.pow(total_proba , 1/n_stage))
    return perStageProba

def read_img(imgPath , labelPath , mergePath) : # fixed
    imgPath   = imgPath.numpy().decode("utf_8")
    labelPath = labelPath.numpy().decode("utf_8")
    mergePath = mergePath.numpy().decode("utf_8")

    img   = nib.as_closest_canonical(nib.load(imgPath)) # standard image shape
    label = nib.as_closest_canonical(nib.load(labelPath))
    imgTensor   = img.get_fdata()
    labelTesnor = label.get_fdata()

    mergeTensor = tf.cond(
        mergePath != "NoMerge" ,
        lambda : nib.as_closest_canonical(nib.load(mergePath)).get_fdata() , 
        lambda : tf.convert_to_tensor(
            [0] ,
            dtype=tf.float64
        )
    )

    return imgTensor, labelTesnor , mergeTensor

class volume_crop : # make it graph compatable
    def __init__(self , cubeDim):
        self.cubeDim = cubeDim
    def _volCenterExtract(self , label_arr) :
        planeZero   = tf.where(label_arr == 0 , False , True)
        zPlaneLabel = tf.math.reduce_any(planeZero , axis=[0 , 1] , keepdims=False)
        yPlaneLabel = tf.math.reduce_any(planeZero , axis=[0 , 2] , keepdims=False)
        xPlaneLabel = tf.math.reduce_any(planeZero , axis=[1 , 2] , keepdims=False)

        def border(labelPlane) :
            i_min = tf.math.argmax(labelPlane , axis=0 , output_type=tf.int32)
            i_max = tf.shape(labelPlane)[0] - tf.math.argmax(tf.reverse(labelPlane , axis=[0]) , axis=0 , output_type=tf.int32) - 1
            return i_min , i_max

        zMin , zMax = border(zPlaneLabel)
        yMin , yMax = border(yPlaneLabel)
        xMin , xMax = border(xPlaneLabel)
        volumeCenter = (tf.cast((xMin + xMax)/2 , dtype=tf.int32) , tf.cast((yMin + yMax)/2 , dtype=tf.int32) , tf.cast((zMin + zMax)/2 , dtype=tf.int32))
        return volumeCenter
  
    def cropping(self , img_arr , label_arr , merge_arr) : # add tf.cond
        volumeCenter = self._volCenterExtract(label_arr)
        xCenter , yCenter , zCenter     = volumeCenter
        xCubeDim , yCubeDim , zCubeDim  = self.cubeDim
        def HalfDims(dim) :
            return tf.cast(dim/2 , dtype=tf.int32) , tf.cast((dim - 1)/2 , dtype=tf.int32)

        xHalfDim = HalfDims(xCubeDim)
        yHalfDim = HalfDims(yCubeDim)
        zHalfDim = HalfDims(zCubeDim)

        label_arr = tf.cond(
            tf.math.not_equal(
                tf.size(merge_arr) , 
                1
            ) ,
            lambda : merge_arr ,
            lambda : label_arr
        )

        zCropMinIx , zCropMaxIx = tf.cond(
            tf.equal(tf.math.floormod(zCubeDim , 2) , 0) ,
            lambda : (zCenter - zHalfDim[0] , zCenter + zHalfDim[0]) , 
            lambda : (zCenter - zHalfDim[1] , zCenter + zHalfDim[1] + 1)
        )
        yCropMinIx  , yCropMaxIx = tf.cond(
            tf.equal(tf.math.floormod(yCubeDim , 2) , 0) ,
            lambda : (yCenter - yHalfDim[0] , yCenter + yHalfDim[0]) , 
            lambda : (yCenter - yHalfDim[1] , yCenter + yHalfDim[1] + 1)
        )
        xCropMinIx  , xCropMaxIx = tf.cond(
            tf.equal(tf.math.floormod(xCubeDim , 2) , 0) ,
            lambda : (xCenter - xHalfDim[0] , xCenter + xHalfDim[0]) , 
            lambda : (xCenter - xHalfDim[1] , xCenter + xHalfDim[1] + 1)
        )
        #---- dynamic padding
        # Z padding
        img_arr , label_arr = tf.cond(
            zCropMaxIx > tf.shape(img_arr)[2] , 
            lambda : (
                tf.concat([img_arr , tf.cast(tf.fill([tf.shape(img_arr)[0] , tf.shape(img_arr)[1] , zCropMaxIx - tf.shape(img_arr)[2]] , -1024) , dtype=tf.float64)] , axis=2) ,
                tf.concat([label_arr , tf.cast(tf.fill([tf.shape(img_arr)[0] , tf.shape(img_arr)[1] , zCropMaxIx - tf.shape(img_arr)[2]] , 0) , dtype=tf.float64)] , axis=2) ,
                ) ,
            lambda : (
                img_arr ,
                label_arr ,
                )
        )
        img_arr , label_arr , zCenter = tf.cond(
            zCropMinIx < 0 , 
            lambda : (
                tf.concat([tf.cast(tf.fill([tf.shape(img_arr)[0] , tf.shape(img_arr)[1] , -1*zCropMinIx] , -1024) , dtype=tf.float64) , img_arr] , axis=2) ,
                tf.concat([tf.cast(tf.fill([tf.shape(img_arr)[0] , tf.shape(img_arr)[1] , -1*zCropMinIx] , 0) , dtype=tf.float64) , label_arr] , axis=2) , 
                zCenter + -1*zCropMinIx
                ) ,
            lambda : (
                img_arr ,
                label_arr ,
                zCenter
                )
        )
        # Y padding
        img_arr , label_arr = tf.cond(
            yCropMaxIx > tf.shape(img_arr)[1] , 
            lambda : (
                tf.concat([img_arr , tf.cast(tf.fill([tf.shape(img_arr)[0] , yCropMaxIx - tf.shape(img_arr)[1] , tf.shape(img_arr)[2]] , -1024) , dtype=tf.float64)] , axis=1) ,
                tf.concat([label_arr , tf.cast(tf.fill([tf.shape(img_arr)[0] , yCropMaxIx - tf.shape(img_arr)[1] , tf.shape(img_arr)[2]] , 0) , dtype=tf.float64)] , axis=1) ,
                ) ,
            lambda : (
                img_arr ,
                label_arr ,
                )
        )

        img_arr , label_arr , yCenter = tf.cond(
            yCropMinIx < 0 , 
            lambda : (
                tf.concat([tf.cast(tf.fill([tf.shape(img_arr)[0] , -1*yCropMinIx , tf.shape(img_arr)[2]] , -1024) , dtype=tf.float64) , img_arr] , axis=1) ,
                tf.concat([tf.cast(tf.fill([tf.shape(img_arr)[0] , -1*yCropMinIx , tf.shape(img_arr)[2]] , 0) , dtype=tf.float64) , label_arr] , axis=1) ,
                yCenter + -1*yCropMinIx
                ) ,
            lambda : (
                img_arr ,
                label_arr ,
                yCenter
                )
        )

        # X padding
        img_arr , label_arr = tf.cond(
            xCropMaxIx > tf.shape(img_arr)[0] , 
            lambda : (
                tf.concat([img_arr , tf.cast(tf.fill([xCropMaxIx - tf.shape(img_arr)[0] , tf.shape(img_arr)[1] , tf.shape(img_arr)[2]] , -1024) , dtype=tf.float64)] , axis=0) ,
                tf.concat([label_arr , tf.cast(tf.fill([xCropMaxIx - tf.shape(img_arr)[0] , tf.shape(img_arr)[1] , tf.shape(img_arr)[2]] , 0) , dtype=tf.float64)] , axis=0) ,
                ) ,
            lambda : (
                img_arr ,
                label_arr ,
                )
        )

        img_arr , label_arr , xCenter = tf.cond(
            xCropMinIx < 0 , 
            lambda : (
                tf.concat([tf.cast(tf.fill([-1*xCropMinIx , tf.shape(img_arr)[1] , tf.shape(img_arr)[2]] , -1024) , dtype=tf.float64) , img_arr] , axis=0) ,
                tf.concat([tf.cast(tf.fill([-1*xCropMinIx , tf.shape(img_arr)[1] , tf.shape(img_arr)[2]] , 0) , dtype=tf.float64) , label_arr] , axis=0) ,
                xCenter + -1*xCropMinIx
                ) ,
            lambda : (
                img_arr ,
                label_arr ,
                xCenter
                )
        )

        lxDimIx  , rxDimIx = tf.cond(
            tf.equal(tf.math.floormod(xCubeDim , 2) , 0) ,
            lambda : (xCenter - xHalfDim[0] , xCenter + xHalfDim[0]) , 
            lambda : (xCenter - xHalfDim[1] , xCenter + xHalfDim[1] + 1)
        )
        lyDimIx  , ryDimIx = tf.cond(
            tf.equal(tf.math.floormod(yCubeDim , 2) , 0) ,
            lambda : (yCenter - yHalfDim[0] , yCenter + yHalfDim[0]) , 
            lambda : (yCenter - yHalfDim[1] , yCenter + yHalfDim[1] + 1)
        )
        lzDimIx  , rzDimIx = tf.cond(
            tf.equal(tf.math.floormod(zCubeDim , 2) , 0) ,
            lambda : (zCenter - zHalfDim[0] , zCenter + zHalfDim[0]) , 
            lambda : (zCenter - zHalfDim[1] , zCenter + zHalfDim[1] + 1)
        )

        labelCr = label_arr[lxDimIx:rxDimIx , lyDimIx:ryDimIx , lzDimIx:rzDimIx]
        imgCr   = img_arr[lxDimIx:rxDimIx , lyDimIx:ryDimIx , lzDimIx:rzDimIx]
        return imgCr , labelCr
        
        
#-----------------------------
class randomGeo : # should be wrapped with py func
    def __init__(self , p):
        self.p = p
    def rot(self , img_arr , label_arr , * , imgOrder , lblOrder , imgCval , lblCval) :
        angle = tf.random.uniform(shape=() , minval=0 , maxval=360 , dtype=tf.int32)
        chance = tf.random.uniform(shape=()  , dtype=tf.float32)
        img_arr , label_arr = tf.cond(
            chance <= self.p ,
            lambda : (
                ndimage.rotate(
                    img_arr ,
                    angle ,
                    axes=(0 , 1) ,
                    reshape=False,
                    order=imgOrder ,
                    mode='constant' ,
                    cval=imgCval
                ) , 
                ndimage.rotate(
                    label_arr ,
                    angle ,
                    axes=(0 , 1) ,
                    reshape=False,
                    order=lblOrder ,
                    mode='constant' ,
                    cval=lblCval
                )
            ) , 
            lambda : (img_arr , label_arr)
        )
        return img_arr , label_arr
    
    def flipX(self , img_arr , label_arr) : 
        chance = tf.random.uniform(shape=()  , dtype=tf.float32)
        img_arr , label_arr = tf.cond(
            chance <= self.p ,
            lambda : (
                tf.reverse(
                    img_arr ,
                    axis=[1]
                ) ,
                tf.reverse(
                    label_arr ,
                    axis=[1]
                )
            ) ,
            lambda : (img_arr , label_arr)
        )
        return img_arr , label_arr
    
    def flipY(self , img_arr , label_arr) : 
        chance = tf.random.uniform(shape=()  , dtype=tf.float32)
        img_arr , label_arr = tf.cond(
            chance <= self.p ,
            lambda : (
                tf.reverse(
                    img_arr ,
                    axis=[2]
                ) ,
                tf.reverse(
                    label_arr ,
                    axis=[2]
                )
            ) ,
            lambda : (img_arr , label_arr)
        )
        return img_arr , label_arr
    
    def flipZ(self , img_arr , label_arr) : 
        chance = tf.random.uniform(shape=()  , dtype=tf.float32)
        img_arr , label_arr = tf.cond(
            chance <= self.p ,
            lambda : (
                tf.reverse(
                    img_arr ,
                    axis=[0]
                ) ,
                tf.reverse(
                    label_arr ,
                    axis=[0]
                )
            ) ,
            lambda : (img_arr , label_arr)
        )
        return img_arr , label_arr
    
# vectorize functions

def normalize(img , label) :
    max   = tf.math.reduce_max(
        img , 
        axis=[1 , 2 , 3 , 4] ,
        keepdims=True
    )
    min   = tf.math.reduce_min(
        img ,
        axis=[1 , 2 , 3 , 4] ,
        keepdims=True
    )
    n_img = (img - min)/(max - min)
    return n_img , label

class channelOps :
    def __init__(self):
        pass

    def add_channel_dim(self , img , label) :
        img   = tf.expand_dims(img   , axis=0)
        label = tf.expand_dims(label , axis=0)
        return img , label
    
    def convert_to_channel_last(self , img , label) :
        img = tf.transpose(
            img , 
            perm = [0 , 2 , 3 , 4 , 1]
        )

        label = tf.transpose(
            label ,
            perm = [0 , 2 , 3 , 4 , 1]
        )
        return img , label



def channelize(img , label) :
    img   = tf.expand_dims(img   , axis=0)
    label = tf.expand_dims(label , axis=0)
    return img , label
 
def convert_to_channel_last(img , label) :
    img = tf.transpose(
        img , 
        perm = [0 , 2 , 3 , 4 , 1]
    )

    label = tf.transpose(
        label ,
        perm = [0 , 2 , 3 , 4 , 1]
    )
    return img , label

class tile :
    def __init__(self , tile_dim):
        self.tile_dim = tile_dim
    def tile(self , img , label) :
        img   = tf.tile(img , self.tile_dim)
        #label = tf.tile(label , self.tile_dim)
        return img , label
    
def cast32(img , label) :
    img   = tf.cast(img   , dtype=tf.float32)
    label = tf.cast(label , dtype=tf.float32)

    return img , label

def cast16(img , label) :
    img   = tf.cast(img   , dtype=tf.float16)
    label = tf.cast(label , dtype=tf.float16)
    
    return img , label


class setShape :
    def __init__(self , imgShape  , labelShape):
        self.imgShape   = imgShape
        self.labelShape = labelShape
    def set(self , img , label) :
        img.set_shape(self.imgShape)
        label.set_shape(self.labelShape)
        
        return img , label
    
    
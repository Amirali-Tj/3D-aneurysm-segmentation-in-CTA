import nibabel as nib
import numpy as np
from scipy import ndimage
import tensorflow as tf


def perStageProbaForAug(total_proba , n_stage) : # return per stage proba when you want to set total augmentation probability
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

    if mergePath != "NoMerge" :
        mergeTensor = nib.as_closest_canonical(nib.load(mergePath)).get_fdata()
    else : 
        mergeTensor = np.array([0] , dtype=np.float64)


    return imgTensor, labelTesnor , mergeTensor

class volume_crop : # make it graph compatable
    def __init__(self , cubeDim , paddDim):
        self.cubeDim = cubeDim
        self.paddDim = paddDim
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

        # static padding for efficeint calculation
        padd_dims = tf.constant([[self.paddDim , self.paddDim] , [self.paddDim , self.paddDim] , [self.paddDim , self.paddDim]])

        xCenter = xCenter + self.paddDim
        yCenter = yCenter + self.paddDim
        zCenter = zCenter + self.paddDim

        img_arr = tf.pad(
           img_arr , 
           padd_dims , 
           mode="CONSTANT" ,
           constant_values=-1024
        )

        label_arr = tf.pad(
           label_arr , 
           padd_dims , 
           mode="CONSTANT" ,
           constant_values=0
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
class randomGeo : 
    def __init__(self , p) :
        self.p = p
    def rot(self , img_arr , label_arr , * , imgOrder , lblOrder , imgCval , lblCval) : # should be wrapped with py func
        angle  = np.random.uniform(low=0 , high=360)
        chance = np.random.uniform()

        if chance < self.p : 
            img_arr = ndimage.rotate(
                                img_arr ,
                                angle ,
                                axes=(0 , 1) ,
                                reshape=False,
                                order=imgOrder ,
                                mode='constant' ,
                                cval=imgCval
                            )
            label_arr = ndimage.rotate(
                                label_arr ,
                                angle ,
                                axes=(0 , 1) ,
                                reshape=False,
                                order=lblOrder ,
                                mode='constant' ,
                                cval=lblCval
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
    
    
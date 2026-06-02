import nibabel as nib
import numpy as np
from scipy import ndimage
#import tensorflow_addons as tf_add
import tensorflow as tf
src   = "sample/2 segments on top/925049-ZINATHAMZEHEI/925049-ZINATHAMZEHEI_Brain_-CTA_20151215001854_4.nii"
seg = "sample/2 segments on top/925049-ZINATHAMZEHEI/925049-ZINATHAMZEHEI_Brain_-CTA_20151215001854_4_2823-label.nii"

# 1 - CT images are 16 bit integer of SV in dicom CT images need rescale to to actual HU of CT images (no change in dicom to nifti process as all slices are same)
# 2 - get_fdata automatically apply rescaling
# 3 - graph is language-indepandant data structure (C++), sees full plan at once (computation and data) which make parallism more efficiently and speedy
# 4 - challenges (reading file and loading to graph which is more speedy if possible) , (compatibility of functions and graph)
# 5 - pipline : volcrop --> cache on SSD --> random geo aug --> random window aug
# 6 - sequential and parallel running (simple core as worker vs multi worker on different core)
# 7 - ram consumtion ==> operations , batchs(heart beat or flat cache) , cache , shuffle buffer
# 8 - shapes can be None but values couldent so it will throw an error in trace mode of graph , sometimes error are run times
# 9 - compilation errors on datatype

# debug shape problem
def read_img(imgPath) : # fixed
    imgPath , labelPath = imgPath
    img   = nib.as_closest_canonical(nib.load(imgPath)) # standard image shape
    label = nib.as_closest_canonical(nib.load(labelPath))
    imgTensor   = img.get_fdata()
    labelTesnor = label.get_fdata()
    return imgTensor, labelTesnor , img , label

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
  
    def cropping(self , img_arr , label_arr) :
        volumeCenter = self._volCenterExtract(label_arr)
        xCenter , yCenter , zCenter     = volumeCenter
        xCubeDim , yCubeDim , zCubeDim  = self.cubeDim
        def HalfDims(dim) :
            return tf.cast(dim/2 , dtype=tf.int32) , tf.cast((dim - 1)/2 , dtype=tf.int32)

        xHalfDim = HalfDims(xCubeDim)
        yHalfDim = HalfDims(yCubeDim)
        zHalfDim = HalfDims(zCubeDim)

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
    def __init__(self , p , * , imgOrder , lblOrder , imgCval , lblCval):
        self.p = p
        self.imgorder = imgOrder
        self.lblorder = lblOrder
        self.imgCval  = imgCval
        self.lblCval  = lblCval
    def rot(self , img_arr , label_arr) :
        angle = tf.random.uniform(shape=() , minval=0 , maxval=360 , dtype=tf.int32)
        self.chance = tf.random.uniform(shape=()  , dtype=tf.float32)
        img_arr , label_arr = tf.cond(
            self.chance <= self.p ,
            lambda : (
                ndimage.rotate(
                    img_arr ,
                    angle ,
                    axes=(0 , 1) ,
                    reshape=False,
                    order=self.imgorder ,
                    mode='constant' ,
                    cval=self.imgCval
                ) , 
                ndimage.rotate(
                    label_arr ,
                    angle ,
                    axes=(0 , 1) ,
                    reshape=False,
                    order=self.lblorder ,
                    mode='constant' ,
                    cval=self.lblCval
                )
            ) , 
            lambda : (img_arr , label_arr)
        )
        return img_arr , label_arr
    
    def flip(self , img_arr , label_arr) : 
        chance = tf.random.uniform(shape=()  , dtype=tf.float32)
        axis   = tf.random.uniform(shape=() , minval=0 , maxval=3 , dtype=tf.int32)
        img_arr , label_arr = tf.cond(
            chance <= self.p ,
            lambda : (
                tf.reverse(
                    img_arr ,
                    axis=[axis]
                ) ,
                tf.reverse(
                    label_arr ,
                    axis=[axis]
                )
            ) ,
            lambda : (img_arr , label_arr)
        )
        return img_arr , label_arr

#------
class multiWindowStacking : # cache
    def __init__(self  , *ranges):
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
        label_arr = tf.tile(tf.expand_dims(label_arr , axis=0) , [tf.shape(self.ranges)[0] , 1 , 1 , 1])
        #---
        return img_arr , label_arr
#---
#------------- test area
# cropped ------------------- 
img_arr , label_arr , img , label = read_img((src , seg))
img_arr   = tf.convert_to_tensor(img_arr)
label_arr = tf.convert_to_tensor(label_arr)
volume = volume_crop((128 , 128 , 128))
imgCr , labelCr = volume.cropping(img_arr , label_arr)

#geometrical aug
geo = randomGeo(
    p=0.5 , # set chance
    imgOrder=1 ,
    lblOrder=0 ,
    imgCval=-1024,
    lblCval=0
) 
imgCrRot , labelCrRot = geo.rot(imgCr , labelCr)
imgCrRotFlip , labelCrRotFlip = geo.flip(imgCrRot , labelCrRot) # bug source
# window stacking

windower = randomMultiWindowStackig(
    default = (200 , 620) ,
    wlRange=(170 , 225) ,
    wwRange=(600 , 650) ,
    p_wl=1 ,
    p_ww=1
)

#windower = multiWindowStacking((200 , 620))
imgWindowed, labelWindowed = windower.WindowStacking(tf.convert_to_tensor(imgCrRotFlip) , tf.convert_to_tensor(labelCrRotFlip))

imgni   = nib.Nifti1Image(imgWindowed[0]   , affine=img.affine   , header=img.header)
labelni = nib.Nifti1Image(labelWindowed[0] , affine=img.affine , header=img.header)
nib.save(labelni , "W3_Label.nii")
nib.save(imgni, "W3_Image.nii")


# setting up the pipeline
'''
imgni   = nib.Nifti1Image(imgWindowed[1]   , affine=img.affine   , header=img.header)
labelni = nib.Nifti1Image(labelWindowed[1] , affine=img.affine , header=img.header)
nib.save(labelni , "new_label_1_W2.nii")
nib.save(imgni, "new_img_1_W2.nii")


imgni   = nib.Nifti1Image(imgWindowed[2]   , affine=img.affine   , header=img.header)
labelni = nib.Nifti1Image(labelWindowed[2] , affine=img.affine , header=img.header)
nib.save(labelni , "new_label_1_W3.nii")
nib.save(imgni, "new_img_1_W3.nii")
'''
    
    
    
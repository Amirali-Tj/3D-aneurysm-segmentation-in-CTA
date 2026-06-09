import keras
import tensorflow as tf
import surface_distance as sd
import numpy as np
# cheking for datatypes

def decimal_numbers(num , decimalNum) :
    modNum = tf.multiply(
        num , 
        tf.cast(tf.math.pow(10 , decimalNum) , dtype=tf.float64)
    )
    intPart = tf.cast(modNum , dtype=tf.int32)
    return tf.math.divide(
        intPart ,
        tf.math.pow(10 , decimalNum)
    ) # return tf.float64


def V_Recall(gr , pr) :   # lesion voxel detection rate
    pr = tf.where(pr >=0.5 , 1.0 , 0.0)
    tPos  = tf.math.multiply(gr , pr)
    NtPos = tf.math.count_nonzero(
       tPos , 
       axis=[1 , 2 , 3 , 4] ,
       keepdims=True , 
       dtype=tf.int32
    )
    NgrPos = tf.math.count_nonzero(
       gr , 
       axis=[1 , 2 , 3 , 4] ,
       keepdims=True , 
       dtype=tf.int32
    )
    V_Recall = tf.math.divide(NtPos , NgrPos)*100
    return decimal_numbers(V_Recall , 3) #tf.float64    

def V_accuracy(gr , pr) : #voxel detection rate (lesion and backgeound)
    pr = tf.where(pr >=0.5 , 1.0 , 0.0)
    tPos  = tf.math.multiply(gr , pr)
    NtPos = tf.math.count_nonzero(
       tPos , 
       axis=[1 , 2 , 3 , 4] , # batch dim
       keepdims=True , 
       dtype=tf.int32
    )
    NgrPos = tf.math.count_nonzero(
       gr , 
       axis=[1 , 2 , 3 ,4] ,
       keepdims=True , 
       dtype=tf.int32
    )
    #-----
    rPr = tf.where(pr == 0 , 1.0 , 0.0)
    rGr = tf.where(gr == 0 , 1.0 , 0.0)
    tNeg  = tf.math.multiply(rGr , rPr)
    NtNeg = tf.math.count_nonzero(
       tNeg , 
       axis=[1 , 2 , 3 ,4] , # batch dim
       keepdims=True , 
       dtype=tf.int32
    )
    NofVoxels = tf.size(tf.reduce_mean(gr , axis=0)) # num of voxels
    V_accuracy =tf.math.multiply(tf.math.divide(NtPos + NtNeg , NofVoxels) , 100)
    return decimal_numbers(V_accuracy , 3) # tf.float64

def HD(y_true , y_pred) :
    y_true = np.where( # make a bool numpy array
        y_true.numpy() == 1 ,
        True ,
        False
    )
    y_pred = np.where(
        y_pred.numpy() == 1 ,
        True ,
        False
    )
    batchHD = []
    for y_true_vol , y_pred_vol in zip(y_true , y_pred) :
        y_true3D  = y_true_vol[: , : , : , 0] # just one channel used to make it to 3d tensor
        distances = sd.compute_surface_distances(
            y_true3D , 
            y_pred_vol ,
            spacing_mm=[0.48 , 0.48 , 0.5] # averge image spacing used images . 
        )
        hausdorff = sd.compute_robust_hausdorff(
            distances , 
            percent=95
        )
        batchHD.append(hausdorff)

    return decimal_numbers(tf.reduce_mean(batchHD) , 3)
     
def dice(y_true , y_pred) :
    y_pred = tf.where(y_pred >=0.5 , 1.0 , 0.0)
    overLapArea = 2*tf.math.count_nonzero(
        tf.multiply(y_true , y_pred) ,
        axis = [1 , 2 , 3 ,4] ,
        keepdims=True ,
        dtype=tf.int32
    )
    grPos = tf.math.count_nonzero(
        y_true , 
        axis = [1 , 2 , 3 ,4] ,
        keepdims=True ,
        dtype=tf.int32
    )
    prPos = tf.math.count_nonzero(
        y_pred , 
        axis = [1 , 2 , 3 ,4] ,
        keepdims=True ,
        dtype=tf.int32
    )
    Dice_score = tf.math.divide(overLapArea , grPos + prPos)
    return decimal_numbers(Dice_score , 3) # return tf.float64


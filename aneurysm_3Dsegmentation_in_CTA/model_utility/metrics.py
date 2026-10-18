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
    return V_Recall #tf.float64    

def HD(y_true , y_pred) :
    y_true = np.where( # make a bool numpy array
        y_true == 1 ,
        True ,
        False
    )
    y_pred = np.where(
        y_pred == 1 ,
        True ,
        False
    )
    batchHD = []
    for y_true_vol , y_pred_vol in zip(y_true , y_pred) :
        y_true3D  = y_true_vol[: , : , : , 0] # just one channel used to make it to 3d tensor
        y_pred3D  = y_pred_vol[: , : , : , 0] # just one channel used to make it to 3d tensor
        distances = sd.compute_surface_distances(
            y_true3D , 
            y_pred3D ,
            spacing_mm=[0.48 , 0.48 , 0.5] # averge image spacing used images . 
        )
        hausdorff = sd.compute_robust_hausdorff(
            distances , 
            percent=95
        )
        batchHD.append(hausdorff)

    return tf.convert_to_tensor(batchHD)
     
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
    return Dice_score # return tf.float64

def precision(gr , pr) :
    pr = tf.where(pr >=0.5 , 1.0 , 0.0)
    tPos  = tf.math.multiply(gr , pr)
    NtPos = tf.math.count_nonzero(
        tPos , 
        axis=[1 , 2 , 3 , 4] ,
        keepdims=True , 
        dtype=tf.int32
    )

    fPos  = tf.math.subtract(
        tf.math.subtract(gr - pr) , 
        tf.constant([1] , dtype=tf.float32)
        )
    NfPos = tf.math.count_nonzero(
        fPos ,
        axis=[1 , 2 , 3 , 4] ,
        keepdims=True ,
        dtype=tf.float32
    )

    
    precision = tf.math.divide(NtPos , (NtPos + NfPos))*100
    return precision #tf.float64


def f1_score(gr , pr) :
    v_recall = V_Recall(gr , pr)
    prec     = precision(gr , pr)

    f1_score = tf.math.divide(2*v_recall*prec , tf.math.sum(v_recall , prec))*100

    return f1_score #tf.float64


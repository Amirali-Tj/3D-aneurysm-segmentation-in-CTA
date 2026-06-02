from inputPipeline import utiliy
from inputPipeline import windowing
import tensorflow as tf
import os
from pprint import pprint
import nibabel as nib

mainDir    = "sample/Tsample"
samples    = os.listdir(mainDir)

imgLis   = []
labelLis = []
for name in samples :
    if name != ".DS_Store" :
        imgLbl = os.listdir(os.path.join(mainDir , name))
        for file in imgLbl :
            if "label" in file :
                labelLis.append(os.path.join(mainDir , name , file))
            else :
                imgLis.append(os.path.join(mainDir , name , file))
#------

# configuering
geo      = utiliy.randomGeo(p=0.5)
crop     = utiliy.volume_crop((128 , 128 , 128))
windower = windowing.randomMultiWindowStackig(
    default = (200 , 620) ,
    wlRange=(170 , 225) ,
    wwRange=(600 , 650) ,
    p_wl=1 ,
    p_ww=1
) 

# wrapping
@tf.py_function(Tout=[tf.float64 , tf.float64])
def rimg(imgPath , labelPath) :
    return utiliy.read_img(imgPath , labelPath)
def read_img(img , label) :
    imglbl = rimg(img , label)
    img   = imglbl[0]
    label = imglbl[1]
    return img , label

@tf.py_function(Tout=[tf.float64 , tf.float64])
def rotate(img , label) :
    img , label = geo.rot(
        img , 
        label ,
        imgOrder=1 ,
        lblOrder=0 ,
        imgCval=-1024 ,
        lblCval=0
    )
    img , label = geo.flip(
        img , 
        label
    )
    return img , label
def rot(img , label) :
    imglbl = rotate(img , label)
    img   = imglbl[0]
    label = imglbl[1]
    return img , label


# loading ...
dataloader = (
    tf.data.Dataset.from_tensor_slices((imgLis , labelLis))
    .map(read_img)
    .map(crop.cropping)
    .map(rot)
    .map(windower.WindowStacking)
    .map(utiliy.normalize)
)


cnt=0
for data in dataloader :
    img   = nib.load(imgLis[cnt])
    label = nib.load(labelLis[cnt])

    imgarr , labelarr = data # data loaded by loader

    imgnifti   = nib.Nifti1Image(tf.squeeze(imgarr , axis=0)   , affine=img.affine , header=img.header)
    labelnifti = nib.Nifti1Image(tf.squeeze(labelarr , axis=0) , affine=img.affine , header=img.header)

    nib.save(imgnifti , f"sample/loaded_sample/loader_{cnt}_img.nii")
    nib.save(labelnifti , f"sample/loaded_sample/loader_{cnt}_label.nii")
    cnt+=1




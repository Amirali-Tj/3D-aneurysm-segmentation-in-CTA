import segmentation_models_3D as sm3
import keras
from model_utility import utility
from pprint import pprint



# setting up parameters ==> whole work as 80 , 20 training test , following per epoch approach
# 1 - loss weights in combo mode ==> done
# 2 - number of epoch ==> 500
# 3 - compilation set up (metrics of interst ...)
# 4 - batch size
# 6 - encoder architecture and freezing layers
# 7 - run and test setup
# 8 - instance normalization or batch??



binaryFocalLoss = sm3.losses.binary_focal_loss
diceLoss        = sm3.losses.dice_loss 
weightedBinaryFocalDiceLoss = utility.WeightedSumOfLosses(binaryFocalLoss , diceLoss , alpha=0.8)

model = sm3.models.unet.Unet(
    backbone_name="seresnet18" , 
    input_shape=(128 , 128 , 128 , 3) , # alts = (64 , 64 , 64 , 3) , (64 , 64 , 64 , 1) , (128 , 128 , 128 , 1)
    classes=1 , 
    activation="sigmoid" ,
    encoder_weights="imagenet" ,
    encoder_freeze=True ,
    decoder_block_type="transpose" ,
    encoder_features=utility.encoderF_d2
)

model = utility.unfreeze_model(
    model , 
    utility.unfreeze34_border , 
    keras.src.layers.normalization.batch_normalization.BatchNormalization
)

model.summary()


import segmentation_models_3D as sm3
import keras
from pprint import pprint


class WeightedSumOfLosses(sm3.base.objects.SumOfLosses) :
    def __init__(self , l1 , l2 , alpha=[1 , 1]) :
        super().__init__(l1 , l2)
        self.alpha = alpha
    def __call__(self , gt , pr) :
        return self.alpha[0]*self.l1(gt , pr) + self.alpha[1]*self.l2(gt , pr)

   
def unfreeze_model(model , unfreeze_point_name , except_layers_type) :
    unfreeze = False
    for layer in model.layers :
        if layer.name == unfreeze_point_name :
            unfreeze = True
        if unfreeze == True and not isinstance(layer , except_layers_type):
            layer.trainable = True

    if unfreeze == False :
        raise TypeError("Invalid unfreeze point name !!!")
    return model



# skip connections
encoderF_d4_res18 = [ # defualt mode
    "stage4_unit1_relu1" ,
    "stage3_unit1_relu1" ,
    "stage2_unit1_relu1" ,
    "relu0" ,
]
encoderF_d3_res18 = [
    "stage4_unit1_relu1" ,
    "stage3_unit1_relu1" ,
    "stage2_unit1_relu1" ,
]
encoderF_d3_res18 =[
    "stage4_unit1_relu1" ,
    "stage3_unit1_relu1" ,
]

encoderF_d4_res50 = [
    "activation_65" ,
    "activation_35" , 
    "activation_15" ,
    "activation" ,
]
encoderF_d3_res50 = [
    "activation_65" ,
    "activation_35" , 
    "activation_15" ,
]
encoderF_d2_res50 = [
    "activation_65" ,
    "activation_35" , 
]


# Res18 Conv_3 and Conv_4 Group 

unfreeze_Conv2_unit1ToEnd_res18   = "stage2_unit1_conv1"
unfreeze_Conv2_unit2ToEnd_res18   = "stage2_unit2_conv1"
unfreeze_Conv3_unit1ToEnd_res18   = "stage3_unit1_conv1"
unfreeze_Conv3_unit2ToEnd_res18   = "stage3_unit2_conv1"
unfreeze_Conv4_unit1ToEnd_res18   = "stage4_unit1_conv1"
unfreeze_Conv4_unit2ToEnd_res18   = "stage4_unit2_conv1" 

# Res50 Conv_5 and Conv_4 Group

unfreeze_Conv4_unit5ToEnd_res50   = "conv3d_59"
unfreeze_Conv4_unit6ToEnd_res50   = "conv3d_64"
unfreeze_Conv5_unit1ToEnd_res50   = "conv3d_69"
unfreeze_Conv5_unit2ToEnd_res50   = "conv3d_75"
unfreeze_Conv5_unit3ToEnd_res50   = "conv3d_80"


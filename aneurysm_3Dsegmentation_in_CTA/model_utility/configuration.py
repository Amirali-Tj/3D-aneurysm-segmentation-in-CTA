import segmentation_models_3D as sm3
import keras
from pprint import pprint


class WeightedSumOfLosses(sm3.base.objects.SumOfLosses) :
    def __init__(self , l1 , l2 , alpha=0.5) :
        super().__init__(l1 , l2)
        self.alpha = alpha
    def __call__(self , gt , pr) :
        return self.alpha*self.l1(gt , pr) + (1 - self.alpha)*self.l2(gt , pr)

   
def unfreeze_model(model , unfreeze_point_name , except_layers_type) :
    unfreeze = False
    for layer in model.layers :
        if layer.name == unfreeze_point_name :
            unfreeze = True
        if unfreeze == True and not isinstance(layer , except_layers_type):
            layer.trainable = True
    return model



# skip connections
encoderF_d4 = [ # defualt mode
    "stage4_unit1_relu1" ,
    "stage3_unit1_relu1" ,
    "stage2_unit1_relu1" ,
    "relu0" ,
]
encoderF_d3 = [
    "stage4_unit1_relu1" ,
    "stage3_unit1_relu1" ,
    "stage2_unit1_relu1" ,
]
encoderF_d2 =[
    "stage4_unit1_relu1" ,
    "stage3_unit1_relu1" ,
]


# unfreeze layers 
unfreeze34_border  = "add_3"
unfreeze4_border   = "add_5" 

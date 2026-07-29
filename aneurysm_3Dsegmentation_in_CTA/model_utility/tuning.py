import segmentation_models_3D as sm3
import keras_tuner as kt
import tensorflow as tf
import keras
from . import metrics
from . import configuration as conf


class HyperSeUnetRes18(kt.HyperModel) :
    def __init__(self , input_shape , classes , activation , encoder_weights , encoder_freeze , encoder_features , decoder_block_type , steps):
        super().__init__()
        self.input_shape        = input_shape
        self.classes            = classes
        self.activation         = activation
        self.encoder_weights    = encoder_weights
        self.decoder_block_type = decoder_block_type
        self.encoder_features   = encoder_features
        self.steps = steps 

    def set_tuning_param(self , unfreeze_point , learning_rate , alphForFocalLoss , alphaForWeightedFocalDiceloss) :
        self.unfreeze_point = unfreeze_point
        self.learning_rate = self.learning_rate
        self.alpha1 = alphForFocalLoss
        self.alpha2 = alphaForWeightedFocalDiceloss

    def build(self , hp):
        # model definition
        res18 = sm3.Unet(
            backbone_name="seresnet18" , 
            input_shape = self.input_shape, 
            classes     = self.classes, 
            activation  = self.activation ,
            encoder_weights = self.encoder_weights ,
            encoder_freeze  = self.encoder_freeze,
            decoder_block_type = self.decoder_block_type ,
            encoder_features   = conf.encoderF_d4_res18 # set as tunable
        )

        # model unfreezing
        res18 = conf.unfreeze_model(
            res18 , 
            hp.choice("unfreeze_point" , self.unfreeze_point) , # set as tunable
            keras.src.layers.normalization.batch_normalization.BatchNormalization
        )

        # loss definition
        binaryFocalLoss = sm3.losses.BinaryFocalLoss(
            alpha=hp.choice("alpha1" , self.alpha1) # set as tunable
        )
        diceLoss        = sm3.losses.dice_loss 
        weightedBinaryFocalDiceLoss = conf.WeightedSumOfLosses(binaryFocalLoss , diceLoss , alpha=hp.choice("alpha2" , self.alpha2)) # set as tunable

        # learning rate difinition
        lrRates = [hp.choice(f"lrStep{n}" , rates) for n , rates in enumerate(self.learning_rate)]

        lr = keras.optimizers.schedules.PiecewiseConstantDecay(
            self.steps ,
            lrRates
        )

        # compilation
        res18.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr) ,
            loss = weightedBinaryFocalDiceLoss ,
            metrics=[
                metrics.V_Recall ,
                metrics.dice
            ] ,
        )

        return res18 

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            **kwargs,
        )

    
        
import segmentation_models_3D as sm3
import keras_tuner as kt
import tensorflow as tf
import keras
from . import metrics
from . import configuration as conf


class HyperUnetResNet(kt.HyperModel) :
    def __init__(self , backbone , input_shape , classes , activation , encoder_weights , encoder_freeze , encoder_features , decoder_block_type , steps):
        super().__init__()
        self.backbone           = backbone 
        self.input_shape        = input_shape
        self.classes            = classes
        self.activation         = activation
        self.encoder_weights    = encoder_weights
        self.decoder_block_type = decoder_block_type
        self.encoder_features   = encoder_features
        self.steps = steps 

    def set_tuning_param(self , unfreeze_point , learning_rate , alphForFocalLoss , alphaForWeightedFocalDiceloss) :
        self.unfreeze_point = unfreeze_point
        self.learning_rate = learning_rate
        self.alpha1 = alphForFocalLoss
        self.alpha2 = alphaForWeightedFocalDiceloss

    def build(self , hp):
        # model definition
        res = sm3.Unet(
            backbone_name = self.backbone , 
            input_shape = self.input_shape, 
            classes     = self.classes, 
            activation  = self.activation ,
            encoder_weights = self.encoder_weights ,
            encoder_freeze  = True,
            decoder_block_type = self.decoder_block_type ,
            encoder_features   = self.encoder_features
        )

        # model unfreezing
        res = conf.unfreeze_model(
            res , 
            hp.Choice("unfreeze_point" , self.unfreeze_point) , # set as tunable
            keras.src.layers.normalization.batch_normalization.BatchNormalization
        )

        # loss definition
        binaryFocalLoss = sm3.losses.BinaryFocalLoss(
            alpha=hp.Choice("alpha1" , self.alpha1) # set as tunable
        )
        diceLoss        = sm3.losses.dice_loss 
        weightedBinaryFocalDiceLoss = conf.WeightedSumOfLosses(binaryFocalLoss , diceLoss , alpha=[hp.Choice("alpha2_1" , self.alpha2[0]) , hp.Choice("alpha2_2" , self.alpha2[1])]) # set as tunable

        # learning rate difinition
        lr = keras.optimizers.schedules.PiecewiseConstantDecay(
            self.steps ,
            [
                hp.Choice("step 1" , self.learning_rate[0]) ,
                hp.Choice("step 2" , self.learning_rate[1]) ,
                hp.Choice("step 3" , self.learning_rate[2])
            ]
        )

        # compilation
        res.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr) ,
            loss = weightedBinaryFocalDiceLoss ,
            metrics=[
                metrics.V_Recall ,
                metrics.dice
            ] ,
        )

        return res 

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            **kwargs,
        )





class HyperFPNResNet(kt.HyperModel) :
    def __init__(self , backbone , input_shape , classes , activation , encoder_weights , encoder_freeze , encoder_features , pyramid_aggregation , pyramid_block_filter , steps):
        super().__init__()
        self.backbone             = backbone 
        self.input_shape          = input_shape
        self.classes              = classes
        self.activation           = activation
        self.encoder_weights      = encoder_weights
        self.pyramid_block_filter = pyramid_block_filter
        self.pyramid_agg          = pyramid_aggregation  
        self.encoder_features     = encoder_features
        self.steps = steps 

    def set_tuning_param(self , unfreeze_point , learning_rate , alphForFocalLoss , alphaForWeightedFocalDiceloss) :
        self.unfreeze_point = unfreeze_point
        self.learning_rate = learning_rate
        self.alpha1 = alphForFocalLoss
        self.alpha2 = alphaForWeightedFocalDiceloss

    def build(self , hp):
        # model definition
        res = sm3.FPN(
            backbone_name = self.backbone , 
            input_shape = self.input_shape, 
            classes     = self.classes, 
            activation  = self.activation ,
            encoder_weights  = self.encoder_weights ,
            encoder_freeze   = True ,
            encoder_features = self.encoder_features ,
            pyramid_block_filters = self.pyramid_block_filter ,
            pyramid_aggregation   = self.pyramid_agg
        )

        # model unfreezing
        res = conf.unfreeze_model(
            res , 
            hp.Choice("unfreeze_point" , self.unfreeze_point) , # set as tunable
            keras.src.layers.normalization.batch_normalization.BatchNormalization
        )

        # loss definition
        binaryFocalLoss = sm3.losses.BinaryFocalLoss(
            alpha=hp.Choice("alpha1" , self.alpha1) # set as tunable
        )
        diceLoss        = sm3.losses.dice_loss 
        weightedBinaryFocalDiceLoss = conf.WeightedSumOfLosses(binaryFocalLoss , diceLoss , alpha=[hp.Choice("alpha2_1" , self.alpha2[0]) , hp.Choice("alpha2_2" , self.alpha2[1])]) # set as tunable

        # learning rate difinition
        lr = keras.optimizers.schedules.PiecewiseConstantDecay(
            self.steps ,
            [
                hp.Choice("step 1" , self.learning_rate[0]) ,
                hp.Choice("step 2" , self.learning_rate[1]) ,
                hp.Choice("step 3" , self.learning_rate[2])
            ]
        )

        # compilation
        res.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr) ,
            loss = weightedBinaryFocalDiceLoss ,
            metrics=[
                metrics.V_Recall ,
                metrics.dice
            ] ,
        )

        return res 

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            **kwargs,
        )

    
        
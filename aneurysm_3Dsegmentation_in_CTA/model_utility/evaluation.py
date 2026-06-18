import matplotlib.pyplot as plt
import pandas as pd

def logPlot(types  , out , filepath) :
    for log in filepath :
        pd_frame = pd.read_csv(log)
        epoch    = pd_frame["epoch"].to_numpy()

        for typ in types :
            t     = pd_frame[typ].to_numpy()
            plt.plot(epoch , t)
            plt.xlabel("epoch")
            plt.ylabel(typ)
        
    plt.savefig(out , dpi=1000)

         

#logPlot(["loss" , "val_loss"] , "tuning_logs/shuffleOpt_randomOpt_B4SH160E100.png" , "tuning_logs/shuffleOpt_randomOpt_B4SH160E100.csv")
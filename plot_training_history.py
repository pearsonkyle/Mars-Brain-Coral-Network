import json
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    
    # load dct model history
    classifier = json.load(open("models/cnn_128_spatial_2_history_5.json","r"))
    ws = 512
    res = 0
    mode = "rescale" #rescale/normalize

    # load unet model metrics
    unet1 = json.load(open(f"models/encoder_{ws}_{res}_{mode}_history.json","r"))
    unet2 = json.load(open(f"models/unet_{ws}_{res}_{mode}_history.json","r"))
    unet3 = json.load(open(f"models/unet_{ws}_{res}_{mode}_ft_history.json","r"))
    
    fig,ax = plt.subplots(3,figsize=(11,9))
    epochs = 1+np.arange(len(classifier['loss']))
    e1 = 1+np.arange(len(unet1['loss']))
    e2 = len(e1)+np.arange(len(unet2['loss']))
    e3 = len(e1)+len(e2)+np.arange(len(unet3['loss']))
    #ax[0].set_title(f"Resolution: {25*(res+1)} cm/px")
    ax[0].plot(epochs,np.log2(classifier['loss']), ls='-', color='orange', label='Classifier')
    ax[0].plot(epochs,np.log2(classifier['val_loss']), ls='--', color='orange' )
    ax[0].plot(e1,np.log2(unet1['loss']), color='red', ls='-', label='Encoder')
    ax[0].plot(e2[:-1],np.log2(unet2['loss'][:-1]), color='green', ls='-', label='U-NET (Fixed Encoder)')
    ax[0].plot(e3[:-1],np.log2(unet3['loss'][:-1]), color='blue', ls='-', label='U-NET (Fine-Tune)')

    ax[0].plot(e1,np.log2(unet1['val_loss']), color='red', ls='--')
    ax[0].plot(e2,np.log2(unet2['val_loss']), color='green', ls='--')
    ax[0].plot(e3[:-1],np.log2(unet3['val_loss'][:-1]), color='blue', ls='--')

    ax[0].set_ylabel(r"Log$_{2}$(Loss)", fontsize=14)

    ax[1].plot(epochs,100*np.array(classifier['binary_accuracy']), ls='-', color='orange', label='Classifier')
    ax[1].plot(epochs,100*np.array(classifier['val_binary_accuracy']), ls='--', color='orange')

    ax[1].plot(e1,np.array(unet1['binary_accuracy'])*100, color='red', ls='-', label='Encoder')
    ax[1].plot(e2[:-1],np.array(unet2['binary_accuracy'][:-1])*100, color='green', ls='-', label='U-NET (Fixed Encoder)')
    ax[1].plot(e3[:-1],np.array(unet3['binary_accuracy'][:-1])*100, color='blue', ls='-', label='U-NET (Fine-Tune)')

    ax[1].plot(e1,np.array(unet1['val_binary_accuracy'])*100, color='red', ls='--')
    ax[1].plot(e2[:-1],np.array(unet2['val_binary_accuracy'][:-1])*100, color='green', ls='--')
    ax[1].plot(e3[:-1],np.array(unet3['val_binary_accuracy'][:-1])*100, color='blue', ls='--')
 
    ax[2].plot(e1[:-1],np.array(unet1['mean_iou'][:-1])*100, color='red', ls='-', label='Encoder')
    ax[2].plot(e2,np.array(unet2['mean_iou'][::2])*100, color='green', ls='-', label='U-NET')
    ax[2].plot(e3,np.array(unet3['mean_iou'][::2])*100, color='blue', ls='-', label='U-NET (ft)')
    ax[2].plot(e1,np.array(unet1['val_mean_iou'])*100, color='red', ls='--')
    ax[2].plot(e2,np.array(unet2['val_mean_iou'])*100, color='green', ls='--')
    ax[2].plot(e3,np.array(unet3['val_mean_iou'])*100, color='blue', ls='--')
 
    ax[1].grid(True,ls='--')
    ax[0].grid(True,ls='--')
    ax[2].grid(True,ls='--')
    ax[1].set_ylabel("Binary Accuracy (%)" , fontsize=14)
    ax[0].set_xlabel("Training Epoch", fontsize=14)
    ax[1].set_xlabel("Training Epoch", fontsize=14)
    ax[2].set_xlabel("Training Epoch", fontsize=14)
    ax[2].set_ylabel("Mean IOU", fontsize=14)
    ax[0].legend(loc='best')
    ax[1].legend(loc='best')
    plt.tight_layout()
    plt.savefig("figures/training_history.png",dpi=300,bbox_inches='tight')
    plt.show()
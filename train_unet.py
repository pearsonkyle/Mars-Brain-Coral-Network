import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import gc
import json
import time
import pickle
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV3Small
from sklearn.model_selection import train_test_split
import numpy as np
import glymur

from create_training_data import create_samples

class MeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)

# normal unet
def make_unet(input_shape=(512,512), 
                down_layer_sizes=[16, 16, 32, 64, 64],
                up_layer_sizes=[32, 32, 32, 16, 16],
                preprocessing='normalize',
                max_pooling=2, batch_normalization=True, padding='same',   
                dropout=0.2, kernel_size=3, strides=(1,1), output_channels=3):

    inputs = layers.Input(shape=input_shape)      # assumes gray scale image, 2d array
    if preprocessing == 'normalize':    
        ninputs = layers.LayerNormalization()(inputs) # mean=0, std=1
    else:
        ninputs = layers.Rescaling(scale=1./1024)(inputs) # min=0, max=1
    resize = layers.Reshape((*input_shape,1))(ninputs) # change 1 to 3 if RGB

    # convolutional layers
    lout = {}
    for i, lsize in enumerate(down_layer_sizes):
        if i == 0:
            lout[f"d_{i}_conv"] = layers.Conv2D(lsize, kernel_size, strides=strides, padding=padding, activation='relu')(
                                    layers.Conv2D(lsize, kernel_size, strides=strides, padding=padding, activation='relu')(resize))
        else:
            if batch_normalization:
                lout[f"d_{i}_conv"] = layers.Conv2D(lsize, kernel_size, strides=strides, padding=padding, activation='relu')(
                                        layers.BatchNormalization(momentum=0.75)(
                                          layers.Conv2D(lsize, kernel_size, strides=strides, padding=padding, activation='relu')(lout[f"d_{i-1}_max"])))
            else:
                lout[f"d_{i}_conv"] = layers.Conv2D(lsize, kernel_size, strides=strides, padding=padding, activation='relu')(
                                        layers.Conv2D(lsize, kernel_size, strides=strides, padding=padding, activation='relu')(lout[f"d_{i-1}_max"]))

        if max_pooling:
            lout[f"d_{i}_max"] = layers.MaxPool2D(pool_size=(max_pooling,max_pooling))(lout[f"d_{i}_conv"])

    dsl = len(down_layer_sizes)

    encoder_model = tf.keras.Model(inputs, 
        layers.Conv2D(output_channels, (1, 1), activation="sigmoid", name="prediction")(lout[f"d_{dsl-1}_conv"])) 

    # start concatenating inputs and up sample
    for i, lsize in enumerate(up_layer_sizes):

        if i >= 1:
            lout[f"u_{i}_conc"] = layers.concatenate([lout[f"u_{i-1}_max"], lout[f"d_{dsl-i}_conv"]])

        if i == 0:
            lout[f"u_{i}_conv"] = layers.Conv2D(lsize, kernel_size, strides=strides, padding=padding, activation='relu')(lout[f"d_{dsl-1}_max"])
        else:
            lout[f"u_{i}_conv"] = layers.Conv2D(lsize, kernel_size, strides=strides, padding=padding, activation='relu')(lout[f"u_{i}_conc"])

        if max_pooling:
            lout[f"u_{i}_max"] = layers.UpSampling2D(size=(max_pooling,max_pooling))(lout[f"u_{i}_conv"])

    output = layers.Conv2D(output_channels, kernel_size+kernel_size, strides=strides, padding=padding, activation='sigmoid', name='output')(lout[f"u_{i}_max"])

    return tf.keras.Model(inputs=inputs, outputs=output), encoder_model

def train_model(model, model_name, X, y, epochs, shuffle, chunksize=500, verbose=True):

    model.summary()

    all_history = {
        'loss':[],
        'binary_accuracy':[],
        'val_binary_accuracy':[],
        'mean_iou':[],
        'val_loss':[],
        'val_mean_iou':[]
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # loop over training epochs
    for i in range(epochs):
        print("Epoch:",i)

        # batch data even more to avoid overloading GPU
        for i in range(0, len(X_train), chunksize):
            chunk = slice(i,i + chunksize)

            history = model.fit(X_train[chunk], y_train[chunk], 
                            validation_data=(X_test, y_test),
                            batch_size=8, epochs=1, verbose=verbose,
                            shuffle = True)

            for k in all_history.keys():
                try:
                    all_history[k].extend(history.history[k])
                except KeyError:
                    for kk in history.history.keys():
                        if k in kk:
                            all_history[k].extend(history.history[kk])
                    # metrics for training sequentially get name altered e.g. 'mean_io_u_1'
            _ = gc.collect()

    model.save_weights(f"models/{model_name}_weights.h5")

    # create plot example
    ri = np.random.randint(X.shape[0])
    est = model.predict(np.expand_dims(X[ri],0))
    fig, ax = plt.subplots(3,2, figsize=(6,9))
    ax[0,0].imshow(X[ri])
    ax[0,0].set_title(f"image {ri}")
    try:
        ax[1,0].imshow(y[ri],vmin=0, vmax=1, cmap='jet')
        ax[2,0].imshow(est[0],vmin=0, vmax=1, cmap='jet')
    except:
        ax[1,0].imshow(y[ri,:,:,0],vmin=0,vmax=1, cmap='jet')
        ax[2,0].imshow(est[0,:,:,0],vmin=0,vmax=1, cmap='jet')

    ax[1,0].set_title("truth")
    ax[2,0].set_title("estimate")

    # save history to json
    with open(f"models/{model_name}_history.json", 'w') as f:
        json.dump(all_history, f, indent=4)

    # clean up memory
    del X_train, X_test, y_train, y_test
    _ = gc.collect()

# unet with mobilenetv3 encoder
def make_unet3(input_shape = (512,512), output_channels=1, alpha=0.5, fine_tune=False, minimalistic=True, weights=None, preprocessing='rescale'):
    inputs = layers.Input(shape=(input_shape))
    if preprocessing == 'normalize':    
        ninputs = layers.LayerNormalization()(inputs) # mean=0, std=1
    else:
        ninputs = layers.Rescaling(scale=1./1024)(inputs) # min=0, max=1
        
    resize = layers.Reshape((*input_shape,1), name="input_image")(ninputs) # change 1 to 3 if RGB

    encoder = MobileNetV3Small(
        alpha=alpha, minimalistic=minimalistic, include_top=False,
        weights=None, input_tensor=resize, classes=output_channels, pooling=None,
        dropout_rate=0.1, classifier_activation=None,
        include_preprocessing=False)

    mnet = {}
    for k,v in encoder._get_trainable_state().items():
        mnet[k.name] = k

    # input -> layer name -> output
    # 1024x1024 -> relu_16 -> 64x64
    # 1024x1024 -> relu_17 -> 32x32
    # 1024x1024 -> relu_22 -> 32x32
    encoder_output = mnet['re_lu_22'].output
    x = layers.SpatialDropout2D(0.25)(encoder_output)
    x = layers.Conv2D(output_channels, (1, 1), activation="sigmoid", name="prediction")(x)
    encoder_model = tf.keras.Model(inputs, x, name="encoder")

    skip_connection_names = ["input_image", "re_lu", "re_lu_2", "re_lu_4", "re_lu_8", "re_lu_17"]
    
    f = [16, 16, 32, 32, 48, 64]
    x = encoder_output
    for i in range(1, len(skip_connection_names)+1, 1):
        print(i)
        x_skip = mnet[skip_connection_names[-i]].output
        x = layers.Concatenate()([x, x_skip])
        
        x = layers.Conv2D(f[-i], (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        x = layers.Conv2D(f[-i], (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        if i < len(skip_connection_names):
            x = layers.UpSampling2D((2, 2))(x)

    # x = layers.Conv2D(32, (3,3), padding="same")(x)
    # x = layers.Activation("relu")(x)

    x = layers.Conv2D(output_channels, (1, 1), padding="same")(x)
    x = layers.Activation("sigmoid")(x)
    
    model = tf.keras.Model(inputs, x)
    
    # Fine tune the pre-trained layers but keep BN in inference mode
    if fine_tune:
        encoder.trainable = fine_tune
        for k,v in model._get_trainable_state().items():
            if "batch" in k.name:
                #print(k.name, k.trainable)
                k.trainable = False
    
    return model, encoder_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--size", help="size of input window", type=int, default=512)
    
    parser.add_argument("-r", "--res", help="resolution to decode JPEG2000 files at (0 is highest)", type=int,
                        default=0)
    parser.add_argument("-e", "--epochs", help="number of training epochs", type=int, default=2)

    parser.add_argument('--weights', default=None,
                        help='which pretrained weights to use for the network')

    parser.add_argument('--verbose', action='store_true', default=False, help="verbose")

    parser.add_argument('--shuffle', action='store_true', default=False, help="shuffle train/test data at each step")

    parser.add_argument('--encoder', action='store_true', default=True, 
            help="Pre-train the encoder on downsampled data before training a decoder with upscaling")

    parser.add_argument("-d", "--datadir", type=str, default="training/",
            help="Choose a directory of images to process")

    parser.add_argument("-t", "--threads", help="number of threads for background class", default=1, type=int)

    parser.add_argument("--gpu", default=4, type=int, help='specify which gpu to use')

    parser.add_argument("--preprocessing", default="rescale", type=str, help="type of preprocessing (normalize/rescale)")
    return parser.parse_args()


if __name__ == "__main__":

    # parse command line arguments
    args = parse_args()

    glymur.set_option('lib.num_threads', args.threads)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
    # Restrict TensorFlow to only use the last GPU
        try:
            if args.gpu > len(gpus):
                raise(f"gpu number {args.gpu_num} not supported")
            tf.config.set_visible_devices(gpus[args.gpu], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    print("creating samples...")
    X0, y0, classes0 = create_samples(datadir=args.datadir, size=args.size, res=args.res)

    # create samples at lower resolution
    X1, y1, classes = create_samples(datadir=args.datadir, size=args.size, res=1)

    # concatenate the two datasets
    X = np.concatenate((X0, X1))
    y = np.concatenate((y0, y1))

    unet, encoder = make_unet3(input_shape=(args.size,args.size), output_channels=y.shape[-1], preprocessing=args.preprocessing)
    encoder.summary()

    # build model name out of args
    model_name = f"unet_{args.size}_{args.res}_{args.preprocessing}"

    # create model graph
    try:
        tf.keras.utils.plot_model(unet,"unet.png",show_shapes=True)
    except:
        print("model plot failed")

    # train the encoder #################################
    encoder_name = model_name.replace("unet","encoder")
    if args.encoder:
        yr = np.zeros((X.shape[0], encoder.output.shape[1], 
                    encoder.output.shape[2], encoder.output.shape[3])) 

        # resize X and y to the size of the encoder
        print('resizing input for encoder...')
        for i in range(X.shape[0]):
            yr[i] = tf.image.resize(y[i].astype(int), encoder.output.shape[1:3]).numpy()

        encoder.compile(optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(),
                MeanIOU(num_classes=len(classes))
            ])

        print("fitting encoder")
        train_model(encoder, encoder_name, X, yr, args.epochs-1, args.shuffle, verbose=args.verbose)
        encoder.trainable = False

        # train just the decoder of unet
        unet.summary()

        unet.compile(optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(),
                MeanIOU(num_classes=len(classes))
            ])

        train_model(unet, model_name, X, y, int(args.epochs/2), args.shuffle, verbose=args.verbose)
        lr = 1e-5
    else:
        lr = 0.001
        #encoder.load_weights(f"models/{encoder_name}_weights.h5")
        encoder.trainable = False
        #print("encoder weights loaded")

    # finetune and train both sides of model ################################
    encoder.trainable = True
    unet.trainable = True # everything is trainable when finetuning

    unet.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
                tf.keras.metrics.BinaryAccuracy(),
                MeanIOU(num_classes=len(classes))
            ])

    train_model(unet, model_name+"_ft", X, y, int(args.epochs-4), args.shuffle, verbose=args.verbose)

    # compute TP, FP, FN, rates
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # create predictions for all the training data
    ytrainclass = y_train.sum(axis=(1,2)).argmax(1)
    ytestclass = y_test.sum(axis=(1,2)).argmax(1)

    # predict on training data then summarize to a class label
    train_pred = np.zeros(y_train.shape)
    for i in range(0, len(X_train), 100):
        chunk = slice(i,i + 100)
        train_pred[chunk] = unet.predict(X_train[chunk])
    ptrainclasses = train_pred.sum(axis=(1,2)).argmax(1)

    start = time.time()
    test_pred = np.zeros(y_test.shape)
    for i in range(0, len(X_test), 420):
        chunk = slice(i,i + 420)
        test_pred[chunk] = unet.predict(X_test[chunk])
    ptestclasses = test_pred.sum(axis=(1,2)).argmax(1)
    dt = time.time() - start
    print("total evaluation time:", dt)
    print("time per eval:", dt/len(X_test))
    
    history = json.load(open(f"models/{model_name}_history.json", 'r'))
    history['dt'] = dt
    history['dtper'] = dt/len(X_test)
    history['Ntrain'] = len(X_train)
    history['Ntest'] = len(X_test)

    # compute detection metrics
    pos_idx = ytrainclass==0 # brain coral
    neg_idx = ytrainclass==1 # background
    tp = 100*np.sum(ptrainclasses[pos_idx]==0)/ptrainclasses[pos_idx].shape[0]
    fn = 100*np.sum(ptrainclasses[pos_idx]==1)/ptrainclasses[pos_idx].shape[0]
    tn = 100*np.sum(ptrainclasses[neg_idx]==1)/ptrainclasses[neg_idx].shape[0]
    fp = 100*np.sum(ptrainclasses[neg_idx]==0)/ptrainclasses[neg_idx].shape[0]
    print(f"Training set: tp: {tp:.1f}%, fn: {fn:.1f}%, tn: {tn:.1f}%, fp: {fp:.1f}%")
    history['train_tp'] = tp
    history['train_fn'] = fn
    history['train_tn'] = tn
    history['train_fp'] = fp

    acc = np.sum(ptrainclasses==ytrainclass)/ptrainclasses.shape[0]
    print(f"Training accuracy: {(100*acc):.1f}%")
    print("Train size:",len(X_train))

    # compute detection metrics
    pos_idx = ytestclass==0 # brain coral
    neg_idx = ytestclass==1 # background
    tp = 100*np.sum(ptestclasses[pos_idx]==0)/ptestclasses[pos_idx].shape[0]
    fn = 100*np.sum(ptestclasses[pos_idx]==1)/ptestclasses[pos_idx].shape[0]
    tn = 100*np.sum(ptestclasses[neg_idx]==1)/ptestclasses[neg_idx].shape[0]
    fp = 100*np.sum(ptestclasses[neg_idx]==0)/ptestclasses[neg_idx].shape[0]
    print(f"Test set: tp: {tp:.1f}%, fn: {fn:.1f}%, tn: {tn:.1f}%, fp: {fp:.1f}%")
    history['test_tp'] = tp
    history['test_fn'] = fn
    history['test_tn'] = tn
    history['test_fp'] = fp
    
    acc = np.sum(ptestclasses==ytestclass)/ptestclasses.shape[0]
    print(f"Test accuracy: {(100*acc):.1f}%")
    print("Test size:",len(X_test))
    history['acc'] = acc

    # save the history
    json.dump(history, open(f"models/{model_name}_history.json", 'w'))
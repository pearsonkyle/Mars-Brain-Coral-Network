import os
import gc
import json
import time
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import MobileNetV3Small
import glymur

try:
    from jpeg2dct.numpy import loads
    from turbojpeg import TurboJPEG, TJSAMP_GRAY, TJPF_GRAY, TJFLAG_ACCURATEDCT
except:
    print("TurboJPEG not installed, DCT mode will not work")

from create_training_data import create_samples

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--mode", help="'spatial' for spatial and 'dct' for frequency domain", type=str, default="spatial")

    parser.add_argument("-s", "--size", help="Segmentation input size [px] -> S x S", type=int, default=256)

    parser.add_argument("-r", "--res", help="resolution to decode JPEG2000 files at (0 is highest)", type=int, default=2)

    parser.add_argument("-d", "--data", help="Directory of training data", type=str, default="D:/MDAP/training")
    
    parser.add_argument("-e", "--epochs", help="Number of training epochs", type=int, default=10)

    parser.add_argument("-bs", "--batch_size", help="Batch size for training epochs", type=int, default=16) 

    parser.add_argument("-th", "--threads", help="number of threads for background class", default=8, type=int)

    parser.add_argument("-v", "--verbose", help="verbosity", action="store_true")

    parser.add_argument("--gpu", default=0, type=int, help='specify which gpu to use')

    return parser.parse_args()

def blockDCT(data):
    """
    Perform block DCT on data

    Parameters
    ----------
    data : np.ndarray
        data to perform block DCT on, of shape: (N, S, S)
        where N is the number of samples and S is the size of the image
        May only work with data of 256x256...

    Returns
    -------
    np.ndarray
        block DCT coefficients of shape: (N, S/8, S/8, 64)
    """
    size = data.shape[1]
    X_dct = np.zeros((len(data), size // 8, size // 8, 64))
    encoder = TurboJPEG()
    for i in range(len(data)):
        img = encoder.encode(np.expand_dims(data[i], -1), quality=100, pixel_format=TJPF_GRAY, jpeg_subsample=TJSAMP_GRAY, flags=TJFLAG_ACCURATEDCT)
        X_dct[i], _, _ = loads(img)
    return X_dct

def reshapeCoef(coef, size):
    """
    Reshape DCT coefficients into a block
    
    Parameters
    ----------
    coef : np.ndarray
        DCT coefficients of shape: (N, S/8, S/8, 64)
    
    size : int
        size of the image
    
    Returns
    -------
    np.ndarray
        block of shape: (S, S)
    """
    block = np.empty((size, size))
    for j in range(size//8):
        block[j // 4, (8 * j) % 32:(8 * j) % 32 + 8] = coef[j // 32, j % 4, 8 * ((j % 32) // 4):8 * ((j % 32) // 4) + 8]
    return block

def reshape64Channels(block, size):
    """
    Reshape block into 64 channels
    
    Parameters
    ----------
    block : np.ndarray
        block of shape: (S, S)
    
    size : int
        size of the image
    
    Returns
    -------
    np.ndarray
        DCT coefficients of shape: (S/8, S/8, 64)
    """
    coef = np.empty((size//8, size//8, 64))
    for j in range(size//8):
        coef[j // 32, j % 4, 8 * ((j % 32) // 4):8 * ((j % 32) // 4) + 8] = block[j // 4, (8 * j) % 32:(8 * j) % 32 + 8]
    return coef

def make_rnet(input_shape=(32,32,64), output_channels=2, dropout_rate=0.15):
    """
    Make ResNet50 model

    Parameters
    ----------
    input_shape : tuple, optional
        input shape of the model, by default (32,32,64)
    
    output_channels : int, optional
        number of output channels, by default 2
    
    dropout_rate : float, optional
        dropout rate, by default 0.15

    Returns
    -------
    tf.keras.Model
        ResNet50 model
    """
    if len(input_shape) == 2: # image and not dct input
        inputs = layers.Input(shape=(input_shape))
        ninputs = layers.LayerNormalization()(inputs) # mean=0, std=1
        resize = layers.Reshape((*input_shape,1), name="input_image")(ninputs)
        base_model = ResNet50(weights=None, include_top=False, input_tensor=resize)
    else:
        base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    predictions = layers.Dense(output_channels, activation='sigmoid')(x)
    return Model(inputs=base_model.input, outputs=predictions)

def make_mnet(input_shape=(32,32), output_channels=2, alpha=1, minimalistic=True, dropout_rate=0.1):
    """
    Make MobileNetV3Small model

    Parameters
    ----------
    input_shape : tuple, optional
        input shape of the model, by default (32,32)

    output_channels : int, optional
        number of output channels, by default 1

    alpha : float, optional
        width multiplier, by default 0.5

    minimalistic : bool, optional
        use minimalistic architecture

    dropout_rate : float, optional
        dropout rate, by default 0.15

    Returns
    -------
    tf.keras.Model
        MobileNetV3Small model
    """
    if len(input_shape) == 2: # single color channel, array of shape ~(256,256)
        inputs = layers.Input(shape=(input_shape))
        ninputs = layers.LayerNormalization()(inputs) # mean=0, std=1
        resize = layers.Reshape((*input_shape,1), name="input_image")(ninputs)
        base_model = MobileNetV3Small(
            alpha=alpha, minimalistic=minimalistic, include_top=False,
            weights=None, input_tensor=resize, classes=output_channels, pooling=None,
            dropout_rate=dropout_rate, classifier_activation=None,
            include_preprocessing=False)
    else: # multiple channels for dct (32,32,64)
        base_model = MobileNetV3Small(
        input_shape=input_shape, alpha=alpha, minimalistic=minimalistic, include_top=False,
        weights=None, input_tensor=None, classes=output_channels, pooling=None, dropout_rate=dropout_rate,
        classifier_activation=None, include_preprocessing=False)

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    predictions = layers.Dense(output_channels, activation='sigmoid')(x)
    return Model(inputs=base_model.input, outputs=predictions)

def make_cnet(input_shape=(32,32,64), output_channels=1, dropout_rate=0.15):
    """
    Make CNN model

    Parameters
    ----------
    input_shape : tuple, optional
        input shape of the model, by default (32,32,64)

    output_channels : int, optional
        number of output channels, by default 1

    dropout_rate : float, optional
        dropout rate, by default 0.15

    Returns
    -------
    tf.keras.Model
        CNN model
    """
    inputs = layers.Input(shape=(input_shape))

    if len(input_shape) == 2: # single color channel, array of shape ~(256,256)
        inputs = layers.Input(shape=(input_shape))
        ninputs = layers.LayerNormalization()(inputs) # mean=0, std=1
        resize = layers.Reshape((*input_shape,1), name="input_image")(ninputs)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(resize)
    else: # DCT input like (32,32,64)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(inputs)

    # basic convolutional block
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # bottleneck convolutional block
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # bottleneck convolutional block
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # average pool + dropout
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    predictions = layers.Dense(output_channels, activation='sigmoid')(x)
    return Model(inputs=inputs, outputs=predictions)


if __name__ == '__main__':

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

    training = True   #if true, train on data and save weights. If false, load weights

    X, yfull, classes = create_samples(datadir=args.data, size=args.size, res=args.res)
    onehot = np.max(yfull, axis=(1,2))
    y = onehot.astype(int)
    X = X.astype(np.float32)

    #perform block DCT
    start = time.time()
    if args.mode == "dct":
        X = blockDCT(X) # something like (N, S/8, S/8, 64)
        print("DCT processing time: " + str(time.time()-start))
    else:
        # X = np.expand_dims(X, -1) # single color channel
        # don't change dimension from (N, S, S) to (N, S, S, 1)
        # Normalization Layer doesn't work with channels in image
        pass

    for model_type in ['mobilenet', 'cnn', 'resnet']:
        #build model
        if model_type == "resnet":
            model = make_rnet(X[0].shape, len(classes))
        elif model_type == "mobilenet":
            model = make_mnet(X[0].shape, len(classes), dropout_rate=0.1)
        elif model_type == "cnn":
            model = make_cnet(X[0].shape, len(classes), dropout_rate=0.1)

        model.summary()

        model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.BinaryAccuracy()])

        # training stuff
        all_history = {
            'loss':[],
            'binary_accuracy':[],
            'val_binary_accuracy':[],
            'val_loss':[],
        }
        chunksize = 100*args.batch_size

        #train model
        if not training:
            model.load_weights(f"models/{model_type}_{args.size}_{args.mode}_{args.res}.h5")
        else:

            # shuffle training data each time
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

            # Training epochs
            for i in range(args.epochs):
                print("Epoch:",i)
            
                # batch data even more to avoid overloading GPU
                for i in range(0, len(X_train), chunksize):
                    chunk = slice(i,i + chunksize)

                    history = model.fit(X_train[chunk], y_train[chunk], 
                                    validation_data=(X_test, y_test), shuffle=True,
                                    batch_size=args.batch_size, epochs=1, verbose=args.verbose)

                    for k in all_history.keys():
                        all_history[k].extend(history.history[k])

                # clean up memory
                _ = gc.collect()

        #evaluate and plot
        start = time.time()
        results = model.predict(X).argmax(axis=-1)
        dt = time.time() - start
        all_history['dt'] = dt
        all_history['dtper'] = dt/len(X)
        loss, acc = model.evaluate(X,y)
        all_history['Ntrain'] = len(X_train)
        all_history['Ntest'] = len(X_test)
        
        # used for measuring variance
        fcount = 0
        fname = f"models/{model_type}_{args.size}_{args.mode}_{args.res}_history_{fcount}.json"
        while os.path.exists(fname):
            fcount += 1
            fname = f"models/{model_type}_{args.size}_{args.mode}_{args.res}_history_{fcount}.json"

        #history = model.fit(X_train, y_train, epochs=args.epochs, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))
        model.save_weights(f"models/{model_type}_{args.size}_{args.mode}_{args.res}_{fcount}.h5")
        #plotLoss(history,args)

        # predict metrics
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

        print(f"{model_type}_{args.size}_{args.mode}_{args.res}")
        yt = y_train.argmax(axis=-1)
        train_pred = np.round(model.predict(X_train)).argmax(axis=-1)
        # order depends on classes
        pos_idx = yt==0 # brain coral
        neg_idx = yt==1 # background
        tp = 100*np.sum(train_pred[pos_idx]==0)/train_pred[pos_idx].shape[0]
        fn = 100*np.sum(train_pred[pos_idx]==1)/train_pred[pos_idx].shape[0]
        tn = 100*np.sum(train_pred[neg_idx]==1)/train_pred[neg_idx].shape[0]
        fp = 100*np.sum(train_pred[neg_idx]==0)/train_pred[neg_idx].shape[0]

        # save to history
        all_history['train_tp'] = tp
        all_history['train_fn'] = fn
        all_history['train_tn'] = tn
        all_history['train_fp'] = fp

        print(f"Training set: tp: {tp:.1f}%, fn: {fn:.1f}%, tn: {tn:.1f}%, fp: {fp:.1f}%")

        # accuracy
        acc = np.sum(train_pred==yt)/train_pred.shape[0]

        print(f"Training accuracy: {(100*acc):.1f}%")

        yt = y_test.argmax(axis=-1)
        test_pred = np.round(model.predict(X_test)).argmax(axis=-1)
        # order depends on classes
        pos_idx = yt==0 # brain coral
        neg_idx = yt==1 # background
        tp = 100*np.sum(test_pred[pos_idx]==0)/test_pred[pos_idx].shape[0]
        fn = 100*np.sum(test_pred[pos_idx]==1)/test_pred[pos_idx].shape[0]
        tn = 100*np.sum(test_pred[neg_idx]==1)/test_pred[neg_idx].shape[0]
        fp = 100*np.sum(test_pred[neg_idx]==0)/test_pred[neg_idx].shape[0]

        # save to history
        all_history['test_tp'] = tp
        all_history['test_fn'] = fn
        all_history['test_tn'] = tn
        all_history['test_fp'] = fp

        print(f"Testing set: tp: {tp:.1f}%, fn: {fn:.1f}%, tn: {tn:.1f}%, fp: {fp:.1f}%")

        # accuracy
        acc = np.sum(test_pred==yt)/test_pred.shape[0]

        print(f"Testing accuracy: {(100*acc):.1f}")
        all_history['acc'] = acc

        # save history to disk
        with open(fname, 'w') as f:
            json.dump(all_history, f, indent=4)
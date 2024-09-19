import os
import glob
import json
import numpy as np

jmodels = glob.glob("*.json")

#  'dt': 27.483352422714233,
#  'dtper': 0.0029634841948149916,
#  'Ntrain': 8346,
#  'Ntest': 928,
#  'train_tp': 0.0,
#  'train_fn': 100.0,
#  'train_tn': 100.0,
#  'train_fp': 0.0,
#  'test_tp': 0.0,
#  'test_fn': 100.0,
#  'test_tn': 100.0,
#  'test_fp': 0.0,
#  'acc': 0.6724137931034483

# rank models by accuracy
acc = {}
stats = {}
for model in jmodels:
    if 'unet' in model or 'encoder' in model:
        continue
    data = json.load(open(model))
    f1_score = 2 * data['test_tp'] / (2 * data['test_tp'] + data['test_fp'] + data['test_fn'])
    acc[model] = f1_score #data['acc']


    # extract size from name
    parts = model.split("_")
    size = int(parts[1])
    model_name = "_".join(parts[:3])
    if model_name not in stats:
        stats[model_name] = []

models = sorted(acc.keys(), key=lambda x: acc[x], reverse=True)

print("Acc, TP, TN, dtper, model")
for model in models:
    # extract size from name
    parts = model.split("_")
    size = int(parts[1])
    model_name = "_".join(parts[:3])

    data = json.load(open(model))
    dtper = data['dtper']
    if size == 128:
        dtper *= 8*8 # effective size is 1024x1024
    elif size == 256:
        dtper *= 4*4 # effective size is 1024x1024

    stats[model_name].append(dtper)

    f1_score = 2 * data['test_tp'] / (2 * data['test_tp'] + data['test_fp'] + data['test_fn'])
    # compute size of model weights in Mb
    #size = os.path.getsize(model.replace(".json", ".h5")) / 1e6

    #print(f"{data['acc']:.3f} {f1_score:.3f} {data['test_tp']:.3f} {data['test_tn']:.3f} {dtper:.5f} {model}")

    # model name, Training Size, Testing Size, Accuracy, F1 Score, TP, TN, FP, FN, dt per input, dt per 1Kx1K window
    #print(f"{os.path.basename(model_name).replace('_','-')} & {data['Ntrain']}& {data['Ntest']}& {f1_score:.3f}& {data['test_tp']:.1f}& {data['test_tn']:.1f}& {data['test_fp']:.1f}& {data['test_fn']:.1f} & \\\\")
    print(f"{model} & {data['Ntrain']}& {data['Ntest']}& {f1_score:.3f}& {data['test_tp']:.1f}& {data['test_tn']:.1f}& {data['test_fp']:.1f}& {data['test_fn']:.1f} & \\\\")


# for each model type, compute mean and std of dtper
for model_name in stats:
    print(f"{1./np.mean(stats[model_name]):.6f} +- {np.std(stats[model_name]):.6f} {model_name}")

"""
67.630849 +- 0.000000 cnn_128_dct
33.897412 +- 0.003867 cnn_128_spatial
27.536372 +- 0.000000 cnn_256_dct
46.367516 +- 0.008888 cnn_256_spatial
97.704327 +- 0.001355 mobilenet_128_spatial
45.871534 +- 0.000000 mobilenet_256_dct
169.033815 +- 0.002213 mobilenet_256_spatial
21.677471 +- 0.008837 resnet_128_spatial
66.511159 +- 0.000000 resnet_256_dct
35.171373 +- 0.004598 resnet_256_spatial
"""
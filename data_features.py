## AUDIO DATA PROCESSING
import os
import librosa
import pickle

import numpy as np

'''
dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension (the depth) is at index 1,
in 'tf' mode is it at index 3. It defaults to the  image_dim_ordering value found in your
Keras config file at ~/.keras/keras.json. If you never set it, then it will be "tf".

print K.image_dim_ordering()
'''

window_size = 512
work_dir = "UrbanSound8K/audio"

## This for mel spectogram resolution
n_bands = 60
n_mfcc = 40
n_frames = 40


def windows(data, n_frames):
    ws = window_size * (n_frames - 1)
    start = 0
    while start < len(data):
        yield start, start + ws, ws
        start += (ws / 2)
        ## OVERLAP OF 50%
## END windows


def extract_features():
    raw_features = []
    _labels = []

    cnt = 0
    for sub_dir in os.listdir(work_dir):
        print("Working on dir: ", sub_dir)
        for fs in os.listdir(work_dir + "/" + sub_dir):
            if ".wav" not in fs: continue
            # print("Try Loading file: ", fs)
            sound_clip, sr = librosa.load(work_dir + "/" + sub_dir + "/" + fs)
            label = fs.split('-')[1]
            print(cnt, "Try Loading file: ", fs, " class: ", label)
            cnt += 1
            ## Work of file bacthes
            for (start, end, ws) in windows(sound_clip, n_frames):
                ## Get the sound part
                signal = sound_clip[start:end]
                if len(signal) == ws:
                    mfcc_spec = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, n_mels=n_bands)
                    mfcc_spec = mfcc_spec.T.flatten()[:, np.newaxis].T
                    raw_features.append(mfcc_spec)
                    _labels.append(label)

    print("Loaded ", cnt, " files")
    ## Add a new dimension
    raw_features = np.asarray(raw_features).reshape(len(raw_features), n_mfcc, n_frames, 1)

    ## Concate 2 elements on axis=3
    _features = np.concatenate((raw_features, np.zeros(np.shape(raw_features))), axis=3)
    _features = np.concatenate((_features, np.zeros(np.shape(raw_features))), axis=3)

    for i in range(len(_features)):
        _features[i, :, :, 1] = librosa.feature.delta(order=1, data=_features[i, :, :, 0])
        _features[i, :, :, 2] = librosa.feature.delta(order=2, data=_features[i, :, :, 0])

    return np.array(_features), np.array(_labels, dtype=np.int)
## END extract_features

features, labels = extract_features()

fd = open("data_x.pkl", 'wb')
pickle.dump(features, fd)
fd2 = open("data_y.pkl", 'wb')
pickle.dump(labels, fd2)

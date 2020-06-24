import librosa
import librosa.display
import re
import hashlib
import numpy as np
import random
import os
import glob
import tensorflow as tf
import csv
import scipy
import pandas as pd
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from kapre.time_frequency import Melspectrogram, Spectrogram
from kapre.utils import Normalization2D


from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D

def load_and_preprocess_data_librosa_mel_spectrogram(file_path, n_fft, hop_length, n_mels):
    """
    Function called inside create dataset, it loads from the file a single sample and extracts the mfcc features
    :param file_path: path of the sample considered
    :param n_fft: dft frequency
    :param hop_length: length by how much the window shift at each iteration
    :param n_mels: number of MFCCs to return
    :return: mfcc features computed from the audio sample
    """
    y, sr = librosa.load(file_path, sr=16000)
    N = y.shape[0]
    print(N)
    target_size = 16000
    if N < target_size:
        tot_pads = target_size - N
        left_pads = int(np.ceil(tot_pads / 2))
        right_pads = int(np.floor(tot_pads / 2))
        y = np.pad(y, [left_pads, right_pads], mode='constant', constant_values=(0, 0))
    elif N < target_size:
        from_ = int((N / 2) - (target_size / 2))
        to_ = from_ + target_size
        y = y[from_:to_]

    librosa_melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024,
                                                     hop_length=128, power=1.0,  # window='hann',
                                                     n_mels=80, fmin=40.0, fmax=sr / 2)

    S_dB = librosa.power_to_db(librosa_melspec, ref=np.max)


    S_dB = S_dB.reshape((S_dB.shape[0], S_dB.shape[1], 1))
    S_dB = normalize_data(S_dB)

    return S_dB.astype(np.float32)



def load_and_preprocess_data_librosa(file_path, n_fft, hop_length, n_mels):
    """
    Function called inside create dataset, it loads from the file a single sample and extracts the mfcc features
    :param file_path: path of the sample considered
    :param n_fft: dft frequency
    :param hop_length: length by how much the window shift at each iteration
    :param n_mels: number of MFCCs to return
    :return: mfcc features computed from the audio sample
    """
    y, sr = librosa.load(file_path)
    N = y.shape[0]
    target_size = 22050
    if N < target_size:
        tot_pads = target_size - N
        left_pads = int(np.ceil(tot_pads / 2))
        right_pads = int(np.floor(tot_pads / 2))
        y = np.pad(y, [left_pads, right_pads], mode='constant', constant_values=(0, 0))
    elif N < target_size:
        from_ = int((N / 2) - (target_size / 2))
        to_ = from_ + target_size
        y = y[from_:to_]

    mel_features = librosa.feature.mfcc(y, sr=sr, n_mfcc=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_features = mel_features.reshape((mel_features.shape[0], mel_features.shape[1], 1))
    mel_features = normalize_data(mel_features)

    return mel_features.astype(np.float32)


def load_and_preprocess_data_python_speech_features(file_path, n_fft, hop_length, n_mels):
    """
    Function called inside create dataset, it loads from the file a single sample and extracts the mfcc features
    :param file_path: path of the sample considered
    :param n_fft: dft frequency
    :param hop_length: length by how much the window shift at each iteration
    :param n_mels: number of MFCCs to return
    :return: mfcc features computed from the audio sample
    """
    n_mels = 40
    window_duration = 0.025
    frame_step = 0.010

    (sr, y) = wav.read(file_path)
    # print(sr)
    # print(y.shape)
    N = y.shape[0]

    target_size = 16000
    if N < target_size:
        tot_pads = target_size - N
        left_pads = int(np.ceil(tot_pads / 2))
        right_pads = int(np.floor(tot_pads / 2))
        y = np.pad(y, [left_pads, right_pads], mode='constant', constant_values=(0, 0))
    elif N < target_size:
        from_ = int((N / 2) - (target_size / 2))
        to_ = from_ + target_size
        y = y[from_:to_]

    mfcc_feat = mfcc(y, sr, winlen=window_duration, winstep=0.01, numcep=n_mels, nfilt=n_mels * 2, ceplifter=0)
    mfcc_feat = -normalize_data(mfcc_feat).T
    mfcc_feat = mfcc_feat.reshape((mfcc_feat.shape[0], mfcc_feat.shape[1], 1))

    return mfcc_feat.astype(np.float32)


def normalize_data(data):
    # Amplitude estimate
    norm_factor = np.percentile(data, 99) - np.percentile(data, 5)
    return (data / norm_factor)


if __name__ == '__main__':

    dataset_path = './speech_commands_v0.02/'

    # read the list for each set and select only wanted classes
    train_reference = pd.read_csv('train_dataset.csv', index_col=0, header=None, names=['label'])
    mask = (train_reference['label'] == 'backward') | (train_reference['label'] == 'bed')
    train_reference = train_reference[mask]

    validation_reference = pd.read_csv('validation_dataset.csv', index_col=0, header=None, names=['label'])
    mask = (validation_reference['label'] == 'backward') | (validation_reference['label'] == 'bed')
    validation_reference = validation_reference[mask]

    test_reference = pd.read_csv('test_dataset.csv', index_col=0, header=None, names=['label'])
    mask = (test_reference['label'] == 'backward') | (test_reference['label'] == 'bed')
    test_reference = test_reference[mask]

    list_subfolders_with_paths = [f.path for f in os.scandir(dataset_path) if f.is_dir()]

    # compute 2 dictionaries for swiching between label in string or int
    classToNum = {}
    numToClass = {}
    num = 0

    for i in list_subfolders_with_paths:
        cl = i.split("/")[-1]
        classToNum[cl] = num
        numToClass[num] = cl
        num += 1

    # change label from string to int
    train_reference['label'] = train_reference['label'].apply(lambda l: classToNum[l])
    validation_reference['label'] = validation_reference['label'].apply(lambda l: classToNum[l])
    test_reference['label'] = test_reference['label'].apply(lambda l: classToNum[l])

    #### Start testing the preprocessing alternatives


    # initialize preprocessing variables
    n_mels = 40
    n_mfcc = 13
    window_duration = 0.025
    frame_step = 0.010
    # the sample rate is always fixed at 16000
    sample_rate = 16000

    #length of the fft window(how many time samples are in a single window)
    n_fft = int(window_duration * sample_rate)
    #how many time samples are in between the starting point of two contiguous windows
    hop_length = int(frame_step * sample_rate)

    #sample that is considered for the feature extraction visualization
    sample = train_reference.iloc[0].name

    # librosa loader has default sample rate of 22050 and we need to specify the actual sr
    y, sr = librosa.load(sample, sr=16000)

    librosaMFCC = librosa.feature.mfcc(y, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, window=scipy.signal.windows.hamming, fmin=300)
    #librosaMFCC = normalize_data(librosaMFCC)

    print(librosaMFCC.shape, "using librosa\n", librosaMFCC)

    plt.figure(figsize=(5, 3))
    plt.pcolormesh(librosaMFCC[1:,:])

    plt.title('MFCC using librosa - parameters given in class')
    plt.ylabel('MFCC Coefficients')
    plt.xlabel('Time')
    plt.colorbar()
    plt.show()

    (rate, sig) = wav.read(sample)
    mfcc_feat = mfcc(sig, rate, winlen=window_duration, winstep=frame_step, numcep=n_mfcc, nfilt=n_mels, ceplifter=0)
    mfcc_feat = normalize_data(mfcc_feat).T

    print(mfcc_feat.shape, "using python speach features\n", mfcc_feat)
    plt.figure(figsize=(5, 3))
    plt.pcolormesh(mfcc_feat[1:, :])

    plt.title('MFCC using python_speach_features - parameters given in class')
    plt.ylabel('MFCC Coefficients')
    plt.xlabel('Time')
    plt.colorbar()
    plt.show()

    y, sr = librosa.load(sample, sr=16000)
    # Check if mel spectrogram matches the one computed with librosa

    librosa_melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024,
                                                     hop_length=128, power=1.0,  # window='hann',
                                                     n_mels=80, fmin=40.0, fmax=sr/2)

    S_dB = librosa.power_to_db(librosa_melspec, ref=np.max)

    #S_dB -= (np.mean(S_dB, axis=0) + 1e-8)
    S_dB =normalize_data(S_dB)
    plt.figure(figsize=(5, 3))
    plt.pcolormesh(S_dB)
    plt.title('Mel Spectrogram - librosa')
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.colorbar()
    plt.show()



    """
    mel_librosa = load_and_preprocess_data_librosa_mel_spectrogram(sample, n_fft, hop_length, n_mels)
    mel_librosa = mel_librosa.reshape(mel_librosa.shape[0], mel_librosa.shape[1])

    plt.figure(figsize=(5, 3))
    plt.pcolormesh(mel_librosa)

    plt.title('Mel Spectrogram - librosa')
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.colorbar()
    #plt.legend()
    plt.show()

    print("max mel_librosa ",np.max(mel_librosa)," min ", np.min(mel_librosa))
    """
    melspecModel = Sequential()

    melspecModel.add(Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, 16000),
                                    padding='same', sr=rate, n_mels=80,
                                    fmin=40.0, fmax=rate / 2, power_melgram=1.0,
                                    return_decibel_melgram=True, trainable_fb=False,
                                    trainable_kernel=False,
                                    name='mel_stft'))

    melspecModel.add(Normalization2D(int_axis=0))


    melspecModel.summary()

    (rate, y) = wav.read(sample)

    print(y.shape)
    y = y.reshape((-1, 1, 16000))
    print(y.shape)
    melspec = melspecModel.predict(y)
    print(melspec.shape)

    plt.figure(figsize=(5, 3))
    plt.pcolormesh(melspec[0,:,:,0])

    plt.title('Mel Spectrogram visualization - kapre')
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.colorbar()
    plt.show()

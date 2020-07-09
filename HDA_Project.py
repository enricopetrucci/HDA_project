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
import pandas as pd
import seaborn as sn
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import math

from sklearn.metrics import confusion_matrix, roc_curve, auc
from tensorflow.keras.layers import Dropout , SpatialDropout2D, Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Add
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D

from scipy.fftpack import dct


def generate_train_val_test_list(dataset_path, name_train, name_val, name_test):
    """
    generates 3 file, one for each of train, validation and test set.
    each of these files contains the path of all the samples that are contained in each dataset
    Args:
    dataset_path: path of the dataset.
    perc_val: How much of the data set to use for validation.
    perc_test: How much of the data set to use for testing.
    name_train: name of the file that will be generated that contains the list for the train set
    name_val: name of the file that will be generated that contains the list for the validation set
    name_test: name of the file that will be generated that contains the list for the test set
    """
    # read split from files and all files in folders
    basePath = dataset_path
    test_list = pd.read_csv(basePath + 'testing_list.txt',
                           sep=" ", header=None)[0].tolist()
    val_list = pd.read_csv(basePath + 'validation_list.txt',
                          sep=" ", header=None)[0].tolist()
    for i in range(len(test_list)):
        test_list[i] = dataset_path+test_list[i]

    for i in range(len(val_list)):
        val_list[i] = dataset_path + val_list[i]

    all_list = []


    for root, dirs, files in os.walk(basePath):
        if root != (dataset_path+"_background_noise_"):
            all_list += [root + '/' + f for f in files if f.endswith('.wav')]
    train_list = list(set(all_list) - set(val_list) - set(test_list))

    silence_path = './silence'
    silence_list = []
    for root, dirs, files in os.walk(silence_path):
        silence_list += [root + '/' +
                           f for f in files if f.endswith('.wav')]

    print(silence_list)
    for i in silence_list:
        r_num = np.random.rand()
        if r_num >= 0.2:
            train_list.append(i)
        elif r_num <= 0.1:
            val_list.append(i)
        else:
            test_list.append(i)

    #print(trainWAVs)
    test_list.sort()
    val_list.sort()
    train_list.sort()


    f = open(name_train, 'w')
    for ele in train_list:
        f.write(ele + "," + ele.split('/')[-2] + '\n')

    f = open(name_val, 'w')
    for ele in val_list:
        f.write(ele + "," + ele.split('/')[-2] + '\n')

    f = open(name_test, 'w')
    for ele in test_list:
        f.write(ele + "," + ele.split('/')[-2] + '\n')

    test_set_path = './speech_commands_test_set_v0.02/'
    real_test_list = []

    for root, dirs, files in os.walk(test_set_path):
        real_test_list += [root + '/' +
                         f for f in files if f.endswith('.wav')]

    f = open('real_'+name_test, 'w')
    for ele in real_test_list:
        f.write(ele + "," + ele.split('/')[-2].replace("_", "") + '\n')


def generate_silence_samples(dataset_path):
    """
    Randomly sample the background noise files in order to
    extract one second short files that can be used during training and validation
    :param dataset_path: path in which the background noises are stored
    """
    file_list = []
    for filename in glob.glob(os.path.join(dataset_path+'_background_noise_', '*.wav')):
        y, sr = librosa.load(filename, sr = 16000)
        file_list.append(y)

    silence_dir = "./silence/"
    #os.mkdir(silence_dir)

    for i in range(4000):
        idx = np.random.randint(0, len(file_list)) #chose a file
        random_start = np.random.randint(0, len(file_list[idx]) - 16000)
        new_sample = file_list[idx][random_start:random_start + 16000]
        librosa.output.write_wav(silence_dir+str(i)+".wav", new_sample, 16000)


def load_and_preprocess_data_librosa_mel_spectrogram(file_path):
    """
    Function called inside create dataset, it loads from the file a single sample and extracts the mfcc features
    :param file_path: path of the sample considered
    :return: mel spectrogram of the audio file
    """
    y, sr = librosa.load(file_path, sr=16000)
    N = y.shape[0]
    #print(N)
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
    y = normalize_data(y)
    #y = librosa.util.normalize(y)
    librosa_melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024,
                                                     hop_length=128, power=1.0,  # window='hann',
                                                     n_mels=80, fmin=40.0, fmax=sr / 2)

    S_dB = librosa.power_to_db(librosa_melspec, ref=np.max)


    S_dB = S_dB.reshape((S_dB.shape[0], S_dB.shape[1], 1))
    result = normalize_data(S_dB)
    #result = librosa.util.normalize(S_dB)
    return result.astype(np.float32)

# compute mfcc features using librosa library
# size (13,32)
def load_and_preprocess_data_librosa_mfcc(file_path):
    """
    Function called inside create dataset, it loads from the file a single sample and extracts the mfcc features
    :param file_path: path of the sample considered
    :return: mel spectrogram of the audio file
    """
    y, sr = librosa.load(file_path, sr=16000)
    N = y.shape[0]
    #print(N)
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

    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)

    # Let's pad on the first and second deltas while we're at it
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    delta2_mfcc = delta2_mfcc.reshape((delta2_mfcc.shape[0], delta2_mfcc.shape[1], 1))
    delta2_mfcc = normalize_data(delta2_mfcc)

    return delta2_mfcc.astype(np.float32)

# compute mfcc features using python speech features library
# size (12,99)
def load_and_preprocess_data_python_speech_features_mfcc(file_path):
    sample_rate, signal = wav.read(file_path)
    N = signal.shape[0]
    # print(N)
    target_size = 16000
    if N < target_size:
        tot_pads = target_size - N
        left_pads = int(np.ceil(tot_pads / 2))
        right_pads = int(np.floor(tot_pads / 2))
        signal = np.pad(signal, [left_pads, right_pads], mode='constant', constant_values=(0, 0))
    elif N < target_size:
        from_ = int((N / 2) - (target_size / 2))
        to_ = from_ + target_size
        signal = signal[from_:to_]

    mfcc_feat = mfcc(signal, sample_rate)
    d_mfcc_feat = delta(mfcc_feat, 2)
    d_mfcc_feat = d_mfcc_feat.T
    d_mfcc_feat = d_mfcc_feat[1:13]
    d_mfcc_feat = normalize_data(d_mfcc_feat)
    d_mfcc_feat = d_mfcc_feat.reshape((d_mfcc_feat.shape[0], d_mfcc_feat.shape[1], 1))

    return d_mfcc_feat.astype(np.float32)

# compute mfcc features -> https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
# size (12,98)
def load_and_preprocess_manual_mfcc(file_path):

    sample_rate, signal = wav.read(file_path)
    N = signal.shape[0]

    target_size = 16000
    if N < target_size:
        tot_pads = target_size - N
        left_pads = int(np.ceil(tot_pads / 2))
        right_pads = int(np.floor(tot_pads / 2))
        signal = np.pad(signal, [left_pads, right_pads], mode='constant', constant_values=(0, 0))
    elif N < target_size:
        from_ = int((N / 2) - (target_size / 2))
        to_ = from_ + target_size
        signal = signal[from_:to_]

    #print(sample_rate)
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    frame_size = 0.025
    frame_stride = 0.01
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)  # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    frames *= np.hamming(frame_length)
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    nfilt = 40
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
    num_ceps = 12
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    cep_lifter = 22
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift  # *
    #print(mfcc.shape)
    mfcc = mfcc.T

    delta2_mfcc = mfcc.reshape((mfcc.shape[0], mfcc.shape[1], 1))
    delta2_mfcc = normalize_data(delta2_mfcc)
    #print(delta2_mfcc.shape)

    return delta2_mfcc.astype(np.float32)

def normalize_data(data):
    """
    normalize sample
    :param data: input data
    :return: normalized data
    """
    # Amplitude estimate
    norm_factor = np.percentile(data, 99) - np.percentile(data, 5)

    return data / norm_factor


def filter_fn(y,):
    """
    Used to filter during training,at each epoch, the number of
    samples that are considered for the "unknown" class
    :param y: input sample in the dataset
    :return: true if the sample has to be used during the current training epoch, false otherwise
    """
    return (not tf.math.equal(y, 11)) or np.random.rand() < 1 / 18


def create_dataset(reference, batch_size, shuffle, filter, repeat, cache_file=None):
    """
    Create dataset and store a cached version
    :param reference: list of the path for each sample in the dataset and their class
    :param batch_size:
    :param shuffle: shuffle the dataset before using it at each iteration
    :param filter: filter the number of unknown samples in order to keep the probability for each class uniform
    :param repeat: repeat the dataset until it is necessary
    :param cache_file: name of the cache file
    """
    file_paths = list(reference.index)
    labels = reference['label']

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    py_func = lambda file_path, label: (tf.numpy_function(load_and_preprocess_data_librosa_mel_spectrogram, [file_path], tf.float32), label)

    dataset = dataset.map(py_func, num_parallel_calls=os.cpu_count())

    # Cache dataset
    if cache_file:
        dataset = dataset.cache(cache_file)

    if filter:
        py_func = lambda x, y: (tf.numpy_function(filter_fn, [y], tf.bool))
        dataset = dataset.filter(py_func)

    # Shuffle
    if shuffle:
        dataset = dataset.shuffle(len(file_paths))

    #Repeat the dataset indefinitely
    if repeat:
        dataset = dataset.repeat()

    # Batch
    dataset = dataset.batch(batch_size=batch_size)#, drop_remainder=True)

    # Prefetch
    dataset = dataset.prefetch(buffer_size=1)

    return dataset

# first model not very useful
def modelconvNN(input_shape, classes):
    """
    :param input_shape: -- shape of the data of the dataset
    :return: model -- a tf.keras.Model() instance
    """

    X_input = tf.keras.Input(input_shape)

    X = tf.keras.layers.Conv2D(32, (8, 16), strides=(1, 4))(X_input)
    X = BatchNormalization(axis=3,)(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.SpatialDropout2D(0.2)(X)
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    X = tf.keras.layers.Conv2D(64, (4, 5), strides=(1, 1))(X)
    X = BatchNormalization(axis=3, )(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.SpatialDropout2D(0.2)(X)
    X = MaxPooling2D((2, 2), name='max_pool1')(X)

    X = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1))(X)
    X = BatchNormalization(axis=3, )(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.SpatialDropout2D(0.2)(X)
    X = MaxPooling2D((2, 2), name='max_pool2')(X)

    X = tf.keras.layers.Flatten()(X)

    X = tf.keras.layers.Dense(classes, activation='softmax', name='fc2')(X)

    # Create the keras model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = tf.keras.Model(inputs=X_input, outputs=X, name='modelconvNN')

    return model


def modelconvNN1(input_shape, classes):
    """
    :param input_shape: -- shape of the data of the dataset
    :return: model -- a tf.keras.Model() instance
    """

    X_input = tf.keras.Input(input_shape)

    X = tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 2))(X_input)
    X = BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.SpatialDropout2D(0.1)(X)
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    X = tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1))(X)
    X = BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.SpatialDropout2D(0.1)(X)
    X = MaxPooling2D((2, 2), name='max_pool1')(X)

    X = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1))(X)
    X = BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.SpatialDropout2D(0.1)(X)
    X = MaxPooling2D((2, 2), name='max_pool2')(X)

    X = tf.keras.layers.Conv2D(24, (3, 3), strides=(1, 1))(X)
    X = BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.SpatialDropout2D(0.1)(X)
    #X = MaxPooling2D((2, 2), name='max_pool3')(X)

    X = tf.keras.layers.Flatten()(X)

    X = tf.keras.layers.Dense(64, activation='relu', name='dense')(X)

    X = tf.keras.layers.Dense(classes, activation='softmax', name='fc2')(X)

    # Create the keras model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = tf.keras.Model(inputs=X_input, outputs=X, name='modelconvNN1')

    return model

# convolutional neural network from the paper
def ConvSpeechModel(input_shape, classes):
    """
    Base fully convolutional model for speech recognition
    """

    X_input = tf.keras.Input(input_shape)

    # note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    # we would rather have it the other way around for LSTMs

    x = tf.keras.layers.Permute((2, 1, 3))((X_input))
    # x = Reshape((94,80)) (x) #this is strange - but now we have (batch_size,
    # sequence, vec_dim)

    c1 = tf.keras.layers.Conv2D(20, (5, 1), activation='relu', padding='same')(x)
    c1 = tf.keras.layers.BatchNormalization()(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 1))(c1)
    p1 = tf.keras.layers.Dropout(0.03)(p1)

    c2 = tf.keras.layers.Conv2D(40, (3, 3), activation='relu', padding='same')(p1)
    c2 = tf.keras.layers.BatchNormalization()(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    p2 = tf.keras.layers.Dropout(0.01)(p2)

    c3 = tf.keras.layers.Conv2D(80, (3, 3), activation='relu', padding='same')(p2)
    c3 = tf.keras.layers.BatchNormalization()(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    p3 = tf.keras.layers.Flatten()(p3)
    p3 = tf.keras.layers.Dense(64, activation='relu')(p3)
    p3 = tf.keras.layers.Dense(32, activation='relu')(p3)

    output = tf.keras.layers.Dense(classes, activation='softmax')(p3)

    model = tf.keras.Model(inputs=[X_input], outputs=[output], name='ConvSpeechModel')

    return model

# original model from paper
# 12_cl: 0.9439672827720642  0.9359918236732483
def AttRNNSpeechModel(input_shape, classes, rnn_func=tf.keras.layers.LSTM):

    X_input = tf.keras.Input(input_shape)

    # note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    # we would rather have it the other way around for LSTMs

    x = tf.keras.layers.Permute((2, 1, 3))(X_input)

    x = tf.keras.layers.Conv2D(10, (5, 1), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(1, (5, 1), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Lambda(lambda q: tf.keras.backend.squeeze(q, -1), name='squeeze_last_dim')(x)
    x = tf.keras.layers.Bidirectional(rnn_func(64, return_sequences=True))(x)  # [b_s, seq_len, vec_dim]
    x = tf.keras.layers.Bidirectional(rnn_func(64, return_sequences=True))(x)  # [b_s, seq_len, vec_dim]

    xFirst = tf.keras.layers.Lambda(lambda q: q[:, -1])(x)  # [b_s, vec_dim]
    query = tf.keras.layers.Dense(128)(xFirst)

    # dot product attention
    attScores = tf.keras.layers.Dot(axes=[1, 2])([query, x])
    attScores = tf.keras.layers.Softmax(name='attSoftmax')(attScores)  # [b_s, seq_len]

    # rescale sequence
    attVector = tf.keras.layers.Dot(axes=[1, 1])([attScores, x])  # [b_s, vec_dim]

    x = tf.keras.layers.Dense(64, activation='relu')(attVector)

    x = tf.keras.layers.Dense(32)(x)


    output = tf.keras.layers.Dense(classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=[X_input], outputs=[output], name='AttRNNSpeechModel')

    return model

#940490 0.9413087964057922
def AttRNNSpeechModellite(input_shape, classes, rnn_func=tf.keras.layers.GRU):

    X_input = tf.keras.Input(input_shape)

    # note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    # we would rather have it the other way around for LSTMs

    x = tf.keras.layers.Permute((2, 1, 3))(X_input)

    x = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Lambda(lambda q: tf.keras.backend.squeeze(q, -1), name='squeeze_last_dim')(x)
    x = tf.keras.layers.Bidirectional(rnn_func(32,  return_sequences=True))(x)  # [b_s, seq_len, vec_dim]
    x = tf.keras.layers.Bidirectional(rnn_func(32, return_sequences=True))(x)  # [b_s, seq_len, vec_dim]

    xFirst = tf.keras.layers.Lambda(lambda q: q[:, -1])(x)  # [b_s, vec_dim]
    query = tf.keras.layers.Dense(64)(xFirst)

    # dot product attention
    attScores = tf.keras.layers.Dot(axes=[1, 2])([query, x])
    attScores = tf.keras.layers.Softmax(name='attSoftmax')(attScores)  # [b_s, seq_len]

    # rescale sequence
    attVector = tf.keras.layers.Dot(axes=[1, 1])([attScores, x])  # [b_s, vec_dim]

    x = tf.keras.layers.Dense(64, activation='relu')(attVector)

    x = tf.keras.layers.Dense(32, activation='relu')(x)


    output = tf.keras.layers.Dense(classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=[X_input], outputs=[output], name='AttRNNSpeechModelite')

    return model

#25K parameters lightest but well regularized
# 12_cl whole dataset 0.9588956832885742
# 12_cl 0.952965259552002
# 35_cl 0.9362108111381531
def AttRNNSpeechModelLightest(input_shape, classes, rnn_func=tf.keras.layers.GRU):

    X_input = tf.keras.Input(input_shape)

    # note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    # we would rather have it the other way around for LSTMs

    x = tf.keras.layers.Permute((2, 1, 3))(X_input)

    x = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = tf.keras.layers.SpatialDropout2D(0.1)(x)

    x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Lambda(lambda q: tf.keras.backend.squeeze(q, -1), name='squeeze_last_dim')(x)
    x = tf.keras.layers.Bidirectional(rnn_func(32, return_sequences=True))(x)  # [b_s, seq_len, vec_dim]
    #x = tf.keras.layers.Bidirectional(rnn_func(32, return_sequences=True))(x)  # [b_s, seq_len, vec_dim]

    xFirst = tf.keras.layers.Lambda(lambda q: q[:, -1])(x)  # [b_s, vec_dim]
    query = tf.keras.layers.Dense(64)(xFirst)

    # dot product attention
    attScores = tf.keras.layers.Dot(axes=[1, 2])([query, x])
    attScores = tf.keras.layers.Softmax(name='attSoftmax')(attScores)  # [b_s, seq_len]

    # rescale sequence
    attVector = tf.keras.layers.Dot(axes=[1, 1])([attScores, x])  # [b_s, vec_dim]

    x = tf.keras.layers.Dense(64, activation='relu')(attVector)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    output = tf.keras.layers.Dense(classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=[X_input], outputs=[output], name='AttRNNSpeechModelLightest')

    return model

# 155K parameters best accuracy for 12_cl
#12_cl 0.9527607560157776, 0.956646203994751 0.9597136974334717
def AttRNNSpeechModelaccuracybest(input_shape, classes, rnn_func=tf.keras.layers.GRU):

    X_input = tf.keras.Input(input_shape)

    # note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    # we would rather have it the other way around for LSTMs

    x = tf.keras.layers.Permute((2, 1, 3))(X_input)

    x = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Lambda(lambda q: tf.keras.backend.squeeze(q, -1), name='squeeze_last_dim')(x)
    x = tf.keras.layers.Bidirectional(rnn_func(64, return_sequences=True))(x)  # [b_s, seq_len, vec_dim]
    x = tf.keras.layers.Bidirectional(rnn_func(64, return_sequences=True))(x)  # [b_s, seq_len, vec_dim]

    xFirst = tf.keras.layers.Lambda(lambda q: q[:, -1])(x)  # [b_s, vec_dim]
    query = tf.keras.layers.Dense(128)(xFirst)

    # dot product attention
    attScores = tf.keras.layers.Dot(axes=[1, 2])([query, x])
    attScores = tf.keras.layers.Softmax(name='attSoftmax')(attScores)  # [b_s, seq_len]

    # rescale sequence
    attVector = tf.keras.layers.Dot(axes=[1, 1])([attScores, x])  # [b_s, vec_dim]

    x = tf.keras.layers.Dense(64, activation='relu')(attVector)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    output = tf.keras.layers.Dense(classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=[X_input], outputs=[output], name='AttRNNSpeechModelbest2')

    return model

# around 80K parameters a good compromise between light and accurate
# 12_cl 0.95030677318573 0.947852790355
def AttRNNSpeechModellitebest(input_shape, classes, rnn_func=tf.keras.layers.GRU):

    X_input = tf.keras.Input(input_shape)

    # note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    # we would rather have it the other way around for LSTMs

    x = tf.keras.layers.Permute((2, 1, 3))(X_input)

    x = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Lambda(lambda q: tf.keras.backend.squeeze(q, -1), name='squeeze_last_dim')(x) # since we only have one filter the output is back to being a single image
    x = tf.keras.layers.Bidirectional(rnn_func(64, return_sequences=True))(x)  # [b_s, seq_len, vec_dim]
    #x = tf.keras.layers.Bidirectional(rnn_func(64, return_sequences=True))(x)  # [b_s, seq_len, vec_dim]

    xFirst = tf.keras.layers.Lambda(lambda q: q[:, -1])(x)  # [b_s, vec_dim]
    query = tf.keras.layers.Dense(128)(xFirst)

    # dot product attention
    attScores = tf.keras.layers.Dot(axes=[1, 2])([query, x])
    attScores = tf.keras.layers.Softmax(name='attSoftmax')(attScores)  # [b_s, seq_len]

    # rescale sequence
    attVector = tf.keras.layers.Dot(axes=[1, 1])([attScores, x])  # [b_s, vec_dim]

    x = tf.keras.layers.Dense(64, activation='relu')(attVector)

    x = tf.keras.layers.Dense(32, activation='relu')(x)


    output = tf.keras.layers.Dense(classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=[X_input], outputs=[output], name='AttRNNSpeechModelitebest')

    return model


# From the paper, poor accuracy, not useful
def RNNSpeechModel(input_shape, classes):

    X_input = tf.keras.Input(input_shape)

    # note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    # we would rather have it the other way around for LSTMs

    x = tf.keras.layers.Permute((2, 1, 3))(X_input)

    x = tf.keras.layers.Conv2D(10, (5, 1), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(1, (5, 1), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # x = Reshape((125, 80)) (x)
    # keras.backend.squeeze(x, axis)
    x = tf.keras.layers.Lambda(lambda q: tf.keras.backend.squeeze(q, -1), name='squeeze_last_dim')(x)

    x = tf.keras.layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(64, return_sequences=True))(
        x)  # [b_s, seq_len, vec_dim]
    x = tf.keras.layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(64))(x)


    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)

    output = tf.keras.layers.Dense(classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=[X_input], outputs=[output], name='RNNSpeechModelOriginal')

    return model


# not working properly, poor results
def cnn_trad_fpool3(input_shape, classes):

    X_input = tf.keras.Input(input_shape)

    X = tf.keras.layers.Conv2D(16, (20, 8), strides=(1, 1))(X_input)
    X = BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    #X = tf.keras.layers.SpatialDropout2D(0.2)(X)
    X = MaxPooling2D((2, 6), name='max_pool')(X)

    X = tf.keras.layers.Conv2D(32, (10, 4), strides=(1, 1))(X)
    X = BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    #X = tf.keras.layers.SpatialDropout2D(0.2)(X)
    X = MaxPooling2D((2, 2), name='max_pool1')(X)


    X = tf.keras.layers.Flatten()(X)

    X = tf.keras.layers.Dense(128, activity_regularizer = tf.keras.regularizers.l2(1e-5), activation='relu', name='fc2')(X)

    X = tf.keras.layers.Dense(classes, activity_regularizer = tf.keras.regularizers.l2(1e-5), activation='softmax', name='fc3')(X)

    # Create the keras model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = tf.keras.Model(inputs=X_input, outputs=X, name='cnn_trad_fpool3')

    return model

##################################################################
# RESIDUAL CNN
# res8
# res8_narrow
# res15
# res15_narrow
# res26
# res26_narrow
##################################################################

# res8
def Res8SpeechModel(input_shape):

    X_input = tf.keras.Input(input_shape)

    # Convolutional layer (3, 3, 19)
    x = tf.keras.layers.Conv2D(45, (3, 3), use_bias=False, activation='relu', padding='valid')(X_input)
    x = tf.keras.layers.BatchNormalization()(x)

    # Average pooling layer
    x = tf.keras.layers.AveragePooling2D(pool_size=(3, 4), strides=None, padding='valid')(x)

    # Res layer x 3
    x = convolutional_block(x, filters=[45, 45], stage=0, num=0, stride=1)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[45, 45], stage=0, num=1, stride=2)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[45, 45], stage=0, num=2, stride=1)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)

    #x = identity_block(x, filters=[19, 19], stage=0)
    #x = identity_block(x, filters=[19, 19], stage=1)
    #x = identity_block(x, filters=[19, 19], stage=2)

    # Average pooling layer
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid')(x)

    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(12, activation='softmax')(x)

    model = tf.keras.Model(inputs=[X_input], outputs=[output], name='Res8')

    return model

# res8 narrow
def Res8SpeechModel_narrow(input_shape):

    X_input = tf.keras.Input(input_shape)

    # Convolutional layer (3, 3, 19)
    x = tf.keras.layers.Conv2D(19, (3, 3), use_bias=False, activation='relu', padding='valid')(X_input)
    x = tf.keras.layers.BatchNormalization()(x)

    # Average pooling layer
    x = tf.keras.layers.AveragePooling2D(pool_size=(3, 4), strides=None, padding='valid')(x)

    # Res layer x 3
    x = convolutional_block(x, filters=[19, 19], stage=0, num=0, stride=1)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[19, 19], stage=0, num=1, stride=2)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[19, 19], stage=0, num=2, stride=1)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)

    #x = identity_block(x, filters=[19, 19], stage=0)
    #x = identity_block(x, filters=[19, 19], stage=1)
    #x = identity_block(x, filters=[19, 19], stage=2)

    # Average pooling layer
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid')(x)

    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(12, activation='softmax')(x)

    model = tf.keras.Model(inputs=[X_input], outputs=[output], name='Res8_narrow')

    return model

# res15
def Res15SpeechModel(input_shape):

    X_input = tf.keras.Input(input_shape)

    # Convolutional layer (3,3,45)
    x = tf.keras.layers.Conv2D(45, (3, 3), use_bias=False, activation='relu', padding='valid')(X_input)
    x = tf.keras.layers.BatchNormalization()(x)

    # Res layer x 6
    x = convolutional_block(x, filters=[45, 45], stage=0, num=0, stride=2)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[45, 45], stage=1, num=1, stride=2)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[45, 45], stage=2, num=2, stride=2)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[45, 45], stage=3, num=3, stride=1)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[45, 45], stage=4, num=4, stride=1)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[45, 45], stage=5, num=5, stride=1)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)

    # Convolutional layer (3,3,45) + dilation (16,16)
    x = tf.keras.layers.Conv2D(45, (3, 3), use_bias=False, activation='relu', dilation_rate=16, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Average pooling layer
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid')(x)

    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(12, activation='softmax')(x)

    model = tf.keras.Model(inputs=[X_input], outputs=[output], name='Res15')

    return model

# res15 narrow
def Res15SpeechModel_narrow(input_shape):

    X_input = tf.keras.Input(input_shape)

    # Convolutional layer (3,3,45)
    x = tf.keras.layers.Conv2D(19, (3, 3), use_bias=False, activation='relu', padding='valid')(X_input)
    x = tf.keras.layers.BatchNormalization()(x)

    # Res layer x 6
    x = convolutional_block(x, filters=[19, 19], stage=0, num=0, stride=2)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[19, 19], stage=1, num=1, stride=2)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[19, 19], stage=2, num=2, stride=2)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[19, 19], stage=3, num=3, stride=1)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[19, 19], stage=4, num=4, stride=1)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[19, 19], stage=5, num=5, stride=1)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)

    # Convolutional layer (3,3,45) + dilation (16,16)
    x = tf.keras.layers.Conv2D(19, (3, 3), use_bias=False, activation='relu', dilation_rate=16, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Average pooling layer
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid')(x)

    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(12, activation='softmax')(x)

    model = tf.keras.Model(inputs=[X_input], outputs=[output], name='Res15_narrow')

    return model

# res26
def Res26SpeechModel(input_shape):

    X_input = tf.keras.Input(input_shape)

    # Convolutional layer (3, 3, 19)
    x = tf.keras.layers.Conv2D(45, (3, 3), use_bias=False, activation='relu', padding='valid')(X_input)
    x = tf.keras.layers.BatchNormalization()(x)

    # Average pooling layer
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid')(x)

    # Res layer x 12
    x = convolutional_block(x, filters=[45, 45], stage=0, num=0, stride=2)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[45, 45], stage=0, num=1, stride=2)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[45, 45], stage=0, num=2, stride=2)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[45, 45], stage=0, num=3, stride=1)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[45, 45], stage=0, num=4, stride=1)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[45, 45], stage=0, num=5, stride=1)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[45, 45], stage=0, num=6, stride=1)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[45, 45], stage=0, num=7, stride=1)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[45, 45], stage=0, num=8, stride=1)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[45, 45], stage=0, num=9, stride=1)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[45, 45], stage=0, num=10, stride=1)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[45, 45], stage=0, num=11, stride=1)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)

    # Average pooling layer
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid')(x)

    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(12, activation='softmax')(x)

    model = tf.keras.Model(inputs=[X_input], outputs=[output], name='Res26')

    return model

# res26 narrow
def Res26SpeechModel_narrow(input_shape):

    X_input = tf.keras.Input(input_shape)

    # Convolutional layer (3, 3, 19)
    x = tf.keras.layers.Conv2D(19, (3, 3), use_bias=False, activation='relu', padding='valid')(X_input)
    x = tf.keras.layers.BatchNormalization()(x)

    # Average pooling layer
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid')(x)

    # Res layer x 12
    x = convolutional_block(x, filters=[19, 19], stage=0, num=0, stride=2)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[19, 19], stage=0, num=1, stride=2)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[19, 19], stage=0, num=2, stride=2)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[19, 19], stage=0, num=3, stride=1)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[19, 19], stage=0, num=4, stride=1)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[19, 19], stage=0, num=5, stride=1)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[19, 19], stage=0, num=6, stride=1)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[19, 19], stage=0, num=7, stride=1)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[19, 19], stage=0, num=8, stride=1)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[19, 19], stage=0, num=9, stride=1)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[19, 19], stage=0, num=10, stride=1)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[19, 19], stage=0, num=11, stride=1)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)

    # Average pooling layer
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid')(x)

    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(12, activation='softmax')(x)

    model = tf.keras.Model(inputs=[X_input], outputs=[output], name='Res26_narrow')

    return model

def identity_block(X, filters, stage):
    """
    Implementation of the identity block as defined in Figure 3

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) +'_branch'
    bn_name_base = 'bn' + str(stage) + '_branch'

    # Retrieve Filters
    F1, F2 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(F1, (3, 3), strides=(1,1), padding='same', use_bias=False, name=conv_name_base + '2a')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F2, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, name=conv_name_base + '2b')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def convolutional_block(X, filters, stage, num, stride):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(num) + '_branch'
    bn_name_base = 'bn' + str(num) + '_branch'

    # Retrieve Filters
    F1, F2 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(F1, (3, 3), strides=stride, padding='same', use_bias=False, name=conv_name_base + '2a', dilation_rate=int(2**(stage // 3)))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F2, kernel_size=(3, 3), strides=(1, 1), use_bias=False, padding='same', dilation_rate=int(2**(stage // 3)), name=conv_name_base + '2c')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(filters=F2, kernel_size=(3, 3), strides=stride, use_bias=False, padding='same', dilation_rate=int(2**(stage // 3)), name=conv_name_base + '1')(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

######################################################################################

def import_datasets_reference(masking_fraction_train = 0.0, task_selected="12_cl"):
    """
    Read the file containing the lists for the train, validation and test set.
    :param masking_fraction_train: fraction of training samples that
            are left out training for faster training
    :return: the three panda dataframes for train, validation and test.
    """
    # read the list for each set and select only wanted classes
    train_reference = pd.read_csv('train_dataset.txt', index_col=0, header=None, names=['label'])


    if masking_fraction_train > 0:
        mask_train = np.random.rand(train_reference.size) >= masking_fraction_train
        print(mask_train)
        print("train_reference size before mask ", train_reference.size)
        train_reference = train_reference[mask_train]
        print("train_reference size after mask ", train_reference.size)
    else:
        print("train_reference size: ", train_reference.size)

    validation_reference = pd.read_csv('validation_dataset.txt', index_col=0, header=None, names=['label'])

    print("Validation_reference size: ", validation_reference.size)
    if task_selected == "12_cl":
        test_reference = pd.read_csv('real_test_dataset.txt', index_col=0, header=None, names=['label'])
        print("test_reference size: ", test_reference.size)
    elif task_selected == "35_cl":
        test_reference = pd.read_csv('test_dataset.txt', index_col=0, header=None, names=['label'])
        print("test_reference size: ", test_reference.size)

    return train_reference, validation_reference, test_reference


def generate_classes_dictionaries(dataset_path, task_selected):
    """
    Compute 2 dictionaries that associate the relative
    number of the class with the class name and vice versa
    :param dataset_path: path in which the dataset is stored
    :return: the 2 dictionaries
    """
    list_subfolders_with_paths = [f.path for f in os.scandir(dataset_path) if f.is_dir()]
    list_subfolders_with_paths.sort()
    # compute 2 dictionaries for swiching between label in string or int
    if task_selected == "12_cl":
        classes_list = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

        classToNum = {}
        numToClass = {}
        num = 0
        unknown_class = 11
        for i in list_subfolders_with_paths:
            cl = i.split("/")[-1]
            if cl in classes_list:  # considering one of the classes that we want to classify or a "silence" sample
                classToNum[cl] = num
                numToClass[num] = cl
                num += 1
            else:
                classToNum[cl] = unknown_class

        classToNum["silence"] = 10
        classToNum["unknown"] = 11

        numToClass[10] = "silence"
        numToClass[11] = "unknown"

    elif task_selected == "35_cl":
        classToNum = {}
        numToClass = {}
        num = 0
        for i in list_subfolders_with_paths:
            cl = i.split("/")[-1]
            if cl != '_background_noise_':  # considering one of the classes that we want to classify or a "silence" sample
                classToNum[cl] = num
                numToClass[num] = cl
                num += 1

    print(classToNum)
    print(numToClass)
    return numToClass, classToNum



def compute_dataset_statistics(dataset_reference, classes):
    """
    compute percentage and number of samples for each class inside a panda dataframe
    :param dataset_reference: panda dataframe
    :return: number of samples and percentage for each class
    """
    samples = np.zeros(classes)
    for index, row in dataset_reference.iterrows():
        samples[row['label']] += 1

    percentage = []
    for i in samples:
        percentage.append(i / np.sum(samples) * 100)
    return samples, percentage


def remove_excessive_samples(dataset_reference, unknown_used_fraction):
    """
    hard masks, before the creation of the cache file, the unwanted "unknown" samples
    :param dataset_reference: panda dataframe containing the samples
    :param unknown_used_fraction: fraction of unwanted "unknown" samples
    :return: the masked dataframe
    """
    mask = dataset_reference['label'].apply(lambda l: (l != 11 or np.random.rand() <= 1 / unknown_used_fraction))
    true = 0
    for i in mask:
        if i == True:
            true += 1

    print("Kept values = ", true, "discarded values", mask.size - true)
    return dataset_reference[mask]


def plot_confusion_matrix(predictions, test_dataset, accuracy, classes, save=None):
    """
    Compute and plot the confusion matrix
    :param predictions: prediction computed thanks to the model
    :param test_dataset: panda dataframe that contains the real labels
    :param accuracy: accuracy found in the dataset
    :param save: filename if we want to save
    :return:
    """
    predictions = np.argmax(predictions, 1)
    real_labels = []
    for element in test_dataset.as_numpy_iterator():
        for i in element[1]:
            real_labels.append(i)

    #cm = tf.math.confusion_matrix(real_labels, predictions)
    cm = confusion_matrix(real_labels, predictions, normalize='true')

    cl = []
    for i in range(classes):
        cl.append(numToClass[i].capitalize())

    cm = pd.DataFrame(cm, index=[i for i in cl],
                         columns=[i for i in cl])

    plt.figure(figsize=(16, 10))
    sn.heatmap(cm, annot=True)
    sn.set(font_scale=0.8)
    plt.title('Confusion matrix in the test set, overall accuracy = ' + str(accuracy))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save is not None:
        plt.savefig(save)
    plt.show()

def plot_loss(train_loss, val_loss):
    """
    plot training and validation loss during training
    :param train_loss:
    :param val_loss:
    """
    # Plot losses
    plt.close('all')
    plt.figure(figsize=(6, 4))
    plt.semilogy(train_loss, label='Train loss')
    plt.semilogy(val_loss, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_accuracy(train_accuracy, val_accuracy):
    """
    plot accuracy for training and validation sets during training
    :param train_accuracy:
    :param val_accuracy:
    :return:
    """
    # Plot Accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(train_accuracy, label='Train Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

def lr_scheduler_linear(epoch, lr):
    decay_rate = 0.2
    decay_step = 2
    if epoch % decay_step == 0 and epoch:
        print(lr*decay_rate)
        return lr * decay_rate
    print(lr)
    return lr


def lr_scheduler_exp(epoch, lr):
    initial_lr= 0.01
    k = 0.1
    lr = initial_lr * np.exp(-k*epoch)
    return lr


def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.4
    epochs_drop = 15.0
    lrate = initial_lrate * math.pow(drop,
                                     math.floor((1 + epoch) / epochs_drop))

    if (lrate < 4e-5):
        lrate = 4e-5

    print('Changing learning rate to {}'.format(lrate))
    return lrate


def plot_roc(predictions, test_dataset):


    real_labels = np.empty((0,12))

    for element in test_dataset.as_numpy_iterator():
        for i in element[1]:
            one_hot = np.zeros(12)
            one_hot[i] = 1
            real_labels = np.vstack((real_labels,one_hot))


    print(real_labels.shape)
    print("real_labels = ",real_labels)
    print("predictions = ",predictions)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(12):
        fpr[i], tpr[i], _ = roc_curve(real_labels[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(real_labels.ravel(), predictions.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], 1-tpr["micro"], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.xlim([0.0, 0.09])
    plt.ylim([0.0, 0.40])
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="upper right")
    plt.show()

if __name__ == '__main__':
    np.random.seed(0)
    tf.random.set_seed(0)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True

    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    ######Run parameters

    dataset_path = './speech_commands_v0.02/'

    refresh_dataset_lists = False

    # this flag enables the use for the entire dataset,
    # it creates a cached file with all the available samples
    # if set to False it use only a subset of "unknown" samples
    # that matches the number of samples in the other sets
    use_all_training_set = True

    # this flag enables the use of the filtering function at each training
    # iteration, this means that the number of samples for each class is almost the same
    # and the unknown samples varies at each epoch
    partition_training_set = True

    train = True
    
    batch_size = 32
    unknown_used_fraction = 18
    num_epochs = 40
    masking_fraction_train = 0.5

    task_selected = "12_cl" # "35_cl"#

    if task_selected == "12_cl":
        classes = 12
    elif task_selected == "35_cl":
        classes = 35

    #generate_silence_samples(dataset_path)

    if refresh_dataset_lists:
        generate_train_val_test_list(dataset_path, name_train='train_dataset.txt', name_val='validation_dataset.txt', name_test='test_dataset.txt')


    train_reference, validation_reference, test_reference = import_datasets_reference(masking_fraction_train=masking_fraction_train, task_selected=task_selected)

    print(train_reference.size)
    if task_selected == "35_cl":
        mask = train_reference["label"] != "silence"
        train_reference = train_reference[mask]
        mask = validation_reference["label"] != "silence"
        validation_reference = validation_reference[mask]
        mask = test_reference["label"] != "silence"
        test_reference = test_reference[mask]
    print(train_reference.size)

    numToClass, classToNum = generate_classes_dictionaries(dataset_path, task_selected)


    # change label from string to int
    train_reference['label'] = train_reference['label'].apply(lambda l: classToNum[l])
    validation_reference['label'] = validation_reference['label'].apply(lambda l: classToNum[l])
    test_reference['label'] = test_reference['label'].apply(lambda l: classToNum[l])


    if use_all_training_set or task_selected == "35_cl":
        samples_train, percentage_train = compute_dataset_statistics(train_reference, classes)
        print("Train samples percentage for each class: ", percentage_train)
        samples_val, percentage_val = compute_dataset_statistics(validation_reference, classes)
        print("Validation samples percentage for each class: ", percentage_val)

        if partition_training_set and task_selected == "12_cl":
            average_sample_count_train = int(np.sum(samples_train) - samples_train[-1] + samples_train[-1]/unknown_used_fraction)
            print("Average_sample_count_train considering uniform distribution among classes", average_sample_count_train)
            percentage_train = samples_train * 100 / average_sample_count_train
            percentage_train[-1] = 100*(samples_train[-1] / unknown_used_fraction)/average_sample_count_train
            print("Train samples average percentage after filtering: ", percentage_train)


            average_sample_count_val = int(np.sum(samples_val) - samples_val[-1] + samples_val[-1] / unknown_used_fraction)
            print("Average_sample_count_val considering uniform distribution among classes", average_sample_count_val)
            percentage = samples_val*100/average_sample_count_val
            percentage[-1] = 100*(samples_val[-1] / unknown_used_fraction)/average_sample_count_val
            print("Validation samples average percentage after filtering: ", percentage)

    else:
        train_reference = remove_excessive_samples(train_reference, unknown_used_fraction)
        samples = compute_dataset_statistics(train_reference, classes)
        print("train masked samples statistic: ", samples)

        validation_reference = remove_excessive_samples(validation_reference, unknown_used_fraction)
        samples = compute_dataset_statistics(validation_reference, classes)
        print("validation masked samples statistic: ", samples)

    samples = compute_dataset_statistics(test_reference, classes)
    print("Test samples statistic: ", samples)


    # create the tensorflow dataset for train, validation and test
    if task_selected == "12_cl":
        if use_all_training_set:
            if partition_training_set:
                train_dataset = create_dataset(train_reference, batch_size, shuffle=True, filter=True, repeat=True,
                                               cache_file=('train_cache' + str(masking_fraction_train)+task_selected))
                validation_dataset = create_dataset(validation_reference, batch_size, shuffle=True, filter=True,
                                                     repeat=False, cache_file='validation_cache'+task_selected)
            else:
                # the same cached datsets can be used, since the filter is only applied during training, after caching
                train_dataset = create_dataset(train_reference, batch_size, shuffle=True, filter=False, repeat=True,
                                               cache_file='train_cache' + str(masking_fraction_train)+task_selected)
                validation_dataset = create_dataset(validation_reference, batch_size, shuffle=True,
                                                    filter=False, repeat=False, cache_file='validation_cache'+task_selected)
        else:
            train_dataset = create_dataset(train_reference, batch_size, shuffle=True, filter=False, repeat=True,
                                           cache_file='train_cache_masked' + str(masking_fraction_train)+task_selected)
            validation_dataset = create_dataset(validation_reference, batch_size, shuffle=True,
                                                filter=False, repeat=True, cache_file='validation_cache_masked'+task_selected)

    elif task_selected == "35_cl":
        train_dataset = create_dataset(train_reference, batch_size, shuffle=True, filter=False, repeat=True,
                                       cache_file='train_cache' + str(masking_fraction_train) + task_selected)
        validation_dataset = create_dataset(validation_reference, batch_size, shuffle=True,
                                            filter=False, repeat=False, cache_file='validation_cache' + task_selected)

    test_dataset = create_dataset(test_reference, batch_size, shuffle=False, filter=False, repeat=False, cache_file='test_cache'+task_selected)

    # check how the samples in the dataset have been processed
    for element in train_dataset.as_numpy_iterator():
        plt.figure(figsize=(17, 6))
        plt.pcolormesh(element[0][0, :, :, 0])
        plt.title('Spectrogram visualization - librosa. Sample class = ' + numToClass[element[1][0]])
        plt.ylabel('Frequency')
        plt.xlabel('Time')
        plt.colorbar()
        plt.show()
        break

    # Call the function to create the model and compile it
    # model = modelconvNN((n_mels, int(1/(frame_step)-1), 1))
    # model = modelconvNN1((80, 126, 1), classes)
    # model = ConvSpeechModel((80, 126, 1), classes)
    # model = RNNSpeechModelOriginal((80, 126, 1), classes)
    # model = AttRNNSpeechModel((80, 126, 1), classes)
    # model = AttRNNSpeechModel1((80, 126, 1), classes)
    # model = AttRNNSpeechModellite((80, 126, 1), classes)
    # model = AttRNNSpeechModelLightest((80, 126, 1), classes)
    # model = AttRNNSpeechModelaccuracybest((80, 126, 1), classes)
    # model = AttRNNSpeechModellitebest((80, 126, 1), classes)
    # model = AttRNNSpeechModel((80, 126, 1), classes)
    # model = cnn_trad_fpool3((80, 126, 1), classes)

    # Residual CNN models
    # model = Res15SpeechModel((80, 126, 1))
    # model = Res15SpeechModel_narrow((80, 126, 1))
    model = Res8SpeechModel((80, 126, 1))
    # model = Res8SpeechModel_narrow((80, 126, 1))
    # model = Res26SpeechModel((80, 126, 1))
    # model = Res26SpeechModel_narrow((80, 126, 1))

    # model = modelconvNN1((80, 126, 1))

    model.summary()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    test_steps = int(np.ceil(len(test_reference) / batch_size))
    #model = tf.keras.models.load_model('ModelEntireDatasetPartitioned_' + model.name + '_' + str(masking_fraction_train) + '.h5')
    # train model and plot accuracy and loss behavior for training and validation sets at each epoch
    if train:
        my_callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, monitor="val_loss", restore_best_weights=True),
            #tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
            #tf.keras.callbacks.TensorBoard(log_dir='./logs'),
            tf.keras.callbacks.LearningRateScheduler(step_decay, verbose=0)
        ]

        if task_selected == "12_cl":
            if use_all_training_set:
                if partition_training_set:
                    steps_per_epoch_train = int(average_sample_count_train / batch_size)
                    steps_per_epoch_val = int(average_sample_count_val / batch_size)

                    print("steps_per_epoch_train ", steps_per_epoch_train)
                    print("steps_per_epoch_val ", steps_per_epoch_val)

                    history = model.fit(train_dataset, verbose=1, epochs=num_epochs, steps_per_epoch=steps_per_epoch_train,
                                        validation_data=validation_dataset, validation_steps=steps_per_epoch_val,callbacks=my_callbacks)

                    plot_loss(history.history['loss'], history.history['val_loss'])
                    plot_accuracy(history.history['accuracy'], history.history['val_accuracy'])

                    model.save('ModelEntireDatasetPartitioned_' +
                                      model.name + '_'+str(masking_fraction_train) + task_selected + '.h5')
                else:
                    train_steps = int(np.ceil(len(train_reference) / batch_size))
                    val_steps = int(np.ceil(len(validation_reference) / batch_size))

                    print("steps_per_epoch_train ", train_steps)
                    print("steps_per_epoch_val ", val_steps)

                    history = model.fit(train_dataset, verbose=1, epochs=num_epochs, steps_per_epoch=train_steps,
                                        validation_data=validation_dataset, validation_steps=val_steps, callbacks=my_callbacks)

                    plot_loss(history.history['loss'], history.history['val_loss'])
                    plot_accuracy(history.history['accuracy'], history.history['val_accuracy'])

                    model.save('ModelEntireDataset_' +
                                      model.name + '_'+str(masking_fraction_train)+ task_selected + '.h5')

            else:
                train_steps = int(np.ceil(len(train_reference) / batch_size))
                val_steps = int(np.ceil(len(validation_reference) / batch_size))

                print("train_steps ", train_steps)
                print("validation_steps ", val_steps)

                # Fit the model
                history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps, validation_data=validation_dataset, validation_steps=val_steps)

                plot_loss(history.history['loss'], history.history['val_loss'])
                plot_accuracy(history.history['accuracy'], history.history['val_accuracy'])

                model.save('ModelPartialDataset_' +
                                      model.name + '_'+str(masking_fraction_train) + task_selected + '.h5')

        elif task_selected == "35_cl":
            train_steps = int(np.ceil(len(train_reference) / batch_size))
            val_steps = int(np.ceil(len(validation_reference) / batch_size))

            print("steps_per_epoch_train ", train_steps)
            print("steps_per_epoch_val ", val_steps)

            history = model.fit(train_dataset, verbose=1, epochs=num_epochs, steps_per_epoch=train_steps,
                                validation_data=validation_dataset, validation_steps=val_steps, callbacks=my_callbacks)

            plot_loss(history.history['loss'], history.history['val_loss'])
            plot_accuracy(history.history['accuracy'], history.history['val_accuracy'])

            model.save('ModelEntireDataset_' +
                       model.name + '_' + str(masking_fraction_train) + task_selected + '.h5')

    if task_selected == "12_cl":
        # analyze results by plotting confusion matrix
        if use_all_training_set:
            if partition_training_set:
                model = tf.keras.models.load_model('ModelEntireDatasetPartitioned_' +
                                      model.name + '_'+str(masking_fraction_train)+ task_selected + '.h5')
                accuracy = model.evaluate(test_dataset, steps=test_steps)
                print("Accuracy in the test_set model trained with all the unknown samples, partitioned= ", accuracy)
                predictions = model.predict(test_dataset, steps=test_steps)
                plot_confusion_matrix(predictions, test_dataset, accuracy, classes, 'confusion_matrix_balanced_training_set_' +
                                      model.name + '_' + str(masking_fraction_train)+ task_selected + '.png')

                plot_roc(predictions, test_dataset)
            else:
                model = tf.keras.models.load_model('ModelEntireDataset_' +
                                      model.name + '_'+str(masking_fraction_train)+ task_selected + '.h5')
                accuracy = model.evaluate(test_dataset, steps=test_steps)
                print("Accuracy in the test_set model trained with all the unknown samples= ", accuracy)
                predictions = model.predict(test_dataset, steps=test_steps)
                plot_confusion_matrix(predictions, test_dataset, accuracy, classes, 'confusion_matrix_entire_training_set_' +
                                      model.name + '_'+str(masking_fraction_train)+ task_selected + '.png')
        else:
            model = tf.keras.models.load_model('ModelPartialDataset_' +
                                      model.name + '_'+str(masking_fraction_train)+ task_selected + '.h5')
            accuracy = model.evaluate(test_dataset, steps=test_steps)
            print("Accuracy in the test_set using only subset of unknown samples= ", accuracy)
            predictions = model.predict(test_dataset, steps=test_steps)
            plot_confusion_matrix(predictions, test_dataset, accuracy, classes, 'confusion_matrix_truncated_training_set_' +
                                      model.name + '_'+str(masking_fraction_train)+ task_selected + '.png')
    elif task_selected == "35_cl":
        model = tf.keras.models.load_model('ModelEntireDataset_' +
                                           model.name + '_' + str(masking_fraction_train) + task_selected + '.h5')
        accuracy = model.evaluate(test_dataset, steps=test_steps)
        print("Accuracy in the test_set model trained with all the unknown samples= ", accuracy)
        predictions = model.predict(test_dataset, steps=test_steps)
        plot_confusion_matrix(predictions, test_dataset, accuracy, classes,'confusion_matrix_entire_training_set_' +
                              model.name + '_' + str(masking_fraction_train) + task_selected + '.png')

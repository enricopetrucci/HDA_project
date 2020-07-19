import numpy as np
import librosa
import librosa.display
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
from sklearn.metrics import confusion_matrix, roc_curve, auc
import math

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


    y, sr = librosa.load(file_path, 16000)
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


######################################################################################

def import_datasets_reference(train, validation, test, masking_fraction_train = 0.0, task_selected="12_cl"):
    """
    Read the file containing the lists for the train, validation and test set.
    :param masking_fraction_train: fraction of training samples that
            are left out training for faster training
    :return: the three panda dataframes for train, validation and test.
    """
    # read the list for each set and select only wanted classes
    train_reference = pd.read_csv(train, index_col=0, header=None, names=['label'])


    if masking_fraction_train > 0:
        mask_train = np.random.rand(train_reference.size) >= masking_fraction_train
        print(mask_train)
        print("train_reference size before mask ", train_reference.size)
        train_reference = train_reference[mask_train]
        print("train_reference size after mask ", train_reference.size)
    else:
        print("train_reference size: ", train_reference.size)

    validation_reference = pd.read_csv(validation, index_col=0, header=None, names=['label'])

    print("Validation_reference size: ", validation_reference.size)
    if task_selected == "12_cl":
        test_reference = pd.read_csv(test, index_col=0, header=None, names=['label'])
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


def plot_confusion_matrix(predictions, test_dataset, accuracy, classes, numToClass, save=None):
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

    if classes == 12:
        plt.figure(figsize=(8, 5))
    else:
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
    print("real_labels = ", real_labels)
    print("predictions = ", predictions)
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


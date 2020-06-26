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

from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D



def generate_train_val_test_list(dataset_path, perc_val, perc_test, name_train, name_val, name_test):
    """generates 3 file, one for each of train, validation and test set.
    each of these files contains the path of all the samples that are contained in each dataset
    Args:
    dataset_path: path of the dataset.
    perc_val: How much of the data set to use for validation.
    perc_test: How much of the data set to use for testing.
    name_train: name of the file that will be generated that contains the list for the train set
    name_val: name of the file that will be generated that contains the list for the validation set
    name_test: name of the file that will be generated that contains the list for the test set
    """
    train_list = []
    val_list = []
    test_list = []

    list_subfolders_with_paths = [f.path for f in os.scandir(dataset_path) if f.is_dir()]

    for path in list_subfolders_with_paths:
        if path.split("/")[-1] != '_background_noise_':
            for filename in glob.glob(os.path.join(path, '*.wav')):
                with open(filename, 'r') as f:
                    chosen_set = which_set(filename, perc_val, perc_test)
                    if chosen_set == 'training':
                        train_list.append([filename, path.split('/')[-1]])
                    elif chosen_set == 'validation':
                        val_list.append([filename, path.split('/')[-1]])
                    elif chosen_set == 'testing':
                        test_list.append([filename, path.split('/')[-1]])

    with open(name_train, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(train_list)

    with open(name_val, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(val_list)

    with open(name_test, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(test_list)
    print("samples for training", len(train_list), "samples for validation", len(val_list), "samples for testing", len(test_list))


def generate_train_val_test_list(dataset_path, name_train, name_val, name_test):
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




def which_set(filename, validation_percentage, testing_percentage):
    """Determines which data partition the file should belong to.
    We want to keep files in the same training, validation, or testing sets even
    if new ones are added over time. This makes it less likely that testing
    samples will accidentally be reused in training when long runs are restarted
    for example. To keep this stability, a hash of the filename is taken and used
    to determine which set it should belong to. This determination only depends on
    the name and the set proportions, so it won't change as other files are added.
    It's also useful to associate particular files as related (for example words
    spoken by the same person), so anything after '_nohash_' in a filename is
    ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
    'bobby_nohash_1.wav' are always in the same set, for example.
    Args:
    :param filename: File path of the data sample.
    :param validation_percentage: How much of the data set to use for validation.
    :param testing_percentage: How much of the data set to use for testing.
    Returns:
    String, one of 'training', 'validation', or 'testing'.
    """
    MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1  # ~134M

    base_name = os.path.basename(filename)
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way o# grouping wavs that are close variations of each other.
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    hash_name = hash_name.encode()
    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    hash_name_hashed = hashlib.sha1(hash_name).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1)) * (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'testing'
    else:
        result = 'training'
    return result


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
    #print(sr)
    #print(y.shape)
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

def filter_fn(x, y):
    print(x)
    print(y)
    return ((not tf.math.equal(y, 10)) or np.random.rand() < 1 / 25)


def create_dataset(reference, batch_size, shuffle, window_duration, frame_step, n_mels, repeat, cache_file=None):
    """
    Create dataset and store a cached version
    :param reference: list of the path for each sample in the dataset and their class
    :param batch_size:
    :param shuffle:
    :param window_duration: how long the window for the mfcc feature will be
    :param frame_step: how long each step between 2 windows will be for the mfcc feature
    :param n_mels: number of MFCCs to compute
    :param cache_file: name of the cache file
    :return: the dataset
    """
    sample_rate = 16000
    n_fft = int(window_duration * sample_rate)
    hop_length = int(frame_step * sample_rate)

    file_paths = list(reference.index)
    labels = reference['label']

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    #py_func = lambda file_path, label: (tf.numpy_function(load_and_preprocess_data_librosa, [file_path, n_fft, hop_length, n_mels], tf.float32), label)
    py_func = lambda file_path, label: (tf.numpy_function(load_and_preprocess_data_librosa_mel_spectrogram, [file_path, n_fft, hop_length, n_mels], tf.float32), label)
    #py_func = lambda file_path, label: (tf.numpy_function(load_and_preprocess_data_python_speech_features, [file_path, n_fft, hop_length, n_mels], tf.float32), label)

    dataset = dataset.map(py_func, num_parallel_calls=os.cpu_count())

    # Cache dataset
    if cache_file:
        dataset = dataset.cache(cache_file)

    # Shuffle
    if shuffle:
        dataset = dataset.shuffle(len(file_paths))

    #Repeat the dataset indefinitely
    if repeat:
        dataset = dataset.repeat()

    # Batch
    dataset = dataset.batch(batch_size=batch_size)

    # Prefetch
    dataset = dataset.prefetch(buffer_size=1)

    return dataset

def clear_cache():
    return 1


def modelconvNN(input_shape):
    """
    :param input_shape: -- shape of the data of the dataset
    :return: model -- a tf.keras.Model() instance
    """

    X_input = tf.keras.Input(input_shape)

    X = tf.keras.layers.Conv2D(32, (8, 16), strides=(1, 4))(X_input)
    X = BatchNormalization(axis=3,)(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = MaxPooling2D((2, 2), name='max_pool')(X)

    X = tf.keras.layers.Conv2D(64, (4, 5), strides=(1, 1))(X)
    X = BatchNormalization(axis=3, )(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = MaxPooling2D((2, 2), name='max_pool1')(X)

    X = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1))(X)
    X = BatchNormalization(axis=3, )(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = MaxPooling2D((2, 2), name='max_pool2')(X)

    X = tf.keras.layers.Flatten()(X)

    X = tf.keras.layers.Dense(12, activation='softmax', name='fc2')(X)

    # Create the keras model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = tf.keras.Model(inputs=X_input, outputs=X, name='MyModel')

    return model


if __name__ == '__main__':

    np.random.seed(0)
    dataset_path = './speech_commands_v0.02/'

    #generate_silence_samples(dataset_path)

    refresh_dataset_lists = False

    if refresh_dataset_lists:
        generate_train_val_test_list(dataset_path, name_train='train_dataset.txt', name_val='validation_dataset.txt', name_test='test_dataset.txt')

    # fraction of the total that will be left out from the training
    masking_fraction = 0.90

    # read the list for each set and select only wanted classes
    train_reference = pd.read_csv('train_dataset.txt', index_col=0, header=None, names=['label'])

    mask_train = np.random.rand(train_reference.size) >= masking_fraction
    print(mask_train)

    print("train_reference size before mask ", train_reference.size)
    train_reference = train_reference[mask_train]
    print("train_reference size after mask ", train_reference.size)

    validation_reference = pd.read_csv('validation_dataset.txt', index_col=0, header=None, names=['label'])

    test_reference = pd.read_csv('real_test_dataset.txt', index_col=0, header=None, names=['label'])


    list_subfolders_with_paths = [f.path for f in os.scandir(dataset_path) if f.is_dir()]
    
    # compute 2 dictionaries for swiching between label in string or int

    classes_list = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

    classToNum = {}
    numToClass = {}
    num = 0
    unknown_class = 11
    for i in list_subfolders_with_paths:
        cl = i.split("/")[-1]
        if cl in classes_list:      # considering one of the classes that we want to classify or a "silence" sample
            classToNum[cl] = num
            numToClass[num] = cl
            num += 1
        else:
            classToNum[cl] = unknown_class

    classToNum["silence"] = 10
    classToNum["unknown"] = 11

    numToClass[10] = "silence"
    numToClass[11] = "unknown"

    print(classToNum)
    print(numToClass)

    # change label from string to int
    train_reference['label'] = train_reference['label'].apply(lambda l: classToNum[l])
    #print(train_reference.iloc[0])

    validation_reference['label'] = validation_reference['label'].apply(lambda l: classToNum[l])

    test_reference['label'] = test_reference['label'].apply(lambda l: classToNum[l])

    samples = np.zeros(12)
    for index, row in train_reference.iterrows():
        samples[row['label']] += 1
        # print(element)
        # print(element[0].shape)
    print("train samples statistic: ", samples)

    samples = np.zeros(12)
    for index, row in validation_reference.iterrows():
        samples[row['label']] += 1
        # print(element)
        # print(element[0].shape)
    print("validation samples statistic: ", samples)

    samples = np.zeros(12)
    for index, row in test_reference.iterrows():
        samples[row['label']] += 1
        # print(element)
        # print(element[0].shape)
    print("test samples statistic: ", samples)



    mask = train_reference['label'].apply(lambda l: (l != 11 or np.random.rand() <= 1 / 19))

    true = 0
    for i in mask:
        if i == True:
            true += 1

    print("kept values = ", true, "discarded values", mask.size - true)
    train_masked_reference = train_reference[mask]

    samples = np.zeros(12)
    for index, row in train_masked_reference.iterrows():
        samples[row['label']] += 1
        # print(element)
        # print(element[0].shape)

    print("train masked samples statistic: ", samples)
    """
    dataset = tf.data.Dataset.from_tensor_slices(([1, 2, 3, 2, 3, 4, 5], [10, 5, 6, 10, 10, 10, 10]))
    dataset = dataset.cache("try1")
    dataset = dataset.filter(lambda x, y: y != 10 or np.random.rand() <= 1 / 2)
    print(list(dataset.as_numpy_iterator()))

    file_paths = list(train_reference.index)
    labels = train_reference['label']

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.cache("try2")
    #mask = np.random.rand(len(file_paths)) <= 1/25

    samples = np.zeros(12)
    for i in dataset.as_numpy_iterator():
        samples[i[1]] += 1
        # print(element)
        # print(element[0].shape)
    print(samples)

    dataset = dataset.map(lambda x, y:  (x, y) if(y != 10 or np.random.rand() <= 1 / 2) else (x, np.int64(-1)))

    dataset = dataset.filter(lambda x, y: y != -1)

    samples = np.zeros(12)
    for i in dataset.as_numpy_iterator():
        samples[i[1]] += 1
        # print(element)
        # print(element[0].shape)

    print(samples)

    """

    
    # initialize preprocessing variables
    n_mels = 40
    window_duration = 0.025
    frame_step = 0.010
    
    sample_rate = 16000
    n_fft = int(window_duration * sample_rate)
    hop_length = int(frame_step * sample_rate)

    batch_size = 32
    # create the tensorflow dataset for train, validation and test
    train_masked_dataset = create_dataset(train_masked_reference, batch_size, shuffle=True, window_duration=window_duration, frame_step=frame_step, n_mels=n_mels, repeat=True, cache_file='train_masked_cache')
    train_dataset = create_dataset(train_reference, batch_size, shuffle=True, window_duration=window_duration, frame_step=frame_step, n_mels=n_mels, repeat=True, cache_file='train_cache')
    validation_dataset = create_dataset(validation_reference, batch_size, shuffle=True, window_duration=window_duration, frame_step=frame_step, n_mels=n_mels, repeat=True, cache_file='validation_cache')
    test_dataset = create_dataset(test_reference, batch_size, shuffle=False, window_duration=window_duration, frame_step=frame_step, n_mels=n_mels, repeat=False, cache_file='test_cache')
    
    # check how the samples in the dataset have been processed
    samples = np.zeros(12)
    for element in train_dataset.as_numpy_iterator():
        for i in element[1]:
            samples[i]+=1
        # print(element)
        #print(element[0].shape)

        print(samples)

        #print(element[1][0])
        plt.figure(figsize=(17, 6))
        plt.pcolormesh(element[0][0,:,:,0])

        plt.title('Spectrogram visualization - librosa')
        plt.ylabel('Frequency')
        plt.xlabel('Time')
        plt.show()
        break
        

    # Call the function to create the model and compile it
    # model = modelconvNN((n_mels, int(1/(frame_step)-1), 1))
    model = modelconvNN((80, 126, 1))
    model.summary()

    train_masked_steps = int(np.ceil(len(train_masked_reference) / batch_size))
    train_steps = int(np.ceil(len(train_reference) / batch_size))
    val_steps = int(np.ceil(len(validation_reference) / batch_size))
    test_steps = int(np.ceil(len(test_reference) / batch_size))

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    num_epochs = 30
    
    # Fit the model
    history = model.fit(train_masked_dataset, epochs=num_epochs, steps_per_epoch=train_steps, validation_data=validation_dataset, validation_steps=val_steps)
    model.save('my_model.h5')

    history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps, validation_data=validation_dataset, validation_steps=val_steps)
    model.save('my_model1.h5')


    model = tf.keras.models.load_model('my_model.h5')

    accuracy = model.evaluate(test_dataset, steps=test_steps)
    print("accurcy in the test_set = ", accuracy)

    predictions = model.predict(test_dataset, steps=test_steps)

    predictions = np.argmax(predictions, 1)
    print(predictions.shape)
    print(predictions[1])

    real_labels = []
    for element in test_dataset.as_numpy_iterator():
        for i in element[1]:
            real_labels.append(i)

    cm = tf.math.confusion_matrix(real_labels, predictions)

    cl = []
    for i in range(12):
        cl.append(numToClass[i])

    #cm = pd.DataFrame(cm, index = [i for i in "ABCDEFGHIJKL"], columns = [i for i in "ABCDEFGHIJKL"])#index=[i for i in cl], columns=[i for i in cl])
    plt.figure(figsize=(8, 5))
    sn.heatmap(cm, annot=True)
    sn.set(font_scale=0.8)
    plt.title('Confusion matrix in the test set')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix_imbalanced_training_set.png')
    plt.show()

    model = tf.keras.models.load_model('my_model1.h5')

    accuracy = model.evaluate(test_dataset, steps=test_steps)
    print("accurcy in the test_set = ", accuracy)

    predictions = model.predict(test_dataset, steps=test_steps)

    predictions = np.argmax(predictions, 1)
    print(predictions.shape)
    print(predictions[1])

    real_labels = []
    for element in test_dataset.as_numpy_iterator():
        for i in element[1]:
            real_labels.append(i)

    cm = tf.math.confusion_matrix(real_labels, predictions)

    cl = []
    for i in range(12):
        cl.append(numToClass[i])

    #cm = pd.DataFrame(cm, index = [i for i in "ABCDEFGHIJKL"], columns = [i for i in "ABCDEFGHIJKL"])#index=[i for i in cl], columns=[i for i in cl])
    plt.figure(figsize=(8, 5))
    sn.heatmap(cm, annot=True)
    sn.set(font_scale=0.8)
    plt.title('Confusion matrix in the test set')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix_balanced_training_set.png')
    plt.show()
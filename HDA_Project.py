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

def load_and_preprocess_data(file_path, n_fft, hop_length, n_mels):
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
    return mel_features.astype(np.float32)


def create_dataset(reference, batch_size, shuffle, window_duration, frame_step, n_mels, cache_file=None):
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
    sample_rate = 22050
    n_fft = int(window_duration * sample_rate)
    hop_length = int(frame_step * sample_rate)

    file_paths = list(reference.index)
    labels = reference['label']

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    py_func = lambda file_path, label: (tf.numpy_function(load_and_preprocess_data, [file_path, n_fft, hop_length, n_mels],
                                                          tf.float32), label)
    dataset = dataset.map(py_func, num_parallel_calls=os.cpu_count())

    # Cache dataset
    if cache_file:
        dataset = dataset.cache(cache_file)

    # Shuffle
    if shuffle:
        dataset = dataset.shuffle(len(file_paths))

    # Repeat the dataset indefinitely
    #dataset = dataset.repeat()

    # Batch
    dataset = dataset.batch(batch_size=batch_size)

    # Prefetch
    dataset = dataset.prefetch(buffer_size=1)

    return dataset


def modelconvNN(input_shape):
    """
    :param input_shape: -- shape of the data of the dataset
    :return: model -- a tf.keras.Model() instance
    """

    X_input = tf.keras.Input(input_shape)

    X = tf.keras.layers.Conv2D(100, (8, 16), strides=(1, 4))(X_input)
    X = BatchNormalization(axis=3,)(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = MaxPooling2D((1, 3), name='max_pool')(X)

    X = tf.keras.layers.Conv2D(78, (4, 5), strides=(1,1))(X)
    X = BatchNormalization(axis=3, )(X)
    X = tf.keras.layers.Activation('relu')(X)

    #X = MaxPooling2D((1, 3), name='max_pool1')(X)

    X = tf.keras.layers.Flatten()(X)

    X = tf.keras.layers.Dense(2, activation='softmax', name='fc2')(X)

    # Create the keras model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = tf.keras.Model(inputs=X_input, outputs=X, name='MyModel')

    return model


if __name__ == '__main__':

    hashlib.sha256(str(random.getrandbits(256)).encode('utf-8')).hexdigest()
    'cd183a211ed2434eac4f31b317c573c50e6c24e3a28b82ddcb0bf8bedf387a9f'
    refresh_dataset_lists = False

    dataset_path = './speech_commands_v0.02/'

    if(refresh_dataset_lists):
        generate_train_val_test_list(dataset_path, perc_val=10, perc_test=10, name_train='train_dataset.csv', name_val='validation_dataset.csv', name_test='test_dataset.csv')


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

    classToNum = {}
    numToClass = {}
    num = 0

    for i in list_subfolders_with_paths:
        cl = i.split("/")[-1]
        classToNum[cl] = num
        numToClass[num] = cl
        num += 1


    train_reference['label'] = train_reference['label'].apply(lambda l: classToNum[l])
    validation_reference['label'] = validation_reference['label'].apply(lambda l: classToNum[l])
    test_reference['label'] = test_reference['label'].apply(lambda l: classToNum[l])

    # initialize preprocessing variables
    n_mels = 40
    window_duration = 0.025
    frame_step = 0.010

    batch_size = 32
    # create the tensorflow dataset for train, validation and test
    train_dataset = create_dataset(train_reference, batch_size, shuffle=True, window_duration=window_duration, frame_step=frame_step, n_mels=n_mels, cache_file='train_cache')
    validation_dataset = create_dataset(validation_reference, batch_size, shuffle=True, window_duration=window_duration, frame_step=frame_step, n_mels=n_mels, cache_file='validation_cache')
    test_dataset = create_dataset(test_reference, batch_size, shuffle=True, window_duration=window_duration, frame_step=frame_step, n_mels=n_mels, cache_file='test_cache')

    # check how the samples in the dataset have been processed
    for element in train_dataset.as_numpy_iterator():
        #print(element)
        print(element[0].shape)
        print(element[0])
        break

    # Call the function to create the model and compile it
    model = modelconvNN((n_mels, int(1/(frame_step)+1), 1))
    model.summary()

    train_steps = int(np.ceil(len(train_reference) / batch_size))
    val_steps = int(np.ceil(len(validation_reference) / batch_size))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    num_epochs = 30

    # Fit the model
    history = model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=train_steps, validation_data=validation_dataset, validation_steps=val_steps)
    model.save('my_model.h5')

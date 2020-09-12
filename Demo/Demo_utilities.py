import pyaudio
import wave
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import random
import sys
sys.path.insert(1, '../')
from Utilities import *
from IPython.display import Audio, display, Image

def record_sample():
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1
    fs = 16000
    seconds = 1
    filename = "output.wav"

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    print('Recording in 3,')
    time.sleep(1)
    print('Recording in 2,')
    time.sleep(1)
    print('Recording in 1,')
    time.sleep(1)
    print('Recording...')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for 3 seconds
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print('Finished recording')

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()


def record_noise_sample():
    print("Recording noise sample for 1 second...")
    CHUNKSIZE = 8000  # fixed chunk size
    RATE = 16000
    sr = 16000
    first = True
    sample = 0
    # initialize portaudio
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)

    data = stream.read(2 * CHUNKSIZE)
    noise_sample = np.frombuffer(data, dtype=np.float32)


    print("Done!")
    loud_threshold = np.mean(np.abs(noise_sample))
    #loud_threshold=avg_max_loudness(noise_sample)*3
    print("Loud threshold", loud_threshold)
    return loud_threshold


def continuous_detection(model, loud_threshold, dict, seconds):
    with tf.device("/cpu:0"):

        CHUNKSIZE = 8000  # fixed chunk size
        RATE = 16000
        sr = 16000
        first = True
        sample = 0
        # initialize portaudio
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)

        now = time.time()
        future = now + seconds
        print("Started recording...")

        while (time.time() < future):
            data = stream.read(CHUNKSIZE)
            data = np.frombuffer(data, dtype=np.float32)

            if (first):
                previous_chunk = data
                first = False
            else:
                sample += 1
                current_sample = np.concatenate((previous_chunk, data))
                loudness = np.mean(np.abs(current_sample))#avg_max_loudness(current_sample)
                #print("Loudness: ", loudness)
                print("sample ", sample," loudness = ", loudness, end='')

                if loudness < loud_threshold:
                    print(": nothing detected")
                #                  print("sample ", sample, ' = ', end='')
                #                  print("silence noise")

                else:
                    # librosa.display.waveplot(current_sample, sr=RATE)
                    # plt.close('all')
                    y = normalize_data(current_sample)
                    librosa_melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024,
                                                                     hop_length=128, power=1.0,  # window='hann',
                                                                     n_mels=80, fmin=40.0, fmax=sr / 2)

                    S_dB = librosa.power_to_db(librosa_melspec, ref=np.max)

                    S_dB = S_dB.reshape((S_dB.shape[0], S_dB.shape[1], 1))
                    result = normalize_data(S_dB)
                    spectrogram = np.resize(result, (1, 80, 126, 1))

                    prediction = model.predict(spectrogram)
                    #                   print(prediction)
                    print(". Prediction  = ", end='')
                    print(dict[np.argmax(prediction)], ' with probability ', np.max(prediction))
                    #                     if(dict[np.argmax(prediction)]!= "silence"):
                    #                         break
                    #                     if np.max(prediction)>0.90:
                    #                         print(dict[np.argmax(prediction)])
                    #                     else:
                    #                         print("silence")
                    previous_chunk = data

        # close stream
        stream.stop_stream()
        stream.close()
        p.terminate()
    print("end")


def load_everything():
    classes = 12


    dataset_path = '../speech_commands_v0.02/'
    task_selected = "12_cl"
    masking_fraction_train = 0.0
    if task_selected == "12_cl":
        classes = 12
    elif task_selected == "35_cl":
        classes = 35

    model = tf.keras.models.load_model('Att25K.h5')



    train_reference, validation_reference, test_reference = import_datasets_reference('../train_dataset.txt',
                                                                                      '../validation_dataset.txt',
                                                                                      '../real_test_dataset.txt',
                                                                                      masking_fraction_train=masking_fraction_train,
                                                                                      task_selected=task_selected)

    numToClass, classToNum = generate_classes_dictionaries(dataset_path, task_selected)

    # change label from string to int
    test_reference['label'] = test_reference['label'].apply(lambda l: classToNum[l])

    batch_size = 32
    test_steps = int(np.ceil(len(test_reference) / batch_size))

    test_dataset = create_dataset(test_reference, batch_size, shuffle=False, filter=False, repeat=False,
                                  cache_file='../test_cache' + task_selected)

    real_labels = []
    for element in test_dataset.as_numpy_iterator():
        for i in element[1]:
            real_labels.append(i)

    return model, test_reference, test_dataset, real_labels, test_steps, numToClass

def plot_spectrograms(y, sr, predicted_label, true_label, numToClass):
    y = normalize_data(y)
    librosa_melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024,
                                                     hop_length=128, power=1.0,  # window='hann',
                                                     n_mels=80, fmin=40.0, fmax=sr / 2)

    S_dB = librosa.power_to_db(librosa_melspec, ref=np.max)
    result = normalize_data(S_dB)

    plt.figure(figsize=(17, 6))
    plt.pcolormesh(result)
    plt.title('Sample spectrogram. Sample class = ' + numToClass[true_label])
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.colorbar()
    plt.show()

    plt.figure(figsize=(17, 6))
    plt.subplot(1, 2, 1)
    predicted_average = np.loadtxt('spect_'+str(predicted_label)+'.csv', delimiter=',')
    plt.pcolormesh(predicted_average)
    plt.title('Average spectrogram of the predicted class = ' + numToClass[predicted_label])
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    true_average = np.loadtxt('spect_' + str(true_label) + '.csv', delimiter=',')
    plt.pcolormesh(true_average)
    plt.title('Average spectrogram of the true class = ' + numToClass[true_label])
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.colorbar()
    plt.show()

def avg_max_loudness(current_sample):
    current_sample = abs(current_sample)
    sum_max_loudness = 0
    current_sample.sort(reverse=True)
    for i in range(1000):
        sum_max_loudness+=i
    return sum_max_loudness/100




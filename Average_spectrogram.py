from Models import *
from Utilities import *

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

dataset_path = './speech_commands_v0.02/'

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


train_reference, validation_reference, test_reference = import_datasets_reference('train_dataset.txt', 'validation_dataset.txt', 'real_test_dataset.txt',  masking_fraction_train=masking_fraction_train, task_selected=task_selected)

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

sum_spectrogram = [np.zeros((80,126)),np.zeros((80,126)),np.zeros((80,126)),np.zeros((80,126)),np.zeros((80,126)),np.zeros((80,126)),np.zeros((80,126)),np.zeros((80,126)),np.zeros((80,126)),np.zeros((80,126)),np.zeros((80,126)),np.zeros((80,126)),]
counters = [0,0,0,0,0,0,0,0,0,0,0,0]
number_of_batches=0
for element in train_dataset.as_numpy_iterator():
    for i in range(32):
        #print(element[1][i])
        sum_spectrogram[element[1][i]]+=(element[0][i, :, :, 0])
        counters[element[1][i]]+=1
    number_of_batches+=1
    print(number_of_batches)
    if(number_of_batches==100):
        break

average_spectrogram=[]
for i in range(len(sum_spectrogram)):
    average_spectrogram.append(sum_spectrogram[i]/counters[i])
    np.savetxt('Demo/spect_'+str(i)+'.csv', average_spectrogram[i], delimiter=',')
    plt.figure(figsize=(17, 6))
    plt.pcolormesh(average_spectrogram[i])
    plt.title('Average spectrogram visualization - librosa. Sample class = ' + numToClass[i])
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.colorbar()
    plt.show()


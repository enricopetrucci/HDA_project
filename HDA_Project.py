from Models import *
from Utilities import *

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
    masking_fraction_train = 0.0

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
    #model = AttRNNSpeechModelOriginal((80, 126, 1), classes)
    model = AttRNNSpeechModel25K((80, 126, 1), classes)
    #model = AttRNNSpeechModel40K((80, 126, 1), classes)
    #model = AttRNNSpeechModel50K((80, 126, 1), classes)
    #model = AttRNNSpeechModel87K((80, 126, 1), classes)
    #model = AttRNNSpeechModel155K((80, 126, 1), classes)
    #model = Res8narrowAndATTSpeechModel((80, 126, 1))
    # model = cnn_trad_fpool3((80, 126, 1), classes)
    # model = CNN_TRAD_POOL2((80, 126, 1), classes)

    # Residual CNN models
    # model = Res15SpeechModel((80, 126, 1))
    # model = Res15SpeechModel_narrow((80, 126, 1))
    # model = Res8SpeechModel((80, 126, 1))
    # model = Res8SpeechModel_narrow((80, 126, 1))
    # model = Res26SpeechModel((80, 126, 1))
    # model = Res26SpeechModel_narrow((80, 126, 1))


    ######### input_size, model name, number layers, number maps, pooling, dilation, size pooling, number classes #########
    #model = ResSpeechModelOriginal((80, 126, 1), 'res8', 6, 45, True, False, (3, 4), classes=classes)  # Res8
    #model = ResSpeechModelOriginal((80, 126, 1), 'res15', 13, 45, False, True, (1, 1), classes=classes)  # Res15
    #model = ResSpeechModelOriginal((80, 126, 1), 'res26', 24, 45, True, False, (2,2), classes=classes) #Res26
    #model = ResSpeechModelOriginal((80, 126, 1), 'res8_narrow', 6, 19, True, False, (3, 4), classes=classes)  # Res8_narrow
    #model = ResSpeechModelOriginal((80, 126, 1), 'res15_narrow', 13, 19, False, True, (1, 1), classes=classes)  # Res15_narrow
    #model = ResSpeechModelOriginal((80, 126, 1), 'res26_narrow', 24, 19, True, False, (2, 2), classes=classes)  # Res15_narrow


    model.summary()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    test_steps = int(np.ceil(len(test_reference) / batch_size))
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
    else:
        print("loading model")
        model = tf.keras.models.load_model('ModelEntireDatasetPartitioned_' + model.name + '_' + str(masking_fraction_train)  + task_selected + '.h5')

    if task_selected == "12_cl":
        # analyze results by plotting confusion matrix
        if use_all_training_set:
            if partition_training_set:
                model = tf.keras.models.load_model('ModelEntireDatasetPartitioned_' +
                                      model.name + '_'+str(masking_fraction_train)+ task_selected + '.h5')
                accuracy = model.evaluate(test_dataset, steps=test_steps)
                print("Accuracy in the test_set model trained with all the unknown samples, partitioned= ", accuracy)
                predictions = model.predict(test_dataset, steps=test_steps)
                plot_confusion_matrix(predictions, test_dataset, accuracy, classes, numToClass, 'confusion_matrix_balanced_training_set_' +
                                      model.name + '_' + str(masking_fraction_train)+ task_selected + '.pdf')

                plot_roc(predictions, test_dataset)
            else:
                model = tf.keras.models.load_model('ModelEntireDataset_' +
                                      model.name + '_'+str(masking_fraction_train)+ task_selected + '.h5')
                accuracy = model.evaluate(test_dataset, steps=test_steps)
                print("Accuracy in the test_set model trained with all the unknown samples= ", accuracy)
                predictions = model.predict(test_dataset, steps=test_steps)
                plot_confusion_matrix(predictions, test_dataset, accuracy, classes, numToClass, 'confusion_matrix_entire_training_set_' +
                                      model.name + '_'+str(masking_fraction_train)+ task_selected + '.pdf')
        else:
            model = tf.keras.models.load_model('ModelPartialDataset_' +
                                      model.name + '_'+str(masking_fraction_train)+ task_selected + '.h5')
            accuracy = model.evaluate(test_dataset, steps=test_steps)
            print("Accuracy in the test_set using only subset of unknown samples= ", accuracy)
            predictions = model.predict(test_dataset, steps=test_steps)
            plot_confusion_matrix(predictions, test_dataset, accuracy, classes, 'confusion_matrix_truncated_training_set_' +
                                      model.name + '_'+str(masking_fraction_train)+ task_selected + '.pdf')
    elif task_selected == "35_cl":
        model = tf.keras.models.load_model('ModelEntireDataset_' +
                                           model.name + '_' + str(masking_fraction_train) + task_selected + '.h5')
        accuracy = model.evaluate(test_dataset, steps=test_steps)
        print("Accuracy in the test_set model trained with all the unknown samples= ", accuracy)
        predictions = model.predict(test_dataset, steps=test_steps)
        plot_confusion_matrix(predictions, test_dataset, accuracy, classes,'confusion_matrix_entire_training_set_' +
                              model.name + '_' + str(masking_fraction_train) + task_selected + '.pdf')

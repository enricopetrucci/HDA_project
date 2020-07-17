import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from HDA_Project import *

from sklearn.metrics import confusion_matrix, roc_curve, auc
from tensorflow.keras.layers import Dropout , SpatialDropout2D, Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Add
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D

def compute_roc(predictions, test_dataset):

    real_labels = np.empty((0, 12))
    for element in test_dataset.as_numpy_iterator():
        for i in element[1]:
            one_hot = np.zeros(12)
            one_hot[i] = 1
            real_labels = np.vstack((real_labels, one_hot))


    print(real_labels.shape)
    print("real_labels = ", real_labels)
    print("predictions = ", predictions)
    fpr = dict()
    tpr = dict()
    # roc_auc = dict()
    # sum_fpr = 0
    # sum_tpr = 0
    # for i in range(12):
    #     fpr[i], tpr[i], _ = roc_curve(real_labels[:, i], predictions[:, i])
    #     sum_fpr += fpr[i]
    #     sum_tpr += tpr[i]
    #     roc_auc[i] = auc(fpr[i], tpr[i])


    # average_fpr = sum_fpr/12
    # average_tpr = sum_tpr/12

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(real_labels.ravel(), predictions.ravel())
    return (fpr["micro"], tpr["micro"])

def plot_confusion_matrix_final(predictions, test_dataset, accuracy, classes, model, save=None, ):
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

    # cm = cm
    # strcm = []
    # for i in cm:
    #     s = list(map(str, i))
    #     sperc=[]
    #     for k in s:
    #         k = k+"%"
    #         sperc.append(k)
    #     strcm.append(s)
    cl = []
    for i in range(classes):
        cl.append(numToClass[i].capitalize())

    nrows, ncols = cm.shape
    cm = cm * 100
    annot = np.empty_like(cm).astype(np.str)
    for i in range(nrows):
        for j in range(ncols):
            p = cm[i, j]
            annot[i, j] = np.str('%.1f%%' % p)
    if classes == 12:
        plt.figure(figsize=(7,5))
    else:
        plt.figure(figsize=(16, 10))

    print(annot)
    cm = pd.DataFrame(cm, index=[i for i in cl],
                      columns=[i for i in cl])

    accuracy = accuracy * 100
    sn.heatmap(cm, annot=annot, fmt='')
    #plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.20)
    plt.tick_params(labelsize=7)
    plt.title('Confusion matrix for '+model+'. Accuracy = %.2f%% ' % accuracy , pad=10, fontsize=10)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    sn.set(font_scale=0.6)


    if save is not None:
        plt.savefig(save)
    plt.show()


with tf.device("/cpu:0"):
    dataset_path = './speech_commands_v0.02/'
    task_selected="12_cl"
    masking_fraction_train = 0.0

    if task_selected == "12_cl":
        classes = 12
    elif task_selected == "35_cl":
        classes = 35

    train_reference, validation_reference, test_reference = import_datasets_reference(
        task_selected=task_selected)

    numToClass, classToNum = generate_classes_dictionaries(dataset_path, task_selected)

    # change label from string to int
    test_reference['label'] = test_reference['label'].apply(lambda l: classToNum[l])

    batch_size = 32
    test_steps = int(np.ceil(len(test_reference) / batch_size))

    test_dataset = create_dataset(test_reference, batch_size, shuffle=False, filter=False, repeat=False, cache_file='test_cache'+task_selected)


    models = []
    roc_curves = []

    att_models = [ "Att50K", "Att25K","Att87K", "Att155K"]
    for i in att_models:
        model = tf.keras.models.load_model("Definitive_models/" + i + '.h5')
        model.summary()
        models.append(model)
        accuracy = model.evaluate(test_dataset, steps=test_steps)
        print("Accuracy on the test_set model trained with all the unknown samples, partitioned= ", accuracy)
        predictions = model.predict(test_dataset, steps=test_steps)
        roc_curves.append(compute_roc(predictions, test_dataset))
        plot_confusion_matrix_final(predictions, test_dataset, accuracy[1], classes, i, "ConfusionMatrix_"+i+"flat.pdf")

    res_models = ["Res8_lite", "Res8_narrow", "Res15_narrow", "Res26_narrow"]
    for i in res_models:
        model = tf.keras.models.load_model("Definitive_models/" + i + '.h5')
        model.summary()
        models.append(model)
        accuracy = model.evaluate(test_dataset, steps=test_steps)
        print("Accuracy in the test_set model trained with all the unknown samples, partitioned= ", accuracy)
        predictions = model.predict(test_dataset, steps=test_steps)
        print(predictions.shape)
        roc_curves.append(compute_roc(predictions, test_dataset))
        plot_confusion_matrix_final(predictions, test_dataset, accuracy[1], classes, i,  "ConfusionMatrix_" + i + "flat.pdf")

    plt.close('all')
    plt.figure()
    lw = 2
    for i in range(4):
        plt.plot(roc_curves[i][0], 1-roc_curves[i][1],
             lw=lw, label='ROC curve ' + models[i].name)

    for i in range(4):
        plt.plot(roc_curves[4+i][0], 1 - roc_curves[4+i][1],
                 ':', lw=lw, label='ROC curve ' + res_models[i])

    plt.xlim([0, 0.1])
    plt.ylim([0., 0.1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.title('ROC', pad=10, fontsize=15)
    plt.legend(loc="upper right")
    plt.grid()
    # plt.savefig("ROC.pdf")
    plt.show()

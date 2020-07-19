import tensorflow as tf

from tensorflow.keras.layers import Dropout , SpatialDropout2D, Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Add
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D


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
def AttRNNSpeechModelOriginal(input_shape, classes, rnn_func=tf.keras.layers.LSTM):

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

    model = tf.keras.Model(inputs=[X_input], outputs=[output], name='AttRNNSpeechModelOriginal')

    return model


#25K parameters lightest but well regularized
# 12_cl whole dataset 0.9588956832885742
# 12_cl 0.952965259552002 0.9552147388458252
# 35_cl 0.9362108111381531
def AttRNNSpeechModel25K(input_shape, classes, rnn_func=tf.keras.layers.GRU):

    X_input = tf.keras.Input(input_shape)

    # note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    # we would rather have it the other way around for LSTMs

    x = tf.keras.layers.Permute((2, 1, 3))(X_input)

    x = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = tf.keras.layers.SpatialDropout2D(0.05)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # x = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Lambda(lambda q: tf.keras.backend.squeeze(q, -1), name='squeeze_last_dim')(x)
    x = tf.keras.layers.Bidirectional(rnn_func(32, return_sequences=True, dropout=0.1))(x)  # [b_s, seq_len, vec_dim]
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
    model = tf.keras.Model(inputs=[X_input], outputs=[output], name='Att25K')

    return model

#50k
#940490 0.9413087964057922 0.9519427418708801 0.9472392797470093
def AttRNNSpeechModel50K(input_shape, classes, rnn_func=tf.keras.layers.GRU):

    X_input = tf.keras.Input(input_shape)

    # note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    # we would rather have it the other way around for LSTMs

    x = tf.keras.layers.Permute((2, 1, 3))(X_input)

    x = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = tf.keras.layers.SpatialDropout2D(0.05)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = tf.keras.layers.SpatialDropout2D(0.05)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.SpatialDropout2D(0.01)(x)

    x = tf.keras.layers.Lambda(lambda q: tf.keras.backend.squeeze(q, -1), name='squeeze_last_dim')(x)
    x = tf.keras.layers.Bidirectional(rnn_func(32,  return_sequences=True, dropout=0.1))(x)  # [b_s, seq_len, vec_dim]
    x = tf.keras.layers.Bidirectional(rnn_func(32, return_sequences=True, dropout=0.1))(x)  # [b_s, seq_len, vec_dim]

    xFirst = tf.keras.layers.Lambda(lambda q: q[:, -1])(x)  # [b_s, vec_dim]
    query = tf.keras.layers.Dense(64)(xFirst)

    # dot product attention
    attScores = tf.keras.layers.Dot(axes=[1, 2])([query, x])
    attScores = tf.keras.layers.Softmax(name='attSoftmax')(attScores)  # [b_s, seq_len]

    # rescale sequence
    attVector = tf.keras.layers.Dot(axes=[1, 1])([attScores, x])  # [b_s, vec_dim]

    x = tf.keras.layers.Dense(64, activation='relu')(attVector)
    x = tf.keras.layers.Dropout(0.05)(x)
    x = tf.keras.layers.BatchNormalization()(x)


    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.05)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    output = tf.keras.layers.Dense(classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=[X_input], outputs=[output], name='Att50K')

    return model


# around 80K parameters a good compromise between light and accurate
# 12_cl 0.95030677318573 0.947852790355 0.951329231262207
def AttRNNSpeechModel87K(input_shape, classes, rnn_func=tf.keras.layers.GRU):
    X_input = tf.keras.Input(input_shape)

    # note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    # we would rather have it the other way around for LSTMs

    x = tf.keras.layers.Permute((2, 1, 3))(X_input)

    x = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = tf.keras.layers.SpatialDropout2D(0.05)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = tf.keras.layers.SpatialDropout2D(0.05)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.SpatialDropout2D(0.01)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Lambda(lambda q: tf.keras.backend.squeeze(q, -1), name='squeeze_last_dim')(x) # since we only have one filter the output is back to being a single image
    x = tf.keras.layers.Bidirectional(rnn_func(64,  return_sequences=True, dropout=0.1))(x)  # [b_s, seq_len, vec_dim]

    xFirst = tf.keras.layers.Lambda(lambda q: q[:, -1])(x)  # [b_s, vec_dim]
    query = tf.keras.layers.Dense(128)(xFirst)

    # dot product attention
    attScores = tf.keras.layers.Dot(axes=[1, 2])([query, x])
    attScores = tf.keras.layers.Softmax(name='attSoftmax')(attScores)  # [b_s, seq_len]

    # rescale sequence
    attVector = tf.keras.layers.Dot(axes=[1, 1])([attScores, x])  # [b_s, vec_dim]

    x = tf.keras.layers.Dense(128, activation='relu')(attVector)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    output = tf.keras.layers.Dense(classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=[X_input], outputs=[output], name='Att87K')

    return model

# 155K parameters best accuracy for 12_cl
#12_cl 0.9527607560157776, 0.956646203994751 0.9597136974334717
def AttRNNSpeechModel155K(input_shape, classes, rnn_func=tf.keras.layers.GRU):

    X_input = tf.keras.Input(input_shape)

    # note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    # we would rather have it the other way around for LSTMs

    x = tf.keras.layers.Permute((2, 1, 3))(X_input)

    x = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = tf.keras.layers.SpatialDropout2D(0.1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = tf.keras.layers.SpatialDropout2D(0.1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.SpatialDropout2D(0.01)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Lambda(lambda q: tf.keras.backend.squeeze(q, -1), name='squeeze_last_dim')(x)
    x = tf.keras.layers.Bidirectional(rnn_func(64, return_sequences=True, dropout=0.1))(x)  # [b_s, seq_len, vec_dim]
    x = tf.keras.layers.Bidirectional(rnn_func(64, return_sequences=True, dropout=0.1))(x)  # [b_s, seq_len, vec_dim]

    xFirst = tf.keras.layers.Lambda(lambda q: q[:, -1])(x)  # [b_s, vec_dim]
    query = tf.keras.layers.Dense(128)(xFirst)

    # dot product attention
    attScores = tf.keras.layers.Dot(axes=[1, 2])([query, x])
    attScores = tf.keras.layers.Softmax(name='attSoftmax')(attScores)  # [b_s, seq_len]

    # rescale sequence
    attVector = tf.keras.layers.Dot(axes=[1, 1])([attScores, x])  # [b_s, vec_dim]

    x = tf.keras.layers.Dense(128, activation='relu')(attVector)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.BatchNormalization()(x)


    x = tf.keras.layers.Dense(64, activation='relu')(attVector)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    output = tf.keras.layers.Dense(classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=[X_input], outputs=[output], name='Att155K')

    return model

def Res8narrowAndATTSpeechModel(input_shape, classes):

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

    # Average pooling layer
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid')(x)

    x = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Lambda(lambda q: tf.keras.backend.squeeze(q, -1), name='squeeze_last_dim')(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, return_sequences=True))(x)  # [b_s, seq_len, vec_dim]
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
    model = tf.keras.Model(inputs=[X_input], outputs=[output], name='Res8narrowAndATTSpeechModel')

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

def CNN_TRAD_POOL2(input_shape, classes):
    X_input = tf.keras.Input(input_shape)
    X = tf.keras.layers.Conv2D(64, (20, 8), strides=(1, 1), activation='relu', padding='same')(X_input)
    X = tf.keras.layers.Dropout(0.5)(X)
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    X = tf.keras.layers.Conv2D(64, (10, 4), strides=(1, 1), activation='relu', padding='same')(X)
    X = tf.keras.layers.Dropout(0.5)(X)
    X = MaxPooling2D((2, 2), name='max_pool1')(X)

    X = tf.keras.layers.Flatten()(X)

    X = tf.keras.layers.Dense(classes, activation='softmax', name='fc3')(X)

    # Create the keras model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = tf.keras.Model(inputs=X_input, outputs=X, name='CNN_TRAD_POOL2')
    return model


##################################################################
#                         RESIDUAL CNN
##################################################################

"""
 Res Speech Original Models
 @ input_shape
 @ name name of the model
 @ nlayers number of layers
 @ n_feature_maps number of kernels
 @ pool if there is a pooling layer
 @ dilation if there is dilation in Conv2D
 @ poolSize size of pooling layer
 @ classes number of classes
"""
def ResSpeechModelOriginal(input_shape, name, n_layers, n_feature_maps, pool, dilation, poolSize, classes):

    X_input = tf.keras.Input(input_shape)

    # Convolutional layer (3,3,45)
    x = tf.keras.layers.Conv2D(n_feature_maps, (3, 3), use_bias=False, activation='relu', padding='same')(X_input)

    if pool:
        x = tf.keras.layers.AveragePooling2D(pool_size=poolSize, strides=None, padding='valid')(x)

    if dilation:
        for i in range(n_layers + 1):
            if i == 0:
                old_x = x
                continue
            x = Conv2D(n_feature_maps, (3, 3), strides=1, padding='same', use_bias=False, name=str(i) + 'Conv2D', dilation_rate=int(2**(i // 3)))(x)
            y = Activation('relu')(x)
            if i > 0 and i % 2 == 0:
                x = Add()([y, old_x])
                old_x = x
            else:
                x = y
            if i > 0:
                x = BatchNormalization(axis=3, trainable=False, name=str(i) + 'BatchNormalization')(x)
    else:
        for i in range(n_layers + 1):
            if i == 0:
                old_x = x
                continue
            x = Conv2D(n_feature_maps, (3, 3), strides=1, padding='same', use_bias=False, name=str(i) + 'Conv2D', dilation_rate=1)(x)
            y = Activation('relu')(x)
            if i > 0 and i % 2 == 0:
                x = Add()([y, old_x])
                old_x = x
            else:
                x = y
            if i > 0:
                x = BatchNormalization(axis=3, trainable=False, name=str(i) + 'BatchNormalization')(x)

    x = tf.reduce_mean(x, [1,2])

    output = tf.keras.layers.Dense(classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=[X_input], outputs=[output], name='ResSpeechoriginal' + name)

    return model

"""
 Res8 Speech Model
 @ input_shape
 @ classes number of classes
"""
def Res8SpeechModel(input_shape, classes):

    X_input = tf.keras.Input(input_shape)

    # Convolutional layer (3, 3, 45)
    x = tf.keras.layers.Conv2D(45, (3, 3), use_bias=False, activation='relu', padding='valid')(X_input)
    x = tf.keras.layers.BatchNormalization()(x)

    # Average pooling layer
    x = tf.keras.layers.AveragePooling2D(pool_size=(3, 4), strides=None, padding='valid')(x)

    # Res block x 3
    x = convolutional_block(x, filters=[45, 45], stage=0, num=0, stride=1)
    #x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[45, 45], stage=0, num=1, stride=2)
    #x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[45, 45], stage=0, num=2, stride=1)
    #x = tf.keras.layers.SpatialDropout2D(0.2)(x)

    # Average pooling layer
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid')(x)

    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=[X_input], outputs=[output], name='Res8')

    return model

"""
 Res8narrow Speech Model with 23.4K parameters
 @ input_shape
 @ classes number of classes
"""
def Res8SpeechModel_narrow(input_shape, classes):

    X_input = tf.keras.Input(input_shape)

    # Convolutional layer (3, 3, 19)
    x = tf.keras.layers.Conv2D(19, (3, 3), use_bias=False, activation='relu', padding='valid')(X_input)
    x = tf.keras.layers.BatchNormalization()(x)

    # Average pooling layer
    x = tf.keras.layers.AveragePooling2D(pool_size=(3, 4), strides=None, padding='valid')(x)

    # Res block x 3
    # identity block -> convolution block -> identity block 
    x = convolutional_block(x, filters=[19, 19], stage=0, num=0, stride=1)
    x = convolutional_block(x, filters=[19, 19], stage=0, num=1, stride=2)
    x = convolutional_block(x, filters=[19, 19], stage=0, num=2, stride=1)

    # Reduce Mean layer
    x = tf.reduce_mean(x, [1,2])

    output = tf.keras.layers.Dense(classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=[X_input], outputs=[output], name='Res8_narrow')

    return model

"""
 Res8lite Speech Model with 57K parameters
 @ input_shape
 @ classes number of classes
"""
def Res8SpeechModel_lite(input_shape, classes):

    X_input = tf.keras.Input(input_shape)

    # Convolutional layer (3, 3, 19)
    x = tf.keras.layers.Conv2D(30, (3, 3), use_bias=False, activation='relu', padding='valid')(X_input)
    x = tf.keras.layers.BatchNormalization()(x)

    # Average pooling layer
    x = tf.keras.layers.AveragePooling2D(pool_size=(3, 4), strides=None, padding='valid')(x)

    # Res block x 3
    # identity block -> convolution block -> identity block 
    x = convolutional_block(x, filters=[30, 30], stage=0, num=0, stride=1)
    x = convolutional_block(x, filters=[30, 30], stage=0, num=1, stride=2)
    x = convolutional_block(x, filters=[30, 30], stage=0, num=2, stride=1)

    # Reduce Mean layer
    x = tf.reduce_mean(x, [1,2])

    output = tf.keras.layers.Dense(classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=[X_input], outputs=[output], name='Res8_lite')

    return model

"""
 Res15 Speech Model
 @ input_shape
 @ classes number of classes
"""
def Res15SpeechModel(input_shape, classes):

    X_input = tf.keras.Input(input_shape)

    # Convolutional layer (3,3,45)
    x = tf.keras.layers.Conv2D(45, (3, 3), use_bias=False, activation='relu', padding='valid')(X_input)
    x = tf.keras.layers.BatchNormalization()(x)

    # Res block x 6
    x = convolutional_block(x, filters=[45, 45], stage=0, num=0, stride=2)
    #x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[45, 45], stage=1, num=1, stride=2)
    #x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[45, 45], stage=2, num=2, stride=2)
    #x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[45, 45], stage=3, num=3, stride=1)
    #x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[45, 45], stage=4, num=4, stride=1)
    #x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[45, 45], stage=5, num=5, stride=1)
    #x = tf.keras.layers.SpatialDropout2D(0.2)(x)

    # Convolutional layer (3,3,45) + dilation (16,16)
    x = tf.keras.layers.Conv2D(45, (3, 3), use_bias=False, activation='relu', dilation_rate=16, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Average pooling layer
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid')(x)

    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=[X_input], outputs=[output], name='Res15')

    return model

"""
 Res15narrow Speech Model with 53K parameters
 @ input_shape
 @ classes number of classes
"""
def Res15SpeechModel_narrow(input_shape, classes):

    X_input = tf.keras.Input(input_shape)

    # Convolutional layer (3,3,45)
    x = tf.keras.layers.Conv2D(19, (3, 3), use_bias=False, activation='relu', padding='valid')(X_input)
    x = tf.keras.layers.BatchNormalization()(x)

    # Res block x 6
    # identity -> conv. -> identity -> conv. -> identity -> conv.
    x = convolutional_block(x, filters=[19, 19], stage=0, num=0, stride=1)
    x = convolutional_block(x, filters=[19, 19], stage=1, num=1, stride=2)
    x = convolutional_block(x, filters=[19, 19], stage=2, num=2, stride=1)
    x = convolutional_block(x, filters=[19, 19], stage=3, num=3, stride=2)
    x = convolutional_block(x, filters=[19, 19], stage=4, num=4, stride=1)
    x = convolutional_block(x, filters=[19, 19], stage=5, num=5, stride=2)

    # Convolutional layer (3,3,45)
    x = tf.keras.layers.Conv2D(19, (3, 3), use_bias=False, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Reduce Mean layer
    x = tf.reduce_mean(x, [1,2])

    output = tf.keras.layers.Dense(classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=[X_input], outputs=[output], name='Res15_narrow')

    return model

"""
 Res26 Speech Model
 @ input_shape
 @ classes number of classes
"""
def Res26SpeechModel(input_shape, classes):

    X_input = tf.keras.Input(input_shape)

    # Convolutional layer (3, 3, 19)
    x = tf.keras.layers.Conv2D(45, (3, 3), use_bias=False, activation='relu', padding='valid')(X_input)
    x = tf.keras.layers.BatchNormalization()(x)

    # Average pooling layer
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid')(x)

    # Res block x 12
    x = convolutional_block(x, filters=[45, 45], stage=0, num=0, stride=2)
    #x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[45, 45], stage=0, num=1, stride=2)
    #x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[45, 45], stage=0, num=2, stride=2)
    #x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[45, 45], stage=0, num=3, stride=1)
    #x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[45, 45], stage=0, num=4, stride=1)
    #x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[45, 45], stage=0, num=5, stride=1)
    #x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[45, 45], stage=0, num=6, stride=1)
    #x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[45, 45], stage=0, num=7, stride=1)
    #x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[45, 45], stage=0, num=8, stride=1)
    #x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[45, 45], stage=0, num=9, stride=1)
    #x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[45, 45], stage=0, num=10, stride=1)
    #x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = convolutional_block(x, filters=[45, 45], stage=0, num=11, stride=1)
    #x = tf.keras.layers.SpatialDropout2D(0.2)(x)

    # Average pooling layer
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid')(x)

    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=[X_input], outputs=[output], name='Res26')

    return model

"""
 Res26narrow Speech Model with 92.4K parameters
 @ input_shape
 @ classes number of classes
"""
def Res26SpeechModel_narrow(input_shape, classes):

    X_input = tf.keras.Input(input_shape)

    # Convolutional layer (3, 3, 19)
    x = tf.keras.layers.Conv2D(19, (3, 3), use_bias=False, activation='relu', padding='valid')(X_input)
    x = tf.keras.layers.BatchNormalization()(x)

    # Average pooling layer
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid')(x)

    # Res block x 12
    # indentity x 2 -> conv. -> indentity x 2 -> conv. -> indentity x 2 -> conv. -> indentity x 2 -> conv.
    x = convolutional_block(x, filters=[19, 19], stage=0, num=0, stride=1)
    x = convolutional_block(x, filters=[19, 19], stage=0, num=1, stride=1)
    x = convolutional_block(x, filters=[19, 19], stage=0, num=2, stride=2)
    x = convolutional_block(x, filters=[19, 19], stage=0, num=3, stride=1)
    x = convolutional_block(x, filters=[19, 19], stage=0, num=4, stride=1)
    x = convolutional_block(x, filters=[19, 19], stage=0, num=5, stride=2)
    x = convolutional_block(x, filters=[19, 19], stage=0, num=6, stride=1)
    x = convolutional_block(x, filters=[19, 19], stage=0, num=7, stride=1)
    x = convolutional_block(x, filters=[19, 19], stage=0, num=8, stride=2)
    x = convolutional_block(x, filters=[19, 19], stage=0, num=9, stride=1)
    x = convolutional_block(x, filters=[19, 19], stage=0, num=10, stride=1)
    x = convolutional_block(x, filters=[19, 19], stage=0, num=11, stride=2)

    # Reduce Mean layer
    x = tf.reduce_mean(x, [1,2])

    output = tf.keras.layers.Dense(classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=[X_input], outputs=[output], name='Res26_narrow')

    return model

def identity_block(X, filters, stage):
    """
    Implementation of the identity block

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network

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
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- convolution dilation size
    nun -- position block in the network
    stride -- stride size in convolution layer

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

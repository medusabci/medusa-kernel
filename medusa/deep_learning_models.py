# Built-in imports
import warnings
import os

# External imports
import sklearn.utils as sk_utils
import numpy as np

# Medusa imports
from medusa import components
from medusa import classification_utils
from medusa import tensorflow_integration

# Extras
if os.environ.get("MEDUSA_EXTRAS_GPU_TF") == "1":
    import tensorflow as tf
    import tensorflow.keras as keras
    from tensorflow.keras.layers import Activation, Input, Flatten
    from tensorflow.keras.layers import Dropout, BatchNormalization
    from tensorflow.keras.layers import Conv2D, AveragePooling2D
    from tensorflow.keras.layers import DepthwiseConv2D, Dense
    from tensorflow.keras.layers import SpatialDropout2D, SeparableConv2D
    from tensorflow.keras.layers import Conv3D, AveragePooling3D, Add
    from tensorflow.keras.constraints import max_norm
    from tensorflow.keras.callbacks import EarlyStopping
else:
    raise tensorflow_integration.TFExtrasNotInstalled()


class EEGInceptionv1(components.ProcessingMethod):
    """EEG-Inception as described in Santamaría-Vázquez et al. 2020 [1]. This
    model is specifically designed for EEG classification tasks.

    References
    ----------
    [1] Santamaría-Vázquez, E., Martínez-Cagigal, V., Vaquerizo-Villar, F., &
    Hornero, R. (2020). EEG-Inception: A Novel Deep Convolutional Neural Network
    for Assistive ERP-based Brain-Computer Interfaces. IEEE Transactions on
    Neural Systems and Rehabilitation Engineering.
    """
    def __init__(self, input_time=1000, fs=128, n_cha=8, filters_per_branch=8,
                 scales_time=(500, 250, 125), dropout_rate=0.25,
                 activation='elu', n_classes=2, learning_rate=0.001,
                 gpu_acceleration=None):
        # Super call
        super().__init__(fit=[], predict_proba=['y_pred'])

        # Tensorflow config
        if gpu_acceleration is None:
            tensorflow_integration.check_tf_config(autoconfig=True)
        else:
            tensorflow_integration.config_tensorflow(gpu_acceleration)
        if tensorflow_integration.check_gpu_acceleration():
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        tf.keras.backend.set_image_data_format('channels_last')

        # Parameters
        self.input_time = input_time
        self.fs = fs
        self.n_cha = n_cha
        self.filters_per_branch = filters_per_branch
        self.scales_time = scales_time
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.n_classes = n_classes
        self.learning_rate = learning_rate

        # Useful variables
        self.input_samples = int(input_time * fs / 1000)
        self.scales_samples = [int(s * fs / 1000) for s in scales_time]

        # Create model
        self.model = self.__keras_model()
        # Create training callbacks
        self.training_callbacks = list()
        self.training_callbacks.append(
            EarlyStopping(monitor='val_loss',
                          min_delta=0.001,
                          mode='min',
                          patience=20,
                          verbose=1,
                          restore_best_weights=True)
        )
        # Create fine-tuning callbacks
        self.fine_tuning_callbacks = list()
        self.fine_tuning_callbacks.append(
            EarlyStopping(monitor='val_loss',
                          min_delta=0.0001,
                          mode='min',
                          patience=10,
                          verbose=1,
                          restore_best_weights=True)
        )

    def __keras_model(self):
        # ============================= INPUT ================================ #
        input_layer = Input((self.input_samples, self.n_cha, 1))
        # ================ BLOCK 1: SINGLE-CHANNEL ANALYSIS ================== #
        b1_units = list()
        for i in range(len(self.scales_samples)):
            unit = Conv2D(filters=self.filters_per_branch,
                          kernel_size=(self.scales_samples[i], 1),
                          kernel_initializer='he_normal',
                          padding='same')(input_layer)
            unit = BatchNormalization()(unit)
            unit = Activation(self.activation)(unit)
            unit = Dropout(self.dropout_rate)(unit)

            b1_units.append(unit)

        # Concatenation
        b1_out = keras.layers.concatenate(b1_units, axis=3)
        b1_out = AveragePooling2D((2, 1))(b1_out)

        # ================= BLOCK 2: SPATIAL FILTERING ======================= #
        b2_unit = DepthwiseConv2D((1, self.n_cha),
                                  use_bias=False,
                                  depth_multiplier=2,
                                  depthwise_constraint=max_norm(1.))(b1_out)
        b2_unit = BatchNormalization()(b2_unit)
        b2_unit = Activation(self.activation)(b2_unit)
        b2_unit = Dropout(self.dropout_rate)(b2_unit)
        b2_out = AveragePooling2D((2, 1))(b2_unit)

        # ================ BLOCK 3: MULTI-CHANNEL ANALYSIS =================== #
        b3_units = list()
        for i in range(len(self.scales_samples)):
            unit = Conv2D(filters=self.filters_per_branch,
                          kernel_size=(int(self.scales_samples[i] / 4), 1),
                          kernel_initializer='he_normal',
                          use_bias=False,
                          padding='same')(b2_out)
            unit = BatchNormalization()(unit)
            unit = Activation(self.activation)(unit)
            unit = Dropout(self.dropout_rate)(unit)

            b3_units.append(unit)

        # Concatenate + Average pooling
        b3_out = keras.layers.concatenate(b3_units, axis=3)
        b3_out = AveragePooling2D((2, 1))(b3_out)

        # ==================== BLOCK 4: OUTPUT-BLOCK ========================= #
        b4_u1 = Conv2D(filters=int(self.filters_per_branch *
                                   len(self.scales_samples) / 2),
                       kernel_size=(8, 1),
                       kernel_initializer='he_normal',
                       use_bias=False,
                       padding='same')(b3_out)
        b4_u1 = BatchNormalization()(b4_u1)
        b4_u1 = Activation(self.activation)(b4_u1)
        b4_u1 = AveragePooling2D((2, 1))(b4_u1)
        b4_u1 = Dropout(self.dropout_rate)(b4_u1)

        b4_u2 = Conv2D(filters=int(self.filters_per_branch *
                                   len(self.scales_samples) / 4),
                       kernel_size=(4, 1),
                       kernel_initializer='he_normal',
                       use_bias=False,
                       padding='same')(b4_u1)
        b4_u2 = BatchNormalization()(b4_u2)
        b4_u2 = Activation(self.activation)(b4_u2)
        b4_u2 = AveragePooling2D((2, 1))(b4_u2)
        b4_out = Dropout(self.dropout_rate)(b4_u2)

        # =========================== OUTPUT ================================= #
        # Output layer
        output_layer = Flatten()(b4_out)
        output_layer = Dense(self.n_classes, activation='softmax')(output_layer)
        # ============================ MODEL ================================= #
        # Optimizer
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate,
                                          beta_1=0.9, beta_2=0.999,
                                          amsgrad=False)
        # Create and compile model
        model = keras.models.Model(inputs=input_layer,
                                   outputs=output_layer)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        return model

    @staticmethod
    def transform_data(X, y=None):
        """Transforms input data to the correct dimensions for EEG-Inception

        Parameters
        ----------
        X: np.ndarray
            Feature matrix. If shape is [n_observ x n_samples x n_channels],
            this matrix will be adapted to the input dimensions of EEG-Inception
            [n_observ x n_samples x n_channels x 1]
        y: np.ndarray
            Labels array. If labels are in categorical format, they will be
            converted to one-hot encoding.
        """
        if len(X.shape) == 3 or X.shape[-1] != 1:
            X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
        if y is None:
            return X
        else:
            if len(y.shape) == 1 or y.shape[-1] == 1:
                y = classification_utils.one_hot_labels(y)
            return X, y

    def fit(self, X, y, fine_tuning=False, shuffle_before_fit=False, **kwargs):
        """Fit the model. All additional keras parameters of class
        tensorflow.keras.Model will pass through. See keras documentation to
        know what can you do: https://keras.io/api/models/model_training_apis/.

        If no parameters are specified, some default options are set [1]:

            - Epochs: 100 if fine_tuning else 500
            - Batch size: 32 if fine_tuning else 1024
            - Callbacks: self.fine_tuning_callbacks if fine_tuning else
                self.training_callbacks

        Parameters
        ----------
        X: np.ndarray
            Feature matrix. If shape is [n_observ x n_samples x n_channels],
            this matrix will be adapted to the input dimensions of EEG-Inception
            [n_observ x n_samples x n_channels x 1]
        y: np.ndarray
            Feature matrix. If shape is [n_observ x n_samples x n_channels],
            this matrix will be adapted to the input dimensions of EEG-Inception
            [n_observ x n_samples x n_channels x 1]
        fine_tuning: bool
            Set to True to use the default training parameters for fine
            tuning. False by default.
        shuffle_before_fit: bool
            If True, the data will be shuffled before training just once. Note
            that if you use the keras native argument 'shuffle', the data is
            shuffled each epoch.
        kwargs:
            Key-value arguments will be passed to the fit function of the model.
            This way, you can set your own training parameters using keras API.
            See https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
        """
        # Check numpy arrays
        X = np.array(X)
        y = np.array(y)
        # Check GPU
        if not tensorflow_integration.check_gpu_acceleration():
            warnings.warn('GPU acceleration is not available. The training '
                          'time is drastically reduced with GPU.')
        # Shuffle the data before fitting
        if shuffle_before_fit:
            X, y = sk_utils.shuffle(X, y)
        # Training parameters
        if not fine_tuning:
            # Rewrite default values
            kwargs['epochs'] = kwargs['epochs'] if \
                'epochs' in kwargs else 500
            kwargs['batch_size'] = kwargs['batch_size'] if \
                'batch_size' in kwargs else 1024
            kwargs['callbacks'] = kwargs['callbacks'] if \
                'callbacks' in kwargs else self.training_callbacks
        else:
            kwargs['epochs'] = kwargs['epochs'] if \
                'epochs' in kwargs else 100
            kwargs['batch_size'] = kwargs['batch_size'] if \
                'batch_size' in kwargs else 32
            kwargs['callbacks'] = kwargs['callbacks'] if \
                'callbacks' in kwargs else self.fine_tuning_callbacks

        # Transform data if necessary
        X, y = self.transform_data(X, y)
        # Fit
        with tf.device(tensorflow_integration.get_tf_device_name()):
            return self.model.fit(X, y, **kwargs)

    def predict_proba(self, X):
        """Model prediction scores for the given features.

        Parameters
        ----------
        X: np.ndarray
            Feature matrix. If shape is [n_observ x n_samples x n_channels],
            this matrix will be adapted to the input dimensions of EEG-Inception
            [n_observ x n_samples x n_channels x 1]
        """
        # Check numpy arrays
        X = np.array(X)
        # Transform data if necessary
        X = self.transform_data(X)
        # Predict
        with tf.device(tensorflow_integration.get_tf_device_name()):
            return self.model.predict(X)

    def to_pickleable_obj(self):
        # Parameters
        kwargs = {
            'input_time': self.input_time,
            'fs': self.fs,
            'n_cha': self.n_cha,
            'filters_per_branch': self.filters_per_branch,
            'scales_time': self.scales_time,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'n_classes': self.n_classes,
            'learning_rate': self.learning_rate,
        }
        weights = self.model.get_weights()
        # Pickleable object
        pickleable_obj = {
            'kwargs': kwargs,
            'weights': weights
        }
        return pickleable_obj

    @classmethod
    def from_pickleable_obj(cls, pickleable_obj):
        pickleable_obj['kwargs']['gpu_acceleration'] = None
        model = cls(**pickleable_obj['kwargs'])
        model.model.set_weights(pickleable_obj['weights'])
        return model

    def set_weights(self, weights):
        return self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def save_weights(self, path):
        return self.model.save_weights(path)

    def load_weights(self, weights_path):
        return self.model.load_weights(weights_path)


class EEGNet(components.ProcessingMethod):
    """EEG-Inception as described in Lawhern et al. 2018 [1]. This model is
    specifically designed for EEG classification tasks.

    Original source https://github.com/vlawhern/arl-eegmodels

    References
    ----------
    [1] Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung,
    C. P., & Lance, B. J. (2018). EEGNet: a compact convolutional neural network
    for EEG-based brain–computer interfaces. Journal of neural engineering,
    15(5), 056013.
    """
    def __init__(self, nb_classes, n_cha=64, samples=128, dropout_rate=0.5,
                 kern_length=64, F1=8, D=2, F2=16, norm_rate=0.25,
                 dropout_type='Dropout', learning_rate=0.001,
                 gpu_acceleration=None):

        # Super call
        super().__init__(fit=[], predict_proba=['y_pred'])

        # Tensorflow config
        if gpu_acceleration is None:
            tensorflow_integration.check_tf_config(autoconfig=True)
        else:
            tensorflow_integration.config_tensorflow(gpu_acceleration)
        if tensorflow_integration.check_gpu_acceleration():
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        tf.keras.backend.set_image_data_format('channels_last')

        # Parameters
        self.nb_classes = nb_classes
        self.n_cha = n_cha
        self.samples = samples
        self.dropout_rate = dropout_rate
        self.kern_length = kern_length
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.norm_rate = norm_rate
        self.dropout_type = dropout_type
        self.learning_rate = learning_rate

        # Create model
        self.model = self.__keras_model()

        # Create training callbacks
        self.training_callbacks = list()
        self.training_callbacks.append(
            EarlyStopping(monitor='val_loss',
                          min_delta=0.001,
                          mode='min',
                          patience=20,
                          verbose=1,
                          restore_best_weights=True)
        )
        # Create fine-tuning callbacks
        self.fine_tuning_callbacks = list()
        self.fine_tuning_callbacks.append(
            EarlyStopping(monitor='val_loss',
                          min_delta=0.0001,
                          mode='min',
                          patience=10,
                          verbose=1,
                          restore_best_weights=True)
        )

    def __keras_model(self):
        """ Keras Implementation of EEGNet
        http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta
        Note that this implements the newest version of EEGNet and NOT the
        earlier version (version v1 and v2 on arxiv). We strongly recommend
        using this architecture as it performs much better and has nicer
        properties than our earlier version. For example:

            1. Depthwise Convolutions to learn spatial filters within a
            temporal convolution. The use of the depth_multiplier option maps
            exactly to the number of spatial filters learned within a temporal
            filter. This matches the setup of algorithms like FBCSP which learn
            spatial filters within each filter in a filter-bank. This also
            limits the number of free parameters to fit when compared to a
            fully-connected convolution.

            2. Separable Convolutions to learn how to optimally combine spatial
            filters across temporal bands. Separable Convolutions are Depthwise
            Convolutions followed by (1x1) Pointwise Convolutions.


        While the original paper used Dropout, we found that SpatialDropout2D
        sometimes produced slightly better results for classification of ERP
        signals. However, SpatialDropout2D significantly reduced performance
        on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
        the default Dropout in most cases.

        Assumes the input signal is sampled at 128Hz. If you want to use this
        model for any other sampling rate you will need to modify the lengths of
        temporal kernels and average pooling size in blocks 1 and 2 as needed
        (double the kernel lengths for double the sampling rate, etc). Note
        that we haven't tested the model performance with this rule so this
        may not work well.

        The model with default parameters gives the EEGNet-8,2 model as
        discussed in the paper. This model should do pretty well in general,
        although it is advised to do some model searching to get optimal
        performance on your particular dataset.

        We set F2 = F1 * D (number of input filters = number of output filters)
        for the SeparableConv2D layer. We haven't extensively tested other
        values of this parameter (say, F2 < F1 * D for compressed learning,
        and F2 > F1 * D for overcomplete). We believe the main parameters to
        focus on are F1 and D.

        Inputs:

          nb_classes      : int, number of classes to classify
          Chans, Samples  : number of channels and time points in the EEG data
          dropoutRate     : dropout fraction
          kernLength      : length of temporal convolution in first layer. We
                            found that setting this to be half the sampling
                            rate worked well in practice. For the SMR dataset in
                            particular since the data was high-passed at 4Hz
                            we used a kernel length of 32.
          F1, F2          : number of temporal filters (F1) and number of
                            pointwise filters (F2) to learn. Default: F1 = 8,
                            F2 = F1 * D.
          D               : number of spatial filters to learn within each
                            temporal convolution. Default: D = 2
          dropoutType     : Either SpatialDropout2D or Dropout, passed as a
                            string.
        """

        if self.dropout_type == 'SpatialDropout2D':
            dropout_type = SpatialDropout2D
        elif self.dropout_type == 'Dropout':
            dropout_type = Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D '
                             'or Dropout, passed as a string.')

        input1 = Input(shape=(self.n_cha, self.samples, 1))

        ##################################################################
        block1 = Conv2D(self.F1, (1, self.kern_length), padding='same',
                        input_shape=(self.n_cha, self.samples, 1),
                        use_bias=False)(input1)
        block1 = BatchNormalization()(block1)
        block1 = DepthwiseConv2D((self.n_cha, 1), use_bias=False,
                                 depth_multiplier=self.D,
                                 depthwise_constraint=max_norm(1.))(block1)
        block1 = BatchNormalization()(block1)
        block1 = Activation('elu')(block1)
        block1 = AveragePooling2D((1, 4))(block1)
        block1 = dropout_type(self.dropout_rate)(block1)

        block2 = SeparableConv2D(self.F2, (1, 16),
                                 use_bias=False, padding='same')(block1)
        block2 = BatchNormalization()(block2)
        block2 = Activation('elu')(block2)
        block2 = AveragePooling2D((1, 8))(block2)
        block2 = dropout_type(self.dropout_rate)(block2)

        flatten = Flatten(name='flatten')(block2)

        dense = Dense(self.nb_classes, name='dense',
                      kernel_constraint=max_norm(self.norm_rate))(flatten)
        softmax = Activation('softmax', name='softmax')(dense)

        # Create and compile model
        model = keras.models.Model(inputs=input1, outputs=softmax)
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate,
                                          beta_1=0.9, beta_2=0.999,
                                          amsgrad=False)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        return model

    @staticmethod
    def transform_data(X, y=None):
        """Transforms input data to the correct dimensions for EEG-Inception

        Parameters
        ----------
        X: np.ndarray
            Feature matrix. If shape is [n_observ x n_samples x n_channels],
            this matrix will be adapted to the input dimensions of EEG-Inception
            [n_observ x n_samples x n_channels x 1]
        y: np.ndarray
            Labels array. If labels are in categorical format, they will be
            converted to one-hot encoding.
        """
        if len(X.shape) == 3 or X.shape[-1] != 1:
            X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
            X = np.swapaxes(X, 1, 2)
        if y is None:
            return X
        else:
            if len(y.shape) == 1 or y.shape[-1] == 1:
                y = classification_utils.one_hot_labels(y)
            return X, y

    def fit(self, X, y, fine_tuning=False, shuffle_before_fit=False, **kwargs):
        """Fit the model. All additional keras parameters of class
        tensorflow.keras.Model will pass through. See keras documentation to
        know what can you do: https://keras.io/api/models/model_training_apis/.

        If no parameters are specified, some default options are set [1]:

            - Epochs: 100 if fine_tuning else 500
            - Batch size: 32 if fine_tuning else 1024
            - Callbacks: self.fine_tuning_callbacks if fine_tuning else
                self.training_callbacks

        Parameters
        ----------
        X: np.ndarray
            Feature matrix. If shape is [n_observ x n_samples x n_channels],
            this matrix will be adapted to the input dimensions of EEG-Inception
            [n_observ x n_samples x n_channels x 1]
        y: np.ndarray
            Feature matrix. If shape is [n_observ x n_samples x n_channels],
            this matrix will be adapted to the input dimensions of EEG-Inception
            [n_observ x n_samples x n_channels x 1]
        fine_tuning: bool
            Set to True to use the default training parameters for fine
            tuning. False by default.
        shuffle_before_fit: bool
            If True, the data will be shuffled before training.
        kwargs:
            Key-value arguments will be passed to the fit function of the model.
            This way, you can set your own training parameters for keras.
            See https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
        """
        # Check numpy arrays
        X = np.array(X)
        y = np.array(y)
        # Check GPU
        if not tensorflow_integration.check_gpu_acceleration():
            warnings.warn('GPU acceleration is not available. The training '
                          'time is drastically reduced with GPU.')

        # Shuffle the data before fitting
        if shuffle_before_fit:
            X, y = sk_utils.shuffle(X, y)

        # Training parameters
        if not fine_tuning:
            # Rewrite default values
            kwargs['epochs'] = kwargs['epochs'] if \
                'epochs' in kwargs else 500
            kwargs['batch_size'] = kwargs['batch_size'] if \
                'batch_size' in kwargs else 1024
            kwargs['callbacks'] = kwargs['callbacks'] if \
                'callbacks' in kwargs else self.training_callbacks
        else:
            kwargs['epochs'] = kwargs['epochs'] if \
                'epochs' in kwargs else 100
            kwargs['batch_size'] = kwargs['batch_size'] if \
                'batch_size' in kwargs else 32
            kwargs['callbacks'] = kwargs['callbacks'] if \
                'callbacks' in kwargs else self.fine_tuning_callbacks

        # Transform data if necessary
        X, y = self.transform_data(X, y)
        # Fit
        with tf.device(tensorflow_integration.get_tf_device_name()):
            return self.model.fit(X, y, **kwargs)

    def predict_proba(self, X):
        """Model prediction scores for the given features.

        Parameters
        ----------
        X: np.ndarray
            Feature matrix. If shape is [n_observ x n_samples x n_channels],
            this matrix will be adapted to the input dimensions of EEG-Inception
            [n_observ x n_samples x n_channels x 1]
        """
        # Check numpy arrays
        X = np.array(X)
        # Transform data if necessary
        X = self.transform_data(X)
        # Predict
        with tf.device(tensorflow_integration.get_tf_device_name()):
            return self.model.predict(X)

    def to_pickleable_obj(self):
        # Key data
        kwargs = {
            'nb_classes': self.nb_classes,
            'n_cha': self.n_cha,
            'samples': self.samples,
            'dropout_rate': self.dropout_rate,
            'kern_length': self.kern_length,
            'F1': self.F1,
            'D': self.D,
            'F2': self.F2,
            'norm_rate': self.norm_rate,
            'dropout_type': self.dropout_type,
            'learning_rate': self.learning_rate
        }
        weights = self.model.get_weights()
        # Pickleable object
        pickleable_obj = {
            'kwargs': kwargs,
            'weights': weights
        }
        return pickleable_obj

    @classmethod
    def from_pickleable_obj(cls, pickleable_obj):
        pickleable_obj['kwargs']['gpu_acceleration'] = None
        model = cls(**pickleable_obj['kwargs'])
        model.model.set_weights(pickleable_obj['weights'])
        return model

    def set_weights(self, weights):
        return self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def save_weights(self, path):
        return self.model.save_weights(path)

    def load_weights(self, weights_path):
        return self.model.load_weights(weights_path)


class EEGSym(components.ProcessingMethod):
    """EEGSym as described in Pérez-Velasco et al. 2022 [1]. This
    model is specifically designed for EEG classification tasks.

    References
    ----------
    [1] Pérez-Velasco, S., Santamaría-Vázquez, E., Martínez-Cagigal, V.,
    Marcos-Martínez, D., & Hornero, R. (2022). EEGSym: Overcoming Intersubject
    Variability in Motor Imagery Based BCIs with Deep Learning. IEEE
    Transactions on Neural Systems and Rehabilitation Engineering.
    """
    #TODO: Implement automatic ordering of channels
    #TODO: Implement trial iterator and data augmentation
    def __init__(self, input_time=3000, fs=128, n_cha=8, filters_per_branch=24,
           scales_time=(500, 250, 125), dropout_rate=0.4, activation='elu',
           n_classes=2, learning_rate=0.001, ch_lateral=3,
           spatial_resnet_repetitions=1, residual=True, symmetric=True,
                 gpu_acceleration=None):
        # Super call
        super().__init__(fit=[], predict_proba=['y_pred'])

        # Tensorflow config
        if gpu_acceleration is None:
            tensorflow_integration.check_tf_config(autoconfig=True)
        else:
            tensorflow_integration.config_tensorflow(gpu_acceleration)
        if tensorflow_integration.check_gpu_acceleration():
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        tf.keras.backend.set_image_data_format('channels_last')

        # Parameters
        self.input_time = input_time
        self.fs = fs
        self.n_cha = n_cha
        self.filters_per_branch = filters_per_branch
        self.scales_time = scales_time
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.ch_lateral = ch_lateral
        self.spatial_resnet_repetitions = spatial_resnet_repetitions
        self.residual = residual
        self.symmetric = symmetric
        self.initialized = False

        # Useful variables
        self.input_samples = int(input_time * fs / 1000)
        self.scales_samples = [int(s * fs / 1000) for s in scales_time]

        # Create model
        self.model = self.__keras_model()
        # Create training callbacks
        self.training_callbacks = list()
        self.training_callbacks.append(
            EarlyStopping(monitor='val_loss',
                          min_delta=0.001,
                          mode='min',
                          patience=20,
                          verbose=1,
                          restore_best_weights=True)
        )
        # Create fine-tuning callbacks
        self.fine_tuning_callbacks = list()
        self.fine_tuning_callbacks.append(
            EarlyStopping(monitor='val_loss',
                          min_delta=0.0001,
                          mode='min',
                          patience=10,
                          verbose=1,
                          restore_best_weights=True)
        )

    def __keras_model(self):
        # ============================= INPUT ================================ #
        # ==================================================================== #
        input_layer = Input((self.input_samples, self.n_cha, 1))
        input = tf.expand_dims(input_layer, axis=1)
        ncha = self.n_cha
        if self.symmetric:
            superposition = False
            if self.ch_lateral < self.n_cha // 2:
                superposition = True
            ncha = self.n_cha - self.ch_lateral

            left_idx = list(range(self.ch_lateral))
            ch_left = tf.gather(input, indices=left_idx, axis=-2)
            right_idx = list(np.array(left_idx) + int(ncha))
            ch_right = tf.gather(input, indices=right_idx, axis=-2)

            if superposition:
                central_idx = list(
                    np.array(range(ncha - self.ch_lateral)) +
                    self.ch_lateral)
                ch_central = tf.gather(input, indices=central_idx, axis=-2)

                left_init = keras.layers.concatenate((ch_left, ch_central),
                                                     axis=-2)
                right_init = keras.layers.concatenate((ch_right, ch_central),
                                                      axis=-2)
            else:
                left_init = ch_left
                right_init = ch_right

            input = keras.layers.concatenate((left_init, right_init), axis=1)
            division = 2
        else:
            division = 1
        # ================ GENERAL INCEPTION/RESIDUAL MODULE ================= #
        # ==================================================================== #

        # ====================== TEMPOSPATIAL ANALYSIS ======================= #
        # ==================================================================== #
        # ========================== Inception (x2) ========================== #
        b1_out = self.general_module([input],
                                scales_samples=self.scales_samples,
                                filters_per_branch=self.filters_per_branch,
                                ncha=ncha,
                                activation=self.activation,
                                dropout_rate=self.dropout_rate, average=2,
                                spatial_resnet_repetitions=
                                self.spatial_resnet_repetitions,
                                residual=self.residual, init=True)

        b2_out = self.general_module(b1_out, scales_samples=[int(x / 4) for x in
                                                        self.scales_samples],
                                filters_per_branch=self.filters_per_branch,
                                ncha=ncha,
                                activation=self.activation,
                                dropout_rate=self.dropout_rate, average=2,
                                spatial_resnet_repetitions=
                                self.spatial_resnet_repetitions,
                                residual=self.residual)
        # ============================== Residual (x3) =========================== #
        b3_u1 = self.general_module(b2_out, scales_samples=[16],
                               filters_per_branch=int(
                                   self.filters_per_branch * len(
                                       self.scales_samples) / 2),
                               ncha=ncha,
                               activation=self.activation,
                               dropout_rate=self.dropout_rate, average=2,
                               spatial_resnet_repetitions=
                               self.spatial_resnet_repetitions,
                               residual=self.residual)
        b3_u1 = self.general_module(b3_u1,
                               scales_samples=[8],
                               filters_per_branch=int(
                                   self.filters_per_branch * len(
                                       self.scales_samples) / 2),
                               ncha=ncha,
                               activation=self.activation,
                               dropout_rate=self.dropout_rate, average=2,
                               spatial_resnet_repetitions=
                               self.spatial_resnet_repetitions,
                               residual=self.residual)
        b3_u2 = self.general_module(b3_u1, scales_samples=[4],
                               filters_per_branch=int(
                                   self.filters_per_branch * len(
                                       self.scales_samples) / 4),
                               ncha=ncha,
                               activation=self.activation,
                               dropout_rate=self.dropout_rate, average=2,
                               spatial_resnet_repetitions=
                               self.spatial_resnet_repetitions,
                               residual=self.residual)

        t_red = b3_u2[0]
        for _ in range(1):
            t_red_temp = t_red
            t_red_temp = Conv3D(kernel_size=(1, 4, 1),
                                filters=int(self.filters_per_branch * len(
                                    self.scales_samples) / 4),
                                use_bias=False,
                                strides=(1, 1, 1),
                                kernel_initializer='he_normal',
                                padding='same')(t_red_temp)
            t_red_temp = BatchNormalization()(t_red_temp)
            t_red_temp = Activation(self.activation)(t_red_temp)
            t_red_temp = Dropout(self.dropout_rate)(t_red_temp)

            if self.residual:
                t_red = Add()([t_red, t_red_temp])
            else:
                t_red = t_red_temp

        t_red = AveragePooling3D((1, 2, 1))(t_red)
        # ========================= CHANNEL MERGING ========================== #
        # ==================================================================== #
        ch_merg = t_red
        if self.residual:
            for _ in range(2):
                ch_merg_temp = ch_merg
                ch_merg_temp = Conv3D(kernel_size=(division, 1, ncha),
                                      filters=int(self.filters_per_branch * len(
                                          self.scales_samples) / 4),
                                      use_bias=False,
                                      strides=(1, 1, 1),
                                      kernel_initializer='he_normal',
                                      padding='valid')(ch_merg_temp)
                ch_merg_temp = BatchNormalization()(ch_merg_temp)
                ch_merg_temp = Activation(self.activation)(ch_merg_temp)
                ch_merg_temp = Dropout(self.dropout_rate)(ch_merg_temp)

                ch_merg = Add()([ch_merg, ch_merg_temp])

            ch_merg = Conv3D(kernel_size=(division, 1, ncha),
                             filters=int(
                                 self.filters_per_branch * len(
                                     self.scales_samples) / 4),
                             groups=int(
                                 self.filters_per_branch * len(
                                     self.scales_samples) / 8),
                             use_bias=False,
                             padding='valid')(ch_merg)
            ch_merg = BatchNormalization()(ch_merg)
            ch_merg = Activation(self.activation)(ch_merg)
            ch_merg = Dropout(self.dropout_rate)(ch_merg)
        else:
            if self.symmetric:
                ch_merg = Conv3D(kernel_size=(division, 1, 1),
                                 filters=int(
                                     self.filters_per_branch * len(
                                         self.scales_samples) / 4),
                                 groups=int(
                                     self.filters_per_branch * len(
                                         self.scales_samples) / 8),
                                 use_bias=False,
                                 padding='valid')(ch_merg)
                ch_merg = BatchNormalization()(ch_merg)
                ch_merg = Activation(self.activation)(ch_merg)
                ch_merg = Dropout(self.dropout_rate)(ch_merg)
        # ========================== TEMPORAL MERGING ============================ #
        # ======================================================================== #
        t_merg = ch_merg
        for _ in range(1):
            if self.residual:
                t_merg_temp = t_merg
                t_merg_temp = Conv3D(kernel_size=(1, self.input_samples // 64,
                                                  1),
                                     filters=int(self.filters_per_branch * len(
                                         self.scales_samples) / 4),
                                     use_bias=False,
                                     strides=(1, 1, 1),
                                     kernel_initializer='he_normal',
                                     padding='valid')(t_merg_temp)
                t_merg_temp = BatchNormalization()(t_merg_temp)
                t_merg_temp = Activation(self.activation)(t_merg_temp)
                t_merg_temp = Dropout(self.dropout_rate)(t_merg_temp)

                t_merg = Add()([t_merg, t_merg_temp])
            else:
                t_merg_temp = t_merg
                t_merg_temp = Conv3D(kernel_size=(1, self.input_samples // 64, 1),
                                     filters=int(self.filters_per_branch * len(
                                         self.scales_samples) / 4),
                                     use_bias=False,
                                     strides=(1, 1, 1),
                                     kernel_initializer='he_normal',
                                     padding='same')(t_merg_temp)
                t_merg_temp = BatchNormalization()(t_merg_temp)
                t_merg_temp = Activation(self.activation)(t_merg_temp)
                t_merg_temp = Dropout(self.dropout_rate)(t_merg_temp)
                t_merg = t_merg_temp

        t_merg = Conv3D(kernel_size=(1, self.input_samples // 64, 1),
                        filters=int(
                            self.filters_per_branch * len(
                                self.scales_samples) / 4) * 2,
                        groups=int(
                            self.filters_per_branch * len(
                                self.scales_samples) / 4),
                        use_bias=False,
                        padding='valid')(t_merg)
        t_merg = BatchNormalization()(t_merg)
        t_merg = Activation(self.activation)(t_merg)
        t_merg = Dropout(self.dropout_rate)(t_merg)
        # =============================== OUTPUT ================================= #
        output = t_merg
        for _ in range(4):
            output_temp = output
            output_temp = Conv3D(kernel_size=(1, 1, 1),
                                 filters=int(self.filters_per_branch * len(
                                     self.scales_samples) / 2),
                                 use_bias=False,
                                 strides=(1, 1, 1),
                                 kernel_initializer='he_normal',
                                 padding='valid')(output_temp)
            output_temp = BatchNormalization()(output_temp)
            output_temp = Activation(self.activation)(output_temp)
            output_temp = Dropout(self.dropout_rate)(output_temp)
            if self.residual:
                output = Add()([output, output_temp])
            else:
                output = output_temp
        output = Flatten()(output)
        output_layer = Dense(self.n_classes, activation='softmax')(output)
        # ============================ MODEL ================================= #
        # Optimizer
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate,
                                          beta_1=0.9, beta_2=0.999,
                                          amsgrad=False)
        # Create and compile model
        model = keras.models.Model(inputs=input_layer,
                                   outputs=output_layer)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        return model

    @staticmethod
    def general_module(input, scales_samples, filters_per_branch, ncha,
                       activation, dropout_rate, average,
                       spatial_resnet_repetitions=1, residual=True,
                       init=False):
        """General inception/residual module.

            This function returns the input with the operations of a
            inception or residual module from the publication applied.

            Parameters
            ----------
            input : list
                List of input blocks to the module.
            scales_samples : list
                List of samples size of the temporal operations kernels.
            filters_per_branch : int
                Number of filters in each Inception branch. The number should be
                multiplies of 8.
            ncha :
                Number of input channels.
            activation : str
                Activation
            dropout_rate : float
                Dropout rate
            spatial_resnet_repetitions: int
                Number of repetitions of the operations of spatial analysis at
                each step of the spatiotemporal analysis. In the original
                publication this value was set to 1 and not tested its
                variations.
            residual : Bool
                If the residual operations are present in EEGSym architecture.
            init : Bool
                If the module is the first one applied to the input, to apply a
                channel merging operation if the architecture does not include
                residual operations.

            Returns
            -------
            block_out : list
                List of outputs modules
        """
        block_units = list()
        unit_conv_t = list()
        unit_batchconv_t = list()

        for i in range(len(scales_samples)):
            unit_conv_t.append(Conv3D(filters=filters_per_branch,
                                      kernel_size=(1, scales_samples[i], 1),
                                      kernel_initializer='he_normal',
                                      padding='same'))
            unit_batchconv_t.append(BatchNormalization())

        if ncha != 1:
            unit_dconv = list()
            unit_batchdconv = list()
            unit_conv_s = list()
            unit_batchconv_s = list()
            for i in range(spatial_resnet_repetitions):
                # 3D Implementation of DepthwiseConv
                unit_dconv.append(Conv3D(kernel_size=(1, 1, ncha),
                                         filters=filters_per_branch * len(
                                             scales_samples),
                                         groups=filters_per_branch * len(
                                             scales_samples),
                                         use_bias=False,
                                         padding='valid'))
                unit_batchdconv.append(BatchNormalization())

                unit_conv_s.append(Conv3D(kernel_size=(1, 1, ncha),
                                          filters=filters_per_branch,
                                          # groups=filters_per_branch,
                                          use_bias=False,
                                          strides=(1, 1, 1),
                                          kernel_initializer='he_normal',
                                          padding='valid'))
                unit_batchconv_s.append(BatchNormalization())

            unit_conv_1 = Conv3D(kernel_size=(1, 1, 1),
                                 filters=filters_per_branch,
                                 use_bias=False,
                                 kernel_initializer='he_normal',
                                 padding='valid')
            unit_batchconv_1 = BatchNormalization()

        # Temporal analysis stage of the module
        for j in range(len(input)):
            block_side_units = list()
            for i in range(len(scales_samples)):
                unit = input[j]
                unit = unit_conv_t[i](unit)

                unit = unit_batchconv_t[i](unit)
                unit = Activation(activation)(unit)
                unit = Dropout(dropout_rate)(unit)

                block_side_units.append(unit)
            block_units.append(block_side_units)
        # Concatenation of possible inception modules
        block_out = list()
        for j in range(len(input)):
            if len(block_units[j]) != 1:
                block_out.append(
                    keras.layers.concatenate(block_units[j], axis=-1))
                block_out_temp = input[j]
            else:
                block_out.append(block_units[j][0])
                block_out_temp = input[j]
                block_out_temp = unit_conv_1(block_out_temp)
                block_out_temp = unit_batchconv_1(block_out_temp)
                block_out_temp = Activation(activation)(block_out_temp)
                block_out_temp = Dropout(dropout_rate)(block_out_temp)

            if residual:
                block_out[j] = Add()([block_out[j], block_out_temp])
            else:
                block_out[j] = block_out_temp

            if average != 1:
                block_out[j] = AveragePooling3D((1, average, 1))(block_out[j])

        # Spatial analysis stage of the module
        if ncha != 1:
            for i in range(spatial_resnet_repetitions):
                block_out_temp = list()
                for j in range(len(input)):
                    if len(scales_samples) != 1:
                        if residual:
                            block_out_temp.append(block_out[j])

                            block_out_temp[j] = unit_dconv[i](block_out_temp[j])

                            block_out_temp[j] = unit_batchdconv[i](
                                block_out_temp[j])
                            block_out_temp[j] = Activation(activation)(
                                block_out_temp[j])
                            block_out_temp[j] = Dropout(dropout_rate)(
                                block_out_temp[j])

                            block_out[j] = Add()(
                                [block_out[j], block_out_temp[j]])

                        elif init:
                            block_out[j] = unit_dconv[i](block_out[j])
                            block_out[j] = unit_batchdconv[i](block_out[j])
                            block_out[j] = Activation(activation)(block_out[j])
                            block_out[j] = Dropout(dropout_rate)(block_out[j])
                    else:
                        if residual:
                            block_out_temp.append(block_out[j])

                            block_out_temp[j] = unit_conv_s[i](
                                block_out_temp[j])
                            block_out_temp[j] = unit_batchconv_s[i](
                                block_out_temp[j])
                            block_out_temp[j] = Activation(activation)(
                                block_out_temp[j])
                            block_out_temp[j] = Dropout(dropout_rate)(
                                block_out_temp[j])

                            block_out[j] = Add()(
                                [block_out[j], block_out_temp[j]])
        return block_out

    @staticmethod
    def transform_data(X, y=None):
        """Transforms input data to the correct dimensions for EEGSym

        Parameters
        ----------
        X: np.ndarray
            Feature matrix. If shape is [n_observ x n_samples x n_channels],
            this matrix will be adapted to the input dimensions of EEG-Inception
            [n_observ x n_samples x n_channels x 1]
        y: np.ndarray
            Labels array. If labels are in categorical format, they will be
            converted to one-hot encoding.
        """
        if len(X.shape) == 3 or X.shape[-1] != 1:
            X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
        if y is None:
            return X
        else:
            if len(y.shape) == 1 or y.shape[-1] == 1:
                y = classification_utils.one_hot_labels(y)
            return X, y

    def preprocessing_function(self, augmentation=True):
        """Custom Data Augmentation for EEGSym.

            Parameters
            ----------
            augmentation : Bool
                If the augmentation is performed to the input.

            Returns
            -------
            data_augmentation : function
                Data augmentation performed to each trial
        """

        def data_augmentation(trial):
            """Custom Data Augmentation for EEGSym.

                Parameters
                ----------
                trial : tf.tensor
                    Input of the

                Returns
                -------
                data_augmentation : keras.models.Model
                    Data augmentation performed to each trial
            """

            samples, ncha, _ = trial.shape

            augmentations = dict()
            augmentations["patch_perturbation"] = 0
            augmentations["random_shift"] = 0
            augmentations["hemisphere_perturbation"] = 0
            augmentations["no_augmentation"] = 0

            # selectionables = ["patch_perturbation", "random_shift",
            #                   "hemisphere_perturbation", "no_augmentation"]
            # We eliminate hemisphere_perturbation due to it being very
            # dependant on the channels introduced
            selectionables = ["patch_perturbation", "random_shift",
                              "no_augmentation"]
            probabilities = None

            if augmentation:
                selection = np.random.choice(selectionables, p=probabilities)
                augmentations[selection] = 1

                method = np.random.choice((0, 2))
                std = 'self'
                # elif data_augmentation == 1:  # Random shift
                for _ in range(augmentations["random_shift"]):  # Random shift
                    # Select position where to erase that timeframe
                    position = 0
                    if position == 0:
                        samples_shifted = np.random.randint(low=1, high=int(
                            samples * 0.5 / 3))
                    else:
                        samples_shifted = np.random.randint(low=1, high=int(
                            samples * 0.1 / 3))

                    if method == 0:
                        shifted_samples = np.zeros((samples_shifted, ncha, 1))
                    else:
                        if std == 'self':
                            std_applied = np.std(trial)
                        else:
                            std_applied = std
                        center = 0
                        shifted_samples = np.random.normal(center, std_applied,
                                                           (samples_shifted,
                                                            ncha,
                                                            1))
                    if position == 0:
                        trial = np.concatenate((shifted_samples, trial),
                                               axis=0)[:samples]
                    else:
                        trial = np.concatenate((trial, shifted_samples),
                                               axis=0)[samples_shifted:]

                for _ in range(
                        augmentations[
                            "patch_perturbation"]):  # Patch perturbation
                    channels_affected = np.random.randint(low=1, high=ncha - 1)
                    pct_max = 1
                    pct_min = 0.2
                    pct_erased = np.random.uniform(low=pct_min, high=pct_max)
                    # Select time to be erased acording to pct_erased
                    # samples_erased = np.min((int(samples*ncha*pct_erased//channels_affected), samples))#np.random.randint(low=1, high=samples//3)
                    samples_erased = int(samples * pct_erased)
                    # Select position where to erase that timeframe
                    if samples_erased != samples:
                        samples_idx = np.arange(
                            samples_erased) + np.random.randint(
                            samples - samples_erased)
                    else:
                        samples_idx = np.arange(samples_erased)
                    # Select indexes to erase (always keep at least a channel)
                    channel_idx = np.random.permutation(np.arange(ncha))[
                                  :channels_affected]
                    channel_idx.sort()
                    for channel in channel_idx:
                        if method == 0:
                            trial[samples_idx, channel] = 0
                        else:
                            if std == 'self':
                                std_applied = np.std(trial[:, channel]) \
                                              * np.random.uniform(low=0.01,
                                                                  high=2)
                            else:
                                std_applied = std
                            center = 0
                            trial[samples_idx, channel] += \
                                np.random.normal(center, std_applied,
                                                 trial[samples_idx, channel,
                                                 :].shape)
                            # Standarize the channel again after the change
                            temp_trial_ch_mean = np.mean(trial[:, channel],
                                                         axis=0)
                            temp_trial_ch_std = np.std(trial[:, channel],
                                                       axis=0)
                            trial[:, channel] = (trial[:,
                                                 channel] - temp_trial_ch_mean) / temp_trial_ch_std

                for _ in range(augmentations["hemisphere_perturbation"]):
                    # Select side to mix/change for noise
                    left_right = np.random.choice((0, 1))
                    if method == 0:
                        if left_right == 1:
                            channel_idx = np.arange(ncha)[:int((ncha / 2) - 1)]
                            channel_mix = np.random.permutation(
                                channel_idx.copy())
                        else:
                            channel_idx = np.arange(ncha)[-int((ncha / 2) - 1):]
                            channel_mix = np.random.permutation(
                                channel_idx.copy())
                        temp_trial = trial.copy()
                        for channel, channel_mixed in zip(channel_idx,
                                                          channel_mix):
                            temp_trial[:, channel] = trial[:, channel_mixed]
                        trial = temp_trial
                    else:
                        if left_right == 1:
                            channel_idx = np.arange(ncha)[:int((ncha / 2) - 1)]
                        else:
                            channel_idx = np.arange(ncha)[-int((ncha / 2) - 1):]
                        for channel in channel_idx:
                            trial[:, channel] = np.random.normal(0, 1,
                                                                 trial[:,
                                                                 channel].shape)

            return trial

        return data_augmentation

    def trial_iterator(self, X, y, batch_size=32, shuffle=True,
                       augmentation=True):
        """Custom trial iterator to pretrain EEGSym.

            Parameters
            ----------
            X : tf.tensor
                Input tensor of  EEG features.
            y : tf.tensor
                Input tensor of  labels.
            batch_size : int
                Number of features in each batch.
            shuffle : Bool
                If the features are shuffled at each training epoch.
            augmentation : Bool
                If the augmentation is performed to the input.

            Returns
            -------
            trial_iterator : tf.keras.preprocessing.image.NumpyArrayIterator
                Iterator used to train the model.
        """

        trial_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=self.preprocessing_function(
                augmentation=augmentation))

        trial_iterator = tf.keras.preprocessing.image.NumpyArrayIterator(
            X, y, trial_data_generator, batch_size=batch_size, shuffle=shuffle,
            sample_weight=None,
            seed=None, data_format=None, save_to_dir=None, save_prefix='',
            save_format='png', subset=None, dtype=None
        )
        return trial_iterator

    def fit(self, X, y, fine_tuning=False, shuffle_before_fit=False,
            augmentation=True, **kwargs):
        """Fit the model. All additional keras parameters of class
        tensorflow.keras.Model will pass through. See keras documentation to
        know what can you do: https://keras.io/api/models/model_training_apis/.

        If no parameters are specified, some default options are set [1]:

            - Epochs: 100 if fine_tuning else 500
            - Batch size: 32 if fine_tuning else 2048
            - Callbacks: self.fine_tuning_callbacks if fine_tuning else
                self.training_callbacks

        Parameters
        ----------
        X: np.ndarray
            Feature matrix. If shape is [n_observ x n_samples x n_channels],
            this matrix will be adapted to the input dimensions of EEG-Inception
            [n_observ x n_samples x n_channels x 1]
        y: np.ndarray
            Feature matrix. If shape is [n_observ x n_samples x n_channels],
            this matrix will be adapted to the input dimensions of EEG-Inception
            [n_observ x n_samples x n_channels x 1]
        fine_tuning: bool
            Set to True to use the default training parameters for fine
            tuning. False by default.
        shuffle_before_fit: bool
            If True, the data will be shuffled before training just once. Note
            that if you use the keras native argument 'shuffle', the data is
            shuffled each epoch.
        kwargs:
            Key-value arguments will be passed to the fit function of the model.
            This way, you can set your own training parameters using keras API.
            See https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
        """
        # Check GPU
        if not tensorflow_integration.check_gpu_acceleration():
            warnings.warn('GPU acceleration is not available. The training '
                          'time is drastically reduced with GPU.')

        # Shuffle the data before fitting
        if shuffle_before_fit:
            X, y = sk_utils.shuffle(X, y)

        # Creation of validation split
        val_split = kwargs['validation_split'] if \
                'validation_split' in kwargs else 0.4

        val_idx = None
        train_idx = None
        for i, label in enumerate(np.unique(y)):
            idx = np.where((y == label))[0]
            np.random.shuffle(idx)
            val_idx_label = int(np.round(len(idx) * val_split))

            val_idx = np.concatenate((val_idx, idx[:val_idx_label]), axis=0) \
                if val_idx is not None else np.array(idx[:val_idx_label])
            train_idx = np.concatenate((train_idx, idx[val_idx_label:]), axis=0) \
                if train_idx is not None else np.array(idx[val_idx_label:])

        # Training parameters
        if not fine_tuning:
            # Rewrite default values
            kwargs['epochs'] = kwargs['epochs'] if \
                'epochs' in kwargs else 500
            kwargs['batch_size'] = kwargs['batch_size'] if \
                'batch_size' in kwargs else 32
            kwargs['callbacks'] = kwargs['callbacks'] if \
                'callbacks' in kwargs else self.training_callbacks
        else:
            kwargs['epochs'] = kwargs['epochs'] if \
                'epochs' in kwargs else 100
            kwargs['batch_size'] = kwargs['batch_size'] if \
                'batch_size' in kwargs else 32
            kwargs['callbacks'] = kwargs['callbacks'] if \
                'callbacks' in kwargs else self.fine_tuning_callbacks
        if self.initialized:
            for layer in self.model.layers[:-1]:
                layer.trainable = False
        # Transform data if necessary
        X, y = self.transform_data(X, y)
        # Fit
        with tf.device(tensorflow_integration.get_tf_device_name()):
            # return self.model.fit(X, y, **kwargs)
            return self.model.fit(self.trial_iterator(X[train_idx],
                                                      y[train_idx],
                                                      batch_size=kwargs[
                                                          'batch_size'],
                                                      augmentation=augmentation),
                                  validation_data=(X[val_idx], y[val_idx]),
                                  **kwargs)

    def symmetric_channels(self, X, channels):
        """This function takes a set of channels and puts them in a symmetric
        input needed to apply EEGSym.
        """
        left = list()
        right = list()
        middle = list()
        for channel in channels:
            if channel[-1].isnumeric():
                if int(channel[-1]) % 2 == 0:
                    right.append(channel)
                else:
                    left.append(channel)
            else:
                middle.append(channel)
        ordered_channels = left + middle + right
        index_channels = [channels.index(channel) for channel in
                          ordered_channels]
        return np.array(X)[:, :, index_channels], list(np.array(channels)[
            index_channels])

    def predict_proba(self, X):
        """Model prediction scores for the given features.

        Parameters
        ----------
        X: np.ndarray
            Feature matrix. If shape is [n_observ x n_samples x n_channels],
            this matrix will be adapted to the input dimensions of EEG-Sym
            [n_observ x n_samples x n_channels x 1]
        """
        # Transform data if necessary
        X = self.transform_data(X)
        # Predict
        with tf.device(tensorflow_integration.get_tf_device_name()):
            return self.model.predict(X)

    def to_pickleable_obj(self):
        # Parameters
        kwargs = {
            'input_time': self.input_time,
            'fs': self.fs,
            'n_cha': self.n_cha,
            'filters_per_branch': self.filters_per_branch,
            'scales_time': self.scales_time,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'n_classes': self.n_classes,
            'learning_rate': self.learning_rate,
            'ch_lateral': self.ch_lateral
        }
        weights = self.model.get_weights()
        # Pickleable object
        pickleable_obj = {
            'kwargs': kwargs,
            'weights': weights
        }
        return pickleable_obj

    @classmethod
    def from_pickleable_obj(cls, pickleable_obj):
        pickleable_obj['kwargs']['gpu_acceleration'] = None
        model = cls(**pickleable_obj['kwargs'])
        model.model.set_weights(pickleable_obj['weights'])
        return model

    def set_weights(self, weights):
        return self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def save_weights(self, path):
        return self.model.save_weights(path)

    def load_weights(self, weights_path):
        self.initialized = True
        return self.model.load_weights(weights_path)
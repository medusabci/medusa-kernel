# Built-in imports
import os, warnings

# External imports
from tqdm import tqdm

# Medusa imports
from medusa import components
from medusa.classification_utils import *
from medusa import pytorch_integration

# Extras
if os.environ.get("MEDUSA_TORCH_INTEGRATION") == "1":
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import random_split, DataLoader, TensorDataset
else:
    raise pytorch_integration.TorchExtrasNotInstalled()


class EEGInceptionV1(components.ProcessingMethod):
    """
    EEG-Inception as described in Santamaría-Vázquez et al. 2020 [1]. This
    model is specifically designed for EEG classification tasks.

    References
    ----------
    [1] Santamaría-Vázquez, E., Martínez-Cagigal, V., Vaquerizo-Villar, F., &
    Hornero, R. (2020). EEG-Inception: A Novel Deep Convolutional Neural Network
    for Assistive ERP-based Brain-Computer Interfaces. IEEE Transactions on
    Neural Systems and Rehabilitation Engineering.
    """

    def __init__(self, input_time=1000, fs=128, n_cha=8, filters_per_branch=8,
                 scales_time=(500, 250, 125), dropout_rate=0.25, n_classes=2,
                 learning_rate=0.001, device_name=None):
        # Super call
        super().__init__(fit=[], predict_proba=['y_pred'])

        # Pytorch config
        self.device = pytorch_integration.config_pytorch(
            device_name=device_name)

        # Attributes
        self.input_time = input_time
        self.fs = fs
        self.n_cha = n_cha
        self.filters_per_branch = filters_per_branch
        self.scales_time = scales_time
        self.dropout_rate = dropout_rate
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.input_samples = int(input_time * fs / 1000)
        self.scales_samples = [int(s * fs / 1000) for s in scales_time]

        # Initialize model
        self.model = self.__PtModel(
            input_samples=self.input_samples,
            n_cha=self.n_cha,
            scales_samples=self.scales_samples,
            filters_per_branch=self.filters_per_branch,
            dropout_rate=self.dropout_rate,
            n_classes=self.n_classes)

    class __PtModel(nn.Module):

        def __init__(self, input_samples, n_cha, scales_samples,
                     filters_per_branch, dropout_rate, n_classes):
            super().__init__()

            self.scales_samples = scales_samples
            self.filters_per_branch = filters_per_branch
            self.dropout_rate = dropout_rate
            self.n_classes = n_classes

            # Block 1: Single-Channel Analysis
            self.block1 = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(1, filters_per_branch, (scale, 1),
                              padding='same'),
                    nn.BatchNorm2d(filters_per_branch),
                    nn.ELU(),
                    nn.Dropout(dropout_rate)
                ) for scale in scales_samples
            ])
            self.pool1 = nn.AvgPool2d((2, 1))

            # Block 2: Spatial Filtering
            self.block2 = nn.Sequential(
                nn.Conv2d(filters_per_branch * len(scales_samples),
                          filters_per_branch * len(scales_samples) * 2,
                          (1, n_cha),
                          groups=filters_per_branch * len(scales_samples),
                          bias=False),
                nn.BatchNorm2d(filters_per_branch * len(scales_samples) * 2),
                nn.ELU(),
                nn.Dropout(dropout_rate),
            )
            self.pool2 = nn.AvgPool2d((2, 1))

            # Block 3: Multi-Channel Analysis
            self.block3 = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(filters_per_branch * len(scales_samples) * 2,
                              filters_per_branch, (scale // 4, 1),
                              padding='same',
                              bias=False),
                    nn.BatchNorm2d(filters_per_branch),
                    nn.ELU(),
                    nn.Dropout(dropout_rate)
                ) for scale in scales_samples
            ])
            self.pool3 = nn.AvgPool2d((2, 1))

            # Block 4: Output Block
            self.block4_1 = nn.Sequential(
                nn.Conv2d(filters_per_branch * len(scales_samples),
                          int(filters_per_branch * len(scales_samples) / 2),
                          (8, 1),
                          padding='same',
                          bias=False),
                nn.BatchNorm2d(
                    int(filters_per_branch * len(scales_samples) / 2)),
                nn.ELU(),
                nn.AvgPool2d((2, 1)),
                nn.Dropout(dropout_rate)
            )

            self.block4_2 = nn.Sequential(
                nn.Conv2d(int(filters_per_branch * len(scales_samples) / 2),
                          int(filters_per_branch * len(scales_samples) / 4),
                          (4, 1),
                          padding='same',
                          bias=False),
                nn.BatchNorm2d(int(filters_per_branch *
                                   len(scales_samples) / 4)),
                nn.ELU(),
                nn.AvgPool2d((2, 1)),
                nn.Dropout(dropout_rate)
            )

            # Output Layer
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(
                int(filters_per_branch * len(scales_samples) / 4) *
                (input_samples // 32), n_classes)

        def forward(self, x):
            # Block 1
            b1_outputs = [unit(x) for unit in self.block1]
            b1_out = torch.cat(b1_outputs, dim=1)
            b1_out = self.pool1(b1_out)

            # Block 2
            b2_out = self.block2(b1_out)
            b2_out = self.pool2(b2_out)

            # Block 3
            b3_outputs = [unit(b2_out) for unit in self.block3]
            b3_out = torch.cat(b3_outputs, dim=1)
            b3_out = self.pool3(b3_out)

            # Block 4
            b4_out = self.block4_1(b3_out)
            b4_out = self.block4_2(b4_out)

            # Output Layer
            out = self.flatten(b4_out)
            out = self.fc(out)

            return F.log_softmax(out, dim=1)

    def transform_data(self, X, y=None):
        # Convert X to numpy array if not already
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        # Transform X
        if len(X.shape) == 3:
            X = np.expand_dims(X, axis=1)
        X = torch.tensor(X, dtype=torch.float32)

        # Check dimensions
        assert len(X.shape) == 4, ("X must be a 4D tensor "
                                   "(n_observ x 1 x n_samples x n_channels)")

        # Transform y if provided
        if y is not None:
            # Convert y to numpy array if not already
            if not isinstance(y, np.ndarray):
                y = np.array(y)

            if len(y.shape) == 1:
                y = np.expand_dims(y, axis=-1)
            if y.shape[1] != self.n_classes:
                y = one_hot_labels(y)
            y = torch.tensor(y, dtype=torch.float32)

            # Check dimensions
            assert len(y.shape) == 2, "y must be a 2D tensor (n_observ x 1)"
            return X, y
        else:
            return X

    def fit(self, X, y, validation_data=None, validation_split=None,
            verbose=True, **kwargs):
        # Check GPU
        if not pytorch_integration.check_gpu_acceleration():
            pytorch_integration.warn_gpu_not_available()
        # Check dimensions and transform data if necessary
        X, y = self.transform_data(X, y)
        # Create dataset and optionally split into training and validation
        dataset = TensorDataset(X, y)
        # Use validation_data if provided
        if validation_data is not None:
            validation = True
            X_val, y_val = validation_data
            X_val, y_val = self.transform_data(X_val, y_val)
            val_dataset = TensorDataset(X_val, y_val)
            train_dataset = dataset
            train_size = len(dataset)
        elif validation_split is not None:
            validation = True
            assert 0 < validation_split < 1, "Validation split must be between 0 and 1."
            val_size = int(len(dataset) * validation_split)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = random_split(dataset,
                                                      [train_size, val_size])
        else:
            validation = False
            train_dataset = dataset
            train_size = len(dataset)
        # Get training info from kwargs
        batch_size = kwargs.get("batch_size", max(train_size // 16, 1024))
        shuffle = kwargs.get("shuffle", True)
        epochs = kwargs.get('epochs', 500)
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle)
        if validation:
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=shuffle)
        # Set optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        # Set early stopping
        early_stopping = EarlyStopping(mode='min',
                                       min_delta=0.005,
                                       patience=10,
                                       verbose=False)
        # Send model to device
        self.model.to(self.device)
        # Training loop
        for n_epoch, epoch in enumerate(range(epochs)):
            # Train phase
            self.model.train()
            train_loss = 0.0
            # Batches
            loop = tqdm(train_loader)
            for n_batch, (inputs, labels) in enumerate(loop):
                # Send data to device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Training phase
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                # Print info
                if n_batch + 1 < len(loop):
                    if verbose:
                        loop.set_description(f'Epoch [{epoch + 1}/{epochs}]')
                        loop.set_postfix(train_loss=train_loss / (n_batch + 1))
                else:
                    if validation:
                        self.model.eval()
                        val_loss = 0.0
                        with torch.no_grad():
                            for inputs, labels in val_loader:
                                # Send data to device
                                inputs = inputs.to(self.device)
                                labels = labels.to(self.device)
                                # Validation phase
                                outputs = self.model(inputs)
                                loss = criterion(outputs, labels)
                                val_loss += loss.item()
                        train_loss /= len(train_loader)
                        val_loss /= len(val_loader)
                        if verbose:
                            loop.set_description(
                                f'Epoch [{epoch + 1}/{epochs}]')
                            loop.set_postfix(train_loss=f'{train_loss:.4f}',
                                             val_loss=f'{val_loss:.4f}')
                    else:
                        train_loss /= len(train_loader)
                        if verbose:
                            loop.set_description(
                                f'Epoch [{epoch + 1}/{epochs}]')
                            loop.set_postfix(train_loss=f'{train_loss:.4f}')
            # Check early stopping
            monitored_loss = val_loss if validation else train_loss
            stop, best_params = early_stopping.check_epoch(
                n_epoch, monitored_loss, self.get_weights())
            if stop:
                self.set_weights(best_params)
                print(f"Early stopping triggered at epoch "
                      f"{n_epoch + 1}. Best weights restored.")
                break

    def predict_proba(self, X):
        # Transform data
        X = self.transform_data(X).to(self.device)
        # Predict
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            return torch.exp(outputs).to('cpu').numpy()

    def to_pickleable_obj(self):
        # Parameters
        kwargs = {
            'input_time': self.input_time,
            'fs': self.fs,
            'n_cha': self.n_cha,
            'filters_per_branch': self.filters_per_branch,
            'scales_time': self.scales_time,
            'dropout_rate': self.dropout_rate,
            'n_classes': self.n_classes,
            'learning_rate': self.learning_rate,
        }
        weights = self.get_weights()
        # Pickleable object
        pickleable_obj = {
            'kwargs': kwargs,
            'weights': weights
        }
        return pickleable_obj

    @classmethod
    def from_pickleable_obj(cls, pickleable_obj):
        pickleable_obj['kwargs']['device_name'] = None
        model = cls(**pickleable_obj['kwargs'])
        model.set_weights(pickleable_obj['weights'])
        return model

    def set_weights(self, weights):
        if not isinstance(weights, dict):
            raise TypeError("Weights must be a state dictionary (OrderedDict).")
        self.model.load_state_dict(weights)

    def get_weights(self):
        return self.model.state_dict()

    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))


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

    def __init__(self, n_classes, n_cha=64, samples=128, dropout_rate=0.5,
                 kern_length=64, F1=8, D=2, F2=16, norm_rate=0.25,
                 dropout_type='Dropout', learning_rate=0.001, device_name=None):

        # Super call
        super().__init__(fit=[], predict_proba=['y_pred'])

        # Pytorch config
        self.device = pytorch_integration.config_pytorch(
            device_name=device_name)

        # Parameters
        self.n_classes = n_classes
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

        # Initialize model
        self.model = self.__PtModel(n_classes=self.n_classes,
                                    n_cha=self.n_cha,
                                    samples=self.samples,
                                    dropout_rate=self.dropout_rate,
                                    kern_length=self.kern_length,
                                    F1=self.F1,
                                    D=self.D,
                                    F2=self.F2,
                                    norm_rate=self.norm_rate,
                                    dropout_type=self.dropout_type)

    class __PtModel(nn.Module):
        """ Pytorch Implementation of EEGNet
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

        Parameters:
              n_classes    : int, number of classes to classify
              n_cha        : int, number of EEG channels
              samples      : int, number of time points in the EEG data
              dropout_rate : float, dropout fraction
              kern_length  : int, length of the temporal convolution kernel in block 1
              F1           : int, number of temporal filters (first conv layer)
              D            : int, number of spatial filters per temporal filter (depth multiplier)
              F2           : int, number of pointwise filters in the separable conv (often F2 = F1 * D)
              norm_rate    : float, maximum norm constraint value (note: not automatically enforced)
              dropout_type : string, either 'SpatialDropout2D' or 'Dropout'
        """

        def __init__(self, n_classes, n_cha, samples, dropout_rate,
                     kern_length,
                     F1, D, F2, norm_rate, dropout_type='Dropout'):

            super().__init__()

            self.n_classes = n_classes
            self.n_cha = n_cha
            self.samples = samples
            self.dropout_rate = dropout_rate
            self.kern_length = kern_length
            self.F1 = F1
            self.D = D
            self.F2 = F2
            self.norm_rate = norm_rate
            self.dropout_type = dropout_type

            # Choose the dropout layer type: Dropout2d mimics SpatialDropout2D
            if dropout_type == 'SpatialDropout2D':
                dropout_layer = nn.Dropout2d
            elif dropout_type == 'Dropout':
                dropout_layer = nn.Dropout
            else:
                raise ValueError(
                    "dropoutType must be one of 'SpatialDropout2D' or 'Dropout'.")

            # ---------------------
            # Block 1: Temporal + Depthwise Convolution
            # ---------------------
            # Input expected shape: (batch, 1, n_cha, samples)
            # 1. Temporal convolution with kernel size (1, kern_length)
            self.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(kern_length, 1),
                padding='same',
                # requires PyTorch 1.10+; otherwise compute appropriate padding
                bias=False
            )
            self.bn1 = nn.BatchNorm2d(F1)

            # 2. Depthwise convolution to learn spatial filters.
            #    Here, groups=F1 makes this a depthwise conv, and we set out_channels = F1 * D.
            self.depthwise = nn.Conv2d(
                in_channels=F1,
                out_channels=F1 * D,
                kernel_size=(1, n_cha),
                bias=False,
                groups=F1
            )
            self.bn2 = nn.BatchNorm2d(F1 * D)
            self.elu1 = nn.ELU()
            self.pool1 = nn.AvgPool2d(kernel_size=(4, 1))
            self.dropout1 = dropout_layer(dropout_rate)

            # ---------------------
            # Block 2: Separable Convolution
            # ---------------------
            # Separable conv = depthwise conv followed by a pointwise conv.
            # The depthwise part:
            self.depthwise2 = nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F1 * D,
                kernel_size=(16, 1),
                padding='same',  # same padding to preserve the time dimension
                groups=F1 * D,
                bias=False
            )
            # The pointwise part:
            self.pointwise2 = nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F2,
                kernel_size=(1, 1),
                bias=False
            )
            self.bn3 = nn.BatchNorm2d(F2)
            self.elu2 = nn.ELU()
            self.pool2 = nn.AvgPool2d(kernel_size=(8, 1))
            self.dropout2 = dropout_layer(dropout_rate)

            # ---------------------
            # Fully Connected Layer
            # ---------------------
            # After Block 1:
            #   - The temporal dimension (width) becomes samples / 4 (via AvgPool2d with kernel (1,4)).
            # After Block 2:
            #   - The temporal dimension becomes (samples/4) / 8 = samples/32.
            # The output of block2 has shape (batch, F2, 1, samples/32), so the flattened
            # feature size is F2 * (samples // 32).
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(in_features=F2 * (samples // 32),
                                out_features=n_classes)

        def forward(self, x):
            # x should have shape: (batch, 1, n_cha, samples)
            # ---------------------
            # Block 1
            # ---------------------
            x = self.conv1(x)  # Shape: (batch, F1, n_cha, samples)
            x = self.bn1(x)
            x = self.depthwise(
                x)  # Convolution across all channels; shape becomes (batch, F1*D, 1, samples)
            x = self.bn2(x)
            x = self.elu1(x)
            x = self.pool1(
                x)  # Reduces the temporal dimension: shape -> (batch, F1*D, 1, samples/4)
            x = self.dropout1(x)

            # ---------------------
            # Block 2
            # ---------------------
            x = self.depthwise2(
                x)  # Depthwise part of separable conv; shape remains (batch, F1*D, 1, samples/4)
            x = self.pointwise2(
                x)  # Pointwise conv; shape: (batch, F2, 1, samples/4)
            x = self.bn3(x)
            x = self.elu2(x)
            x = self.pool2(
                x)  # Further reduces the temporal dimension: (batch, F2, 1, samples/32)
            x = self.dropout2(x)

            # ---------------------
            # Fully Connected Classification Layer
            # ---------------------
            x = self.flatten(x)  # Flatten to shape: (batch, F2 * (samples//32))
            x = self.fc(x)

            return F.log_softmax(x, dim=1)

    def transform_data(self, X, y=None):
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
        # Convert X to numpy array if not already
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        # Transform X
        if len(X.shape) == 3:
            X = np.expand_dims(X, axis=1)
        X = torch.tensor(X, dtype=torch.float32)

        # Check dimensions
        assert len(X.shape) == 4, ("X must be a 4D tensor "
                                   "(n_observ x 1 x n_samples x n_channels)")

        # Transform y if provided
        if y is not None:
            # Convert y to numpy array if not already
            if not isinstance(y, np.ndarray):
                y = np.array(y)

            if len(y.shape) == 1:
                y = np.expand_dims(y, axis=-1)
            if y.shape[1] != self.n_classes:
                y = one_hot_labels(y)
            y = torch.tensor(y, dtype=torch.float32)

            # Check dimensions
            assert len(y.shape) == 2, "y must be a 2D tensor (n_observ x 1)"
            return X, y
        else:
            return X

    def fit(self, X, y, validation_data=None, validation_split=None,
            verbose=True, **kwargs):
        # Check GPU
        if not pytorch_integration.check_gpu_acceleration():
            pytorch_integration.warn_gpu_not_available()
        # Check dimensions and transform data if necessary
        X, y = self.transform_data(X, y)
        # Create dataset and optionally split into training and validation
        dataset = TensorDataset(X, y)
        # Use validation_data if provided
        if validation_data is not None:
            validation = True
            X_val, y_val = validation_data
            X_val, y_val = self.transform_data(X_val, y_val)
            val_dataset = TensorDataset(X_val, y_val)
            train_dataset = dataset
            train_size = len(dataset)
        elif validation_split is not None:
            validation = True
            assert 0 < validation_split < 1, "Validation split must be between 0 and 1."
            val_size = int(len(dataset) * validation_split)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = random_split(dataset,
                                                      [train_size, val_size])
        else:
            validation = False
            train_dataset = dataset
            train_size = len(dataset)
        # Get training info from kwargs
        batch_size = kwargs.get("batch_size", max(train_size // 16, 1024))
        shuffle = kwargs.get("shuffle", True)
        epochs = kwargs.get('epochs', 500)
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle)
        if validation:
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=shuffle)
        # Set optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        # Set early stopping
        early_stopping = EarlyStopping(mode='min',
                                       min_delta=0.005,
                                       patience=10,
                                       verbose=False)
        # Send model to device
        self.model.to(self.device)
        # Training loop
        for n_epoch, epoch in enumerate(range(epochs)):
            # Train phase
            self.model.train()
            train_loss = 0.0
            # Batches
            loop = tqdm(train_loader)
            for n_batch, (inputs, labels) in enumerate(loop):
                # Send data to device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Training phase
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                # Print info
                if n_batch + 1 < len(loop):
                    if verbose:
                        loop.set_description(f'Epoch [{epoch + 1}/{epochs}]')
                        loop.set_postfix(train_loss=train_loss / (n_batch + 1))
                else:
                    if validation:
                        self.model.eval()
                        val_loss = 0.0
                        with torch.no_grad():
                            for inputs, labels in val_loader:
                                # Send data to device
                                inputs = inputs.to(self.device)
                                labels = labels.to(self.device)
                                # Validation phase
                                outputs = self.model(inputs)
                                loss = criterion(outputs, labels)
                                val_loss += loss.item()
                        train_loss /= len(train_loader)
                        val_loss /= len(val_loader)
                        if verbose:
                            loop.set_description(
                                f'Epoch [{epoch + 1}/{epochs}]')
                            loop.set_postfix(train_loss=f'{train_loss:.4f}',
                                             val_loss=f'{val_loss:.4f}')
                    else:
                        train_loss /= len(train_loader)
                        if verbose:
                            loop.set_description(
                                f'Epoch [{epoch + 1}/{epochs}]')
                            loop.set_postfix(train_loss=f'{train_loss:.4f}')
            # Check early stopping
            monitored_loss = val_loss if validation else train_loss
            stop, best_params = early_stopping.check_epoch(
                n_epoch, monitored_loss, self.get_weights())
            if stop:
                self.set_weights(best_params)
                print(f"Early stopping triggered at epoch "
                      f"{n_epoch + 1}. Best weights restored.")
                break

    def predict_proba(self, X):
        # Transform data
        X = self.transform_data(X).to(self.device)
        # Predict
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            return torch.exp(outputs).to('cpu').numpy()

    def to_pickleable_obj(self):
        # Key data
        kwargs = {
            'n_classes': self.n_classes,
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
        weights = self.get_weights()
        # Pickleable object
        pickleable_obj = {
            'kwargs': kwargs,
            'weights': weights
        }
        return pickleable_obj

    @classmethod
    def from_pickleable_obj(cls, pickleable_obj):
        pickleable_obj['kwargs']['device_name'] = None
        model = cls(**pickleable_obj['kwargs'])
        model.set_weights(pickleable_obj['weights'])
        return model

    def set_weights(self, weights):
        if not isinstance(weights, dict):
            raise TypeError("Weights must be a state dictionary (OrderedDict).")
        self.model.load_state_dict(weights)

    def get_weights(self):
        return self.model.state_dict()

    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))


class EEGInceptionV1ICA(components.ProcessingMethod):
    """EEG-Inception as described in Santamaría-Vázquez et al. 2020 [1]. This
    model is specifically designed for MEG classification tasks with additional
    ICA component analysis. --> AMALIA GIL

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
                 device_name=None):

        # Super call
        super().__init__(fit=['X', 'ica', 'y'], predict_proba=['X', 'ica'])

        # Pytorch config
        self.device = pytorch_integration.config_pytorch(
            device_name=device_name)

        # Attributes
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

        # Initialize model
        self.model = self.__PtModel(
            input_samples=self.input_samples,
            n_cha=self.n_cha,
            scales_samples=self.scales_samples,
            filters_per_branch=self.filters_per_branch,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            n_classes=self.n_classes
        )

    class __PtModel(nn.Module):
        def __init__(self, input_samples, n_cha, scales_samples,
                     filters_per_branch, dropout_rate, activation, n_classes):
            super().__init__()

            self.n_cha = n_cha
            self.scales_samples = scales_samples
            self.filters_per_branch = filters_per_branch
            self.dropout_rate = dropout_rate
            self.n_classes = n_classes

            # Set activation function
            if activation == 'elu':
                self.activation = nn.ELU()
            elif activation == 'relu':
                self.activation = nn.ReLU()
            else:
                self.activation = nn.ELU()  # Default to ELU

            # Weight regularization (L2)
            self.weight_decay = 0.01  # Equivalent to L2 regularization in Keras

            total_filters = filters_per_branch * len(scales_samples)

            # BLOCK 1: Single-Channel Analysis
            self.block1 = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(1, filters_per_branch, (scale, 1),
                              padding='same', bias=True),
                    nn.BatchNorm2d(filters_per_branch),
                    nn.ELU(),
                    nn.Dropout(dropout_rate)
                ) for scale in scales_samples
            ])
            self.pool1 = nn.AvgPool2d(kernel_size=(2, 1))

            # BLOCK 2: Spatial Filtering
            self.pool2 = nn.AvgPool2d(kernel_size=(2, 1))

            # BLOCK 3: Multi-Channel Analysis
            self.block3 = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(total_filters, filters_per_branch,
                              kernel_size=(scale // 4, 1),
                              padding='same',
                              bias=False),
                    nn.BatchNorm2d(filters_per_branch),
                    nn.ELU(),
                    nn.Dropout(dropout_rate)
                ) for scale in scales_samples
            ])
            self.pool3 = nn.AvgPool2d(kernel_size=(2, 1))

            # BLOCK 4: Output Block
            self.block4_1 = nn.Sequential(
                nn.Conv2d(total_filters, int(total_filters / 2),
                          kernel_size=(8, 1), padding='same', bias=False),
                nn.BatchNorm2d(int(total_filters / 2)),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(2, 1)),
                nn.Dropout(dropout_rate)
            )
            self.block4_2 = nn.Sequential(
                nn.Conv2d(int(total_filters / 2), int(total_filters / 4),
                          kernel_size=(4, 1), padding='same', bias=False),
                nn.BatchNorm2d(int(total_filters / 4)),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(2, 1)),
                nn.Dropout(dropout_rate)
            )

            # Global pooling layers
            self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
            self.global_max_pool = nn.AdaptiveMaxPool2d(1)

            # Output Layer
            self.flatten = nn.Flatten()
            # Final dimension: EEG features + ICA features
            final_features = int(total_filters / 4) + 160
            self.fc = nn.Linear(final_features, n_classes)

        def forward(self, x, ica):
            # In PyTorch inputs for Conv2d must be B (batch size), C (channels), H (height), W (width)

            # BLOCK 1
            b1_outputs = [unit(x) for unit in self.block1]
            b1_out = torch.cat(b1_outputs, dim=1)
            b1_out = self.pool1(b1_out)

            # BLOCK 2
            b2_out = self.pool2(b1_out)

            # BLOCK 3
            b3_outputs = [unit(b2_out) for unit in self.block3]
            b3_out = torch.cat(b3_outputs, dim=1)
            b3_out = self.pool3(b3_out)

            # BLOCK 4
            b4_out = self.block4_1(b3_out)
            b4_out = self.block4_2(b4_out)

            # Global average pooling for x
            output_eeginception = self.global_avg_pool(b4_out)
            output_eeginception = self.flatten(output_eeginception)

            # ICA feature: Global max pooling
            output_ICA = self.global_max_pool(ica)
            output_ICA = self.flatten(output_ICA)

            # Debug prints
            print(f"Debug - EEG output shape: {output_eeginception.shape}")
            print(f"Debug - ICA output shape: {output_ICA.shape}")

            # Concatenate and classify
            concat = torch.cat([output_eeginception, output_ICA],
                               dim=1)  # axis=-1 in TF = dim=1 in PyTorch
            print(f"Debug - Concat shape: {concat.shape}")
            out = self.fc(concat)

            return F.log_softmax(out, dim=1)

    def transform_data(self, X, y=None, ica=None):

        # Convert X to numpy array if not already and then to torch tensor
        if not isinstance(X, np.ndarray) and not isinstance(X, torch.Tensor):
            X = np.array(X)

        if not isinstance(X, torch.Tensor):
            # X must be: [n_observ, n_channels, n_samples, 1] for Conv2d
            if len(X.shape) == 3:
                # If X is [n_observ, n_samples, n_channels]
                # Reorder [n_observ, n_channels, n_samples, 1]
                X = np.transpose(X,
                                 (0, 2, 1))  # [n_observ, n_channels, n_samples]
                X = np.expand_dims(X,
                                   axis=-1)  # [n_observ, n_channels, n_samples, 1]
            elif len(X.shape) == 4:
                # If it already has 4 dimensions, verify the order
                if X.shape[1] > X.shape[
                    2]:  # Probably [batch, samples, channels, 1]
                    X = np.transpose(X, (
                    0, 2, 1, 3))  # Change to [batch, channels, samples, 1]

            X = torch.tensor(X, dtype=torch.float32)

        # Check dimensions for X
        print(f"Debug - X shape after transformation: {X.shape}")
        assert len(X.shape) == 4, ("X must be a 4D tensor "
                                       "(n_observ x 1 x n_samples x "
                                       "n_channels)")

        # Transform ica if provided
        if ica is not None:
            if not isinstance(ica, np.ndarray) and not isinstance(ica,
                                                                      torch.Tensor):
                ica = np.array(ica)

            if not isinstance(ica, torch.Tensor):
                # Handle different ICA input shapes
                if len(ica.shape) == 1:
                    # Shape: (n_observ,) -> (n_observ, 1, 1, 1)
                    ica = ica.reshape(-1, 1, 1, 1)
                elif len(ica.shape) == 2:
                    # Shape: (n_observ, n_features) -> (n_observ, 1, n_features, 1)
                    ica = ica.reshape(ica.shape[0], 1, ica.shape[1], 1)
                elif len(ica.shape) == 3:
                    # Shape: (n_observ, n_samples, n_features) -> (n_observ, 1, n_samples, n_features)
                    ica = np.expand_dims(ica, axis=1)
                # If already 4D, keep as is

                ica = torch.tensor(ica, dtype=torch.float32)

            # Ensure ICA has 4 dimensions
            if len(ica.shape) == 1:
                ica = ica.reshape(-1, 1, 1, 1)
            elif len(ica.shape) == 2:
                ica = ica.reshape(ica.shape[0], 1, ica.shape[1], 1)
            elif len(ica.shape) == 3:
                ica = ica.unsqueeze(1)

            print(f"Debug - ICA shape after transformation: {ica.shape}")

        # Transform y if provided
        if y is not None:
            print(
                f"Debug - Original y shape: {y.shape if hasattr(y, 'shape') else 'No shape attr'}")
            print(f"Debug - Original y type: {type(y)}")

            if not isinstance(y, np.ndarray) and not isinstance(y,
                                                                torch.Tensor):
                y = np.array(y)

            # Convert to numpy if it's a torch tensor
            if isinstance(y, torch.Tensor):
                y = y.numpy()

            print(f"Debug - After numpy conversion y shape: {y.shape}")
            print(
                f"Debug - Sample y values: {y.flatten()[:10] if y.size > 0 else 'Empty'}")
            print(f"Debug - Unique y values: {np.unique(y.flatten())}")
            print(f"Debug - Y min/max: {np.min(y)}/{np.max(y)}")

            # Ensure y is properly shaped
            if len(y.shape) > 2:
                print(
                    f"Warning: y has more than 2 dimensions ({y.shape}). Flattening...")
                y = y.reshape(y.shape[0], -1)
                if y.shape[1] > 1:
                    # If after reshaping we still have multiple columns, take the first
                    print(f"Warning: Taking only first column after reshape")
                    y = y[:, 0]

            # Make sure y is 1D for label processing
            if len(y.shape) == 2 and y.shape[1] == 1:
                y = y.flatten()
            elif len(y.shape) == 2 and y.shape[1] == self.n_classes:
                # Already one-hot encoded
                print("Labels appear to be already one-hot encoded")
                y = torch.tensor(y, dtype=torch.float32)
                print(f"Debug - Final y shape: {y.shape}")
                return X, ica, y if ica is not None else (X, y)
            elif len(y.shape) == 2:
                print(
                    f"Warning: y has unexpected shape {y.shape}, taking first column")
                y = y[:, 0]

            # Now y should be 1D with class labels
            print(f"Debug - After reshaping y shape: {y.shape}")
            print(f"Debug - After reshaping unique values: {np.unique(y)}")

            # Validate label values
            unique_labels = np.unique(y)
            if len(unique_labels) > self.n_classes:
                raise ValueError(
                    f"Found {len(unique_labels)} unique labels but model expects {self.n_classes} classes")

            max_label = np.max(unique_labels)
            min_label = np.min(unique_labels)

            if max_label >= self.n_classes:
                raise ValueError(
                    f"Max label value {max_label} exceeds expected range [0, {self.n_classes - 1}]")
            if min_label < 0:
                raise ValueError(f"Min label value {min_label} is negative")

            # Convert to one-hot encoding
            n_samples = len(y)
            y_onehot = np.zeros((n_samples, self.n_classes), dtype=np.float32)

            # Safely convert labels to integers and create one-hot
            labels_int = y.astype(int)
            print(
                f"Debug - Creating one-hot for {n_samples} samples with {self.n_classes} classes")
            print(
                f"Debug - Label range: {np.min(labels_int)} to {np.max(labels_int)}")

            for i in range(n_samples):
                label = labels_int[i]
                if 0 <= label < self.n_classes:
                    y_onehot[i, label] = 1.0
                else:
                    raise ValueError(
                        f"Label {label} at index {i} is out of valid range [0, {self.n_classes - 1}]")

            y = torch.tensor(y_onehot, dtype=torch.float32)
            print(f"Debug - Final y shape: {y.shape}")
            print(f"Debug - One-hot sample: {y[0].numpy()}")

            return X, ica, y
        else:
            if ica is not None:
                return X, ica
            else:
                return X

    def fit(self, X, y, ica=None, validation_data=None,
            validation_split=None,
            verbose=True, **kwargs):

        # Check GPU
        if not pytorch_integration.check_gpu_acceleration():
            pytorch_integration.warn_gpu_not_available()

        # Check dimensions and transform data if necessary
        X, ica, y = self.transform_data(X, y, ica)

        # Combine into dataset
        dataset = TensorDataset(X, ica, y)

        # Handle validation data
        if validation_data is not None:
            validation = True
            if len(validation_data) == 3:  # If ICA is included
                X_val, X_ica_val, y_val = validation_data
                X_val, X_ica_val, y_val = self.transform_data(X_val, y_val, X_ica_val)
                val_dataset = TensorDataset(X_val, X_ica_val, y_val)
                # Define train_dataset and train_size here
                train_dataset = dataset
                train_size = len(dataset)
            else:
                raise ValueError(
                    "validation_data must be a tuple of (X_val, "
                    "X_ica_val, y_val)")
                # train_dataset = dataset
                # train_size = len(dataset)

        elif validation_split is not None:
            validation = True
            assert 0 < validation_split < 1, "Validation split must be " \
                                                 "between 0 and 1. "
            val_size = int(len(dataset) * validation_split)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = random_split(dataset, [train_size,
                                                                    val_size])
        else:
            validation = False
            train_dataset = dataset
            train_size = len(dataset)

        # Get training info from kwargs
        fine_tuning = kwargs.get('fine_tuning', False)
        if fine_tuning:
            default_batch_size = 32
            default_epochs = 100
        else:
            default_batch_size = 2048
            default_epochs = 500

        batch_size = kwargs.get("batch_size",
                                    min(default_batch_size, train_size))
        shuffle = kwargs.get("shuffle", True)
        epochs = kwargs.get('epochs', default_epochs)

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle)
        if validation:
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False)

        # Set optimizer with weight decay (L2 regularization)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.model.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        # Set early stopping
        early_stopping = EarlyStopping(
            mode='min',
            min_delta=0.0001 if fine_tuning else 0.001,
            patience=10 if fine_tuning else 20,
            verbose=verbose
        )

        # Send model to device
        self.model.to(self.device)

        # Training loop
        for n_epoch, epoch in enumerate(range(epochs)):
            # Train phase
            self.model.train()
            train_loss = 0.0

            # Progress bar for batches
            loop = tqdm(train_loader) if verbose else train_loader

            for n_batch, (inputs_eeg, inputs_ica, labels) in enumerate(loop):
                # Send data to device
                inputs_eeg = inputs_eeg.to(self.device)
                inputs_ica = inputs_ica.to(self.device)
                labels = labels.to(self.device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs_eeg, inputs_ica)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Track loss
                train_loss += loss.item()

                # Update progress bar
                if verbose and hasattr(loop, 'set_description'):
                    loop.set_description(f'Epoch [{epoch + 1}/{epochs}]')
                    loop.set_postfix(train_loss=train_loss / (n_batch + 1))

            # Validation phase
            if validation:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs_eeg, inputs_ica, labels in val_loader:
                        # Send data to device
                        inputs_eeg = inputs_eeg.to(self.device)
                        inputs_ica = inputs_ica.to(self.device)
                        labels = labels.to(self.device)

                        # Validation step
                        outputs = self.model(inputs_eeg, inputs_ica)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()

                # Calculate average losses
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)

                if verbose:
                    print(f'Epoch [{epoch + 1}/{epochs}] - '
                          f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

                # Check early stopping
                stop, best_params = early_stopping.check_epoch(
                    n_epoch, val_loss, self.get_weights())
                if stop:
                    self.set_weights(best_params)
                    if verbose:
                        print(f"Early stopping triggered at epoch "
                              f"{n_epoch + 1}. Best weights restored.")
                    break
            else:
                # No validation, just track training loss
                train_loss /= len(train_loader)
                if verbose:
                    print(
                        f'Epoch [{epoch + 1}/{epochs}] - Train Loss: {train_loss:.4f}')

                # Check early stopping on training loss
                stop, best_params = early_stopping.check_epoch(
                    n_epoch, train_loss, self.get_weights())
                if stop:
                    self.set_weights(best_params)
                    if verbose:
                        print(f"Early stopping triggered at epoch "
                              f"{n_epoch + 1}. Best weights restored.")
                    break

    def predict_proba(self, X, ica):
        X, ica = self.transform_data(X, ica=ica)
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X, ica)
            return torch.exp(outputs).cpu().numpy()

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
        weights = self.get_weights()
        # Pickleable object
        pickleable_obj = {
            'kwargs': kwargs,
            'weights': weights
        }
        return pickleable_obj

    @classmethod
    def from_pickleable_obj(cls, pickleable_obj):
        pickleable_obj['kwargs']['device_name'] = None
        model = cls(**pickleable_obj['kwargs'])
        model.set_weights(pickleable_obj['weights'])
        return model

    def set_weights(self, weights):
        if not isinstance(weights, dict):
            raise TypeError(
                "Weights must be a state dictionary (OrderedDict).")
        self.model.load_state_dict(weights)

    def get_weights(self):
        return self.model.state_dict()

    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))

    def summary(self, input_shape=None):
        """Print model summary"""
        print("EEG-Inception Model Summary")
        print("=" * 50)
        print(f"Input time: {self.input_time} ms")
        print(f"Sampling frequency: {self.fs} Hz")
        print(f"Number of channels: {self.n_cha}")
        print(f"Input samples: {self.input_samples}")
        print(f"Filters per branch: {self.filters_per_branch}")
        print(f"Scales (samples): {self.scales_samples}")
        print(f"Dropout rate: {self.dropout_rate}")
        print(f"Activation: {self.activation}")
        print(f"Number of classes: {self.n_classes}")
        print(f"Learning rate: {self.learning_rate}")
        print("=" * 50)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("=" * 50)

        # Print model architecture
        print("\nModel Architecture:")
        print(self.model)

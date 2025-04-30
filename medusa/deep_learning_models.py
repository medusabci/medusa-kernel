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
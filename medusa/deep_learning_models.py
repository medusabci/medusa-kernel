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
                    nn.Conv2d(1, filters_per_branch, (scale, 1), padding='same'),
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
                          int(filters_per_branch *
                              len(scales_samples) / 2), (8, 1),
                          padding='same',
                          bias=False),
                nn.BatchNorm2d(int(filters_per_branch *
                                   len(scales_samples) / 2)),
                nn.ELU(),
                nn.AvgPool2d((2, 1)),
                nn.Dropout(dropout_rate)
            )

            self.block4_2 = nn.Sequential(
                nn.Conv2d(int(filters_per_branch * len(scales_samples) / 2),
                          int(filters_per_branch *
                              len(scales_samples) / 4), (4, 1),
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

    def fit(self, X, y, validation_split=None, verbose=True, **kwargs):
        # Check GPU
        if not pytorch_integration.check_gpu_acceleration():
            pytorch_integration.warn_gpu_not_available()
        # Check dimensions and transform data if necessary
        X, y = self.transform_data(X, y)
        # Create dataset and optionally split into training and validation
        dataset = TensorDataset(X, y)
        if validation_split is not None:
            validation = True
            assert 0 < validation_split < 1, ("Validation split must be "
                                              "between 0 and 1.")
            val_size = int(len(dataset) * validation_split)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = random_split(
                dataset, [train_size, val_size])
        else:
            validation = False
            train_size = len(dataset)
            train_dataset = dataset
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
                if n_batch+1 < len(loop):
                    if verbose:
                        loop.set_description(f'Epoch [{epoch+1}/{epochs}]')
                        loop.set_postfix(train_loss=train_loss/(n_batch+1))
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
                            loop.set_description(f'Epoch [{epoch + 1}/{epochs}]')
                            loop.set_postfix(train_loss=f'{train_loss:.4f}',
                                             val_loss=f'{val_loss:.4f}')
                    else:
                        train_loss /= len(train_loader)
                        if verbose:
                            loop.set_description(f'Epoch [{epoch + 1}/{epochs}]')
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

    def predict_proba(self, X, device_type='cuda'):
        # Transform data
        X = self.transform_data(X).to(self.device)
        # Predict
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            return torch.exp(outputs).to('cpu').numpy()

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


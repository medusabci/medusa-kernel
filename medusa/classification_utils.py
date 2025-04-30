from sklearn.preprocessing import OneHotEncoder
import numpy as np


def categorical_labels(one_hot_labels):
    cat_labels = np.argmax(one_hot_labels, axis=1)
    return cat_labels


def one_hot_labels(categorical_labels):
    enc = OneHotEncoder(handle_unknown='ignore')
    one_hot_labels = \
        enc.fit_transform(categorical_labels.reshape(-1, 1)).toarray()

    return one_hot_labels


def k_fold_split(x, y, k, keys=None, shuffle=False):
    """
    Special implementation of k fold splitting that allows to split the dataset
    into k folds for cross validation in function of keys array.

    It returns a list with the dataset for each iteration (k iterations).

    Parameters
    ----------
    x: numpy array or list
        Training set data. Axis 0 represents each observation. Features could
        have one or more dimensions. For instance, [observations x eeg samples],
        [observations x eeg samples x channels]
    y: numpy array or list
        Training set labels.
    k: int
        Number of folds to split the dataset
    keys: numpy array or list
        Keys to split the dataset. If None, the dataset is splitted considering
        each observation independently. If not None, each position of keys
        array identifies the set that owns the observation. For instance, This
        is useful to split the dataset by subjects or trials.
    shuffle: boolean
        True if you want to shuffle the dataset randomly.

    Returns
    -------
    sets: list
        List that contains a dict with the train and test set for each iteration
        of the k-fold algorithm.

    Examples
    --------
    >>> k_fold_iter = k_fold_split(x, y, k)
    >>> k_fold_acc = 0
    >>> for iter in k_fold_iter:
    >>>     model.fit(iter["x_train"], iter["y_train"])
    >>>     y_test_pred = model.predict(iter["x_test"], iter["y_test"])
    >>>     k_fold_acc += np.sum(y_test_pred == iter["y_test"])/len(iter["y_test"])
    >>> k_fold_acc = k_fold_acc/len(k_fold_iter)

    """
    # Convert to numpy arrays
    x = np.array(x)
    y = np.array(y)
    # If keys is None, each observation is treated independently
    if keys is None:
        keys = np.arange(len(x))
    else:
        keys = np.array(keys)
    if keys.shape[0] != x.shape[0] or keys.shape[0] != y.shape[0]:
        raise ValueError("Dimensions of x, y and keys arrays must match along"
                         " axis 0.")
    # Divide keys array in k folds
    keys_values = np.unique(keys)
    if shuffle:
        np.random.shuffle(keys_values)
    keys_folds = np.array_split(keys_values, k)
    # Divide the dataset
    k_fold_iter = list()
    for i in range(k):
        idx = np.isin(keys, keys_folds[i]).nonzero()
        # Get train set
        x_train = np.delete(x, idx, axis=0)
        y_train = np.delete(y, idx, axis=0)
        # Get test set
        x_test = x[idx]
        y_test = y[idx]
        # Save train and test sets of iteration i
        split = dict()
        split["x_train"] = x_train
        split["y_train"] = y_train
        split["x_test"] = x_test
        split["y_test"] = y_test
        k_fold_iter.append(split)
    return k_fold_iter


class EarlyStopping:
    """
    Implements early stopping to terminate training when a monitored metric
    stops improving.

    Parameters
    ----------
    mode : {'min', 'max'}, optional
        Determines whether the monitored metric should be minimized or
        maximized.
        - 'min' (default): Training stops when the metric does not decrease.
        - 'max': Training stops when the metric does not increase.
    min_delta : float, optional
        The minimum change in the monitored metric to qualify as an improvement.
        Defaults to 0.001.
    patience : int, optional
        Number of epochs to wait after the last improvement before stopping
        training. Defaults to 20.
    verbose : bool, optional
        If True, prints messages when the best metric is updated or when
        patience runs out. Defaults to True.
    """
    def __init__(self, mode='min', min_delta=0.001, patience=20, verbose=True):
        # Init attributes
        self.mode = mode
        self.min_delta = min_delta
        self.patience=patience
        self.verbose = verbose
        # Init states
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.best_params = None
        self.patience_counter = 0

    def check_epoch(self, n_epoch, epoch_loss, epoch_params=None):
        """
        Checks whether training should stop based on the given epoch's loss.

        Parameters
        ----------
        n_epoch : int
            The current epoch number.
        epoch_loss : float
            The loss value for the current epoch.
        epoch_params : dict, optional
            The parameters at the current epoch (e.g., model state dictionary).

        Returns
        -------
        bool
            True if training should stop, False otherwise.
        dict or None
            The best parameters recorded during training, or None if no
            improvement was found.
        """
        # Check if updates are needed
        if self.mode == 'min':
            update_params = epoch_loss < self.best_loss
            update_state = epoch_loss < self.best_loss - self.min_delta
        elif self.mode == 'max':
            update_params = epoch_loss > self.best_loss
            update_state = epoch_loss > self.best_loss + self.min_delta
        else:
            raise ValueError('Mode must be min or max')
        # Update state
        if update_state:
            self.best_loss = epoch_loss
            self.best_epoch = n_epoch
            self.patience_counter = 0
            if self.verbose:
                print(f"\nEarly stopping: New best loss {self.best_loss:.4f} "
                      f"at epoch {n_epoch+1}. Resetting patience.")
        else:
            self.patience_counter += 1
       # Update params
        if update_params:
            self.best_params = epoch_params
        # Check patience
        if self.patience_counter >= self.patience:
            return True, self.best_params
        else:
            return False, self.best_params


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
    >>>     model.fit(iter[x_train, y_train])
    >>>     y_test_pred = model.predict(iter[x_test, y_test])
    >>>     k_fold_acc += np.sum(y_test_pred == y_test)/len(y_test)
    >>>     k_fold_acc = k_fold_acc/len(k_fold_iter)
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


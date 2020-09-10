import numpy as np
import os
import shutil
import scipy.io as sio
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from imblearn import under_sampling, over_sampling


def loadMetadata(filename, silent = False):
    '''
    Loads matlab mat file and formats it for simple use.
    '''
    try:
        # http://stackoverflow.com/questions/6273634/access-array-contents-from-a-mat-file-loaded-using-scipy-io-loadmat-python
        if not silent:
            print('\tReading metadata from %s...' % filename)
        #metadata = sio.loadmat(filename, squeeze_me=True, struct_as_record=False)
        metadata = MatReader().loadmat(filename)
    except:
        print('\tFailed to read the meta file "%s"!' % filename)
        return None
    return metadata


def preparePath(path, clear = False):
    if not os.path.isdir(path):
        os.makedirs(path, 0o777)
    if clear:
        files = os.listdir(path)
        for f in files:
            fPath = os.path.join(path, f)
            if os.path.isdir(fPath):
                shutil.rmtree(fPath)
            else:
                os.remove(fPath)

    return path


class MatReader(object):
    '''
    Loads matlab mat file and formats it for simple use.
    '''
    def __init__(self, flatten1D = True):
        self.flatten1D = flatten1D


    def loadmat(self, filename):
        meta = sio.loadmat(filename, struct_as_record=False) 
        
        meta.pop('__header__', None)
        meta.pop('__version__', None)
        meta.pop('__globals__', None)

        meta = self._squeezeItem(meta)
        return meta


    def _squeezeItem(self, item):
        if isinstance(item, np.ndarray):            
            if item.dtype == np.object:
                if item.size == 1:
                    item = item[0,0]
                else:
                    item = item.squeeze()
            elif item.dtype.type is np.str_:
                item = str(item.squeeze())
            elif (self.flatten1D and len(item.shape) == 2
                  and (item.shape[0] == 1 or item.shape[1] == 1)):
                #import pdb; pdb.set_trace()
                item = item.flatten()
            
            if isinstance(item, np.ndarray) and item.dtype == np.object:
                #import pdb; pdb.set_trace()
                #for v in np.nditer(item, flags=['refs_ok'], op_flags=['readwrite']):
                #    v[...] = self._squeezeItem(v)
                it = np.nditer(item, flags=['multi_index','refs_ok'],
                               op_flags=['readwrite'])
                while not it.finished:
                    item[it.multi_index] = self._squeezeItem(item[it.multi_index])
                    it.iternext()


        if isinstance(item, dict):
            for k,v in item.items():
                item[k] = self._squeezeItem(v)
        elif isinstance(item, sio.matlab.mio5_params.mat_struct):
            for k in item._fieldnames:
                v = getattr(item, k)
                setattr(item, k, self._squeezeItem(v))
                 
        return item
    

def load_data(filename, kfold=3, seed=333, split='random'):
    """
    Function to load the pressure data with stratified k fold

    Parameters
    ----------
    filename : string
        Path to the data file.
    augment : bool, optional
        Whether or not to add random noise to the pressure data.
        The default is False.
    kFold : int, optional
        Number of folds to use in stratified k fold. The default is 3.
    seed : int, optional
        Seed used for shuffling the train test split and stratified k fold.
        The default is None.

    Returns
    -------
    train_data : numpy array
        DESCRIPTION.
    train_labels : numpy array
        DESCRIPTION.
    train_ind : numpy array
        DESCRIPTION.
    val_ind : numpy array
        DESCRIPTION.
    test_data : numpy array
        DESCRIPTION.
    test_labels : numpy array
        DESCRIPTION.
    """

    data = sio.loadmat(filename)
    valid_idx = data['hasValidLabel'].flatten() == 1
    balanced_idx = data['isBalanced'].flatten() == 1
    # indices now gives a subset of the data set that contains only valid
    # pressure frames and the same number of frames for each class
    indices = np.logical_and(valid_idx, balanced_idx)
    pressure = np.transpose(data['pressure'], axes=(0, 2, 1))
    # Prepare the data the same way as in the paper
    pressure = np.clip((pressure.astype(np.float32)-500)/(650-500), 0.0, 1.0)
    # Reshape the data into 1D feature vectors for sklearn utility functions
    pressure = np.reshape(pressure, (-1, 32*32))
    object_id = data['objectId'].flatten()
    
    if split == 'original':
        # Find the samples that were used for training in the paper
        split_idx = data['splitId'].flatten() == 0
        train_indices = np.logical_and(indices, split_idx)
        pressure_train = pressure[train_indices]

        train_data = pressure_train
        train_labels = object_id[train_indices]

        # Find the samples that were used for testing in the paper
        split_idx = data['splitId'].flatten() == 1
        test_indices = np.logical_and(indices, split_idx)
        pressure_test = pressure[test_indices]

        test_data = pressure_test
        test_labels = object_id[test_indices]

        # # Add the rest of the valid data to the test set
        # unbalanced_idx = np.logical_xor(valid_idx, balanced_idx)
        # rest_pressure = pressure[unbalanced_idx]
        # rest_object_id = object_id[unbalanced_idx]
        # test_data = np.append(test_data, rest_pressure, axis=0)
        # test_labels = np.append(test_labels, rest_object_id, axis=0)
        
        #_____________________________________________________________________#
        # # Just to test if the accuracy in the test set itself stays high
        # train_data, test_data,\
        #     train_labels, test_labels = train_test_split(pressure_train,
        #                                                  train_labels,
        #                                                  test_size=0.2,
        #                                                  random_state=seed,
        #                                                  shuffle=True,
        #                                                  stratify=train_labels)
        #_____________________________________________________________________#

        return train_data, train_labels, test_data, test_labels

    elif split == 'random':
        # # Add the rest of the valid data to the test set
        # unbalanced_idx = np.logical_xor(valid_idx, balanced_idx)
        # rest_pressure = pressure[unbalanced_idx]
        # rest_object_id = object_id[unbalanced_idx]

        pressure = pressure[indices]
        object_id = object_id[indices]

        if kfold is not None:
            # Decrease the test size if cross validation is used
            test_size = 0.15
        else:
            test_size = 0.306

        # Split the already balanced dataset in a stratified way -> training
        # and test set will still be balanced
        train_data, test_data,\
            train_labels, test_labels = train_test_split(pressure, object_id,
                                                         test_size=test_size,
                                                         random_state=seed,
                                                         shuffle=True,
                                                         stratify=object_id)
        #print(train_data.shape, train_labels.shape)
        # This generates a k fold split in a stratified way.
        # Easy way to do k fold cross validation
        skf = StratifiedKFold(n_splits=kfold, shuffle=True,
                              random_state=seed+1)
        # train_ind, val_ind = skf.split(train_data, train_labels)
        skf_gen = skf.split(train_data, train_labels)
        
        # # Add the rest of the valid data to the test set
        # test_data = np.append(test_data, rest_pressure, axis=0)
        # test_labels = np.append(test_labels, rest_object_id, axis=0)
    
        return train_data, train_labels, test_data, test_labels, skf_gen

    elif split == 'recording':
        # Each class has three recording IDs that correspond to the different
        # experiment days. There are 81 recording IDs (3*27)
        # 0  - 26 belong to the first recording
        # 27 - 53 belong to the second recording
        # 54 - 81 belong to the third recording
        recording_id = data['recordingId'].flatten()
        recordings = []
        for i in range(3):
            # Find valid samples from the different recording days
            recording_mask = np.logical_and(recording_id >= i*27,
                                            recording_id < (i+1)*27)
            recording_mask = np.logical_and(recording_mask, valid_idx)
            
            # The data is not yet balanced!
            recordings.append([pressure[recording_mask],
                               object_id[recording_mask]])
            
        x1, y1 = recordings[0][0], recordings[0][1]
        x2, y2 = recordings[1][0], recordings[1][1]
        x3, y3 = recordings[2][0], recordings[2][1]
        
        # Balance data using the python package 'imbalanced-learn'
        # Random undersampling.
        undersampler = under_sampling.RandomUnderSampler(random_state=seed+2,
                                                         sampling_strategy='not minority')
        
        # KMeansSMOTE oversampling. This generates NEW samples!
        # Can be seen as data augmentation. kmeans_estimator tells the sampler
        # how many clusters to generate
        oversampler = over_sampling.KMeansSMOTE(random_state=seed+2,
                                                kmeans_estimator=14)
        
        # # First undersample the majority class
        # x1_resampled, y1_resampled = undersampler.fit_resample(x1, y1)
        # x2_resampled, y2_resampled = undersampler.fit_resample(x2, y2)
        # x3_resampled, y3_resampled = undersampler.fit_resample(x3, y3)
        
        # # Then oversample the rest of the classes such that the set is balanced
        # x1_resampled, y1_resampled = oversampler.fit_resample(x1_resampled,
        #                                                       y1_resampled)    
        # x2_resampled, y2_resampled = oversampler.fit_resample(x2_resampled,
        #                                                       y2_resampled)
        # x3_resampled, y3_resampled = oversampler.fit_resample(x3_resampled,
        #                                                       y3_resampled)
        
        # Try to oversample without undersampling class 0 first
        x1_resampled, y1_resampled = oversampler.fit_resample(x1, y1)    
        x2_resampled, y2_resampled = oversampler.fit_resample(x2, y2)
        x3_resampled, y3_resampled = oversampler.fit_resample(x3, y3)

        return x1_resampled, y1_resampled, x2_resampled, y2_resampled,\
            x3_resampled, y3_resampled
    
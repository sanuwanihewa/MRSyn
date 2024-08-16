import scipy
import torch.utils.data
import numpy as np
import random
import torchio as tio


def CreateDatasetSynthesis(phase, input_path):
    input_path = '/data/shew0029/MedSyn/DATA/ISLES/'
    

    target_file = input_path + phase + "/t2.npy"
    data_fs_s1 = LoadDataSet(target_file)



    target_file = input_path + phase + "/flair.npy"
    data_fs_s2 = LoadDataSet(target_file)

    # dataset=torch.utils.data.TensorDataset(torch.from_numpy(data_fs_s1),torch.from_numpy(data_fs_s2))
    dataset = torch.utils.data.TensorDataset(  torch.from_numpy(data_fs_s1), torch.from_numpy(data_fs_s2))
    return dataset




def LoadDataSet(load_dir, variable='slices', padding=True, Norm=True):

    # Load the Numpy array
    data=np.load(load_dir)

    # Transpose and expand dimensions if necessary
    if data.ndim == 3:
        data = np.expand_dims(np.transpose(data, (0, 2, 1)), axis=1)
    else:
        data = np.transpose(data, (1, 0, 3, 2))

    data = data.astype(np.float32)

    if padding:
        pad_x = int((256 - data.shape[2]) / 2)
        pad_y = int((256 - data.shape[3]) / 2)
        print('padding in x-y with:' + str(pad_x) + '-' + str(pad_y))
        data = np.pad(data, ((0, 0), (0, 0), (pad_x, pad_x), (pad_y, pad_y)))

    if Norm:
        data = (data - 0.5) / 0.5

    return data
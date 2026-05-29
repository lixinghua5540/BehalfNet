import os
import cv2
import torch
import skimage
import numpy as np
import torch.utils.data
import sklearn.model_selection

def load_mat_hsi(dataset_name, dataset_dir):
    """ load HSI.mat dataset """
    # available sets
    available_sets = [
        'T1',
        'T2',
        'Bi-Temporal-cat',
        'Bi-Temporal-split',

        'T1_V',
        'T2_V',
        'Bi-Temporal_V-cat',
        'Bi-Temporal_V-split',
    ]
    assert dataset_name in available_sets, "dataset should be one of" + ' ' + str(available_sets)

    image_t1 = image_t2 = None
    gt = skimage.io.imread(os.path.join(dataset_dir, "label.tif"))

    # Viareggio
    if dataset_name.lower().endswith("V"):
        labels = [
            'Undefined',
            'Gym',
            'Suburbs',
            'Pine trees',
            'Parking lot',
            'Warehouse',
            'Public pool',
            'Football field',
            'Helicopter landing area',
        ]

    # Anji
    else:
        labels = [
            'Undefined',
            'Corn',
            'Tea',
            'Potato',
            'Wheat',
            'Yam',
            'Grassland',
            'Broad-leaved forest',
            'Needle-leaved forest',
            'Mixed forests',
            'Building',
            'Road',
            'Water',
        ]

    # Anji
    # Two branch inputs T1
    if dataset_name == 'T1':
        image_t1 = image_t2 = skimage.io.imread(os.path.join(dataset_dir, "Time1.tif"))

    # Two branch inputs T2
    if dataset_name == 'T2':
        image_t1 = image_t2 =  skimage.io.imread(os.path.join(dataset_dir, "Time2.tif"))

    # Two branch inputs (Time1 cat Time2)
    if dataset_name == 'Bi-Temporal-cat':
        image_1 = skimage.io.imread(os.path.join(dataset_dir, "Time1.tif"))
        image_2 = skimage.io.imread(os.path.join(dataset_dir, "Time2.tif"))
        image_t1 = image_t2 = np.concatenate((image_1, image_2), axis=2)

    # Enter Time1 for one and Time2 for the other.
    if dataset_name == 'Bi-Temporal-split':
        image_t1 = skimage.io.imread(os.path.join(dataset_dir, "Time1.tif"))
        image_t2 = skimage.io.imread(os.path.join(dataset_dir, "Time2.tif"))

    # Viareggio
    # Two branch inputs T1
    if dataset_name == 'T1_V':
        image_t1 = image_t2 = skimage.io.imread(os.path.join(dataset_dir, "Time1.tif"))

    if dataset_name == 'T2_V':
        image_t1 = image_t2 =  skimage.io.imread(os.path.join(dataset_dir, "Time2.tif"))

    if dataset_name == 'Bi-Temporal_V-cat':
        image_1 = skimage.io.imread(os.path.join(dataset_dir, "Time1.tif"))
        image_2 = skimage.io.imread(os.path.join(dataset_dir, "Time2.tif"))
        image_t1 = image_t2 = np.concatenate((image_1, image_2), axis=2)

    if dataset_name == 'Bi-Temporal_V-split':
        image_t1 = skimage.io.imread(os.path.join(dataset_dir, "Time1.tif"))
        image_t2 = skimage.io.imread(os.path.join(dataset_dir, "Time2.tif"))

    # after getting image and ground truth (gt), let us do data preprocessing!
    # step1 set undefined index 0 to -1, so class index starts from 0
    gt = gt.astype('int') - 1

    # step2 remove undefined label
    labels = labels[1:]

    # step3 normalise the HSI data (method from SSAN, TGRS 2020)
    image_t1 = np.asarray(image_t1, dtype=np.float32)
    image_t2 = np.asarray(image_t2, dtype=np.float32)
    image_t1 = (image_t1 - np.min(image_t1)) / (np.max(image_t1) - np.min(image_t1))
    image_t2 = (image_t2 - np.min(image_t2)) / (np.max(image_t2) - np.min(image_t2))
    mean_by_c1 = np.mean(image_t1, axis=(0, 1))
    mean_by_c2 = np.mean(image_t2, axis=(0, 1))
    for c in range(image_t1.shape[-1]):
        image_t1[:, :, c] = image_t1[:, :, c] - mean_by_c1[c]

    for c in range(image_t2.shape[-1]):
        image_t2[:, :, c] = image_t2[:, :, c] - mean_by_c2[c]

    return image_t1, image_t2, gt, labels


def sample_gt(gt, percentage, seed):
    """
    :param gt: 2d int array, -1 for undefined or not selected, index starts at 0
    :param percentage: for example, 0.1 for 10%, 0.02 for 2%, 0.5 for 50%
    :param seed: random seed
    :return:
    """
    indices = np.where(gt >= 0)
    X = list(zip(*indices))
    y = gt[indices].ravel()

    train_gt = np.full_like(gt, fill_value=-1)
    test_gt = np.full_like(gt, fill_value=-1)

    train_indices, test_indices = sklearn.model_selection.train_test_split(
        X,
        train_size=percentage,
        random_state=seed,
        stratify=y
    )

    train_indices = [list(t) for t in zip(*train_indices)]
    test_indices = [list(t) for t in zip(*test_indices)]

    train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
    test_gt[tuple(test_indices)] = gt[tuple(test_indices)]

    return train_gt, test_gt


class HSIDataset(torch.utils.data.Dataset):
    def __init__(self, images, gt, patch_size, data_aug=True):
        """
        :param images: tuple of 3d float np arrays of HSI, (image_t1, image_t2)
        :param gt: train_gt or val_gt or test_gt
        :param patch_size: 7 or 9 or 11 ...
        :param data_aug: whether to use data augment, default is True
        """
        super().__init__()
        self.data_aug = data_aug
        self.patch_size = patch_size
        self.ps = self.patch_size // 2  # padding size

        # Unpack the two images
        image_t1, image_t2 = images

        # Pad both images
        self.data_t1 = np.pad(image_t1, ((self.ps, self.ps), (self.ps, self.ps), (0, 0)), mode='reflect')
        self.data_t2 = np.pad(image_t2, ((self.ps, self.ps), (self.ps, self.ps), (0, 0)), mode='reflect')
        self.label = np.pad(gt, ((self.ps, self.ps), (self.ps, self.ps)), mode='reflect')

        mask = np.ones_like(self.label)
        mask[self.label < 0] = 0
        x_pos, y_pos = np.nonzero(mask)

        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)
                                 if self.ps <= x < image_t1.shape[0] + self.ps
                                 and self.ps <= y < image_t1.shape[1] + self.ps])
        self.labels = [self.label[x, y] for x, y in self.indices]
        np.random.shuffle(self.indices)

    def hsi_augment(self, data):
        # e.g. (7 7 200) data = numpy array float32
        do_augment = np.random.random()
        if do_augment > 0.5:
            prob = np.random.random()
            if 0 <= prob <= 0.2:
                data = np.fliplr(data)
            elif 0.2 < prob <= 0.4:
                data = np.flipud(data)
            elif 0.4 < prob <= 0.6:
                data = np.rot90(data, k=1)
            elif 0.6 < prob <= 0.8:
                data = np.rot90(data, k=2)
            elif 0.8 < prob <= 1.0:
                data = np.rot90(data, k=3)
        return data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        # Extract patches from both images
        data_t1 = self.data_t1[x1:x2, y1:y2]
        data_t2 = self.data_t2[x1:x2, y1:y2]
        label = self.label[x, y]

        if self.data_aug:
            # Perform data augmentation (only on 2D patches)
            data_t1 = self.hsi_augment(data_t1)
            data_t2 = self.hsi_augment(data_t2)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data_t1 = np.asarray(np.copy(data_t1).transpose((2, 0, 1)), dtype='float32')
        data_t2 = np.asarray(np.copy(data_t2).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data_t1 = torch.from_numpy(data_t1)
        data_t2 = torch.from_numpy(data_t2)
        label = torch.from_numpy(label)

        # Add a fourth dimension for 3D CNN
        data_t1 = data_t1.unsqueeze(0)
        data_t2 = data_t2.unsqueeze(0)

        return (data_t1, data_t2), label

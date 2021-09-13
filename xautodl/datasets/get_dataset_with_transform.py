##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import os, sys, torch
import os.path as osp
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
from copy import deepcopy
from PIL import Image

from xautodl.config_utils import load_config

from .DownsampledImageNet import ImageNet16
from .SearchDatasetWrap import SearchDataset
from torch.utils.data import DataLoader, Dataset
import random



Dataset2Class = {
    "cifar10": 10,
    "cifar100": 100,
    "imagenet-1k-s": 1000,
    "imagenet-1k": 1000,
    "ImageNet16": 1000,
    "ImageNet16-150": 150,
    "ImageNet16-120": 120,
    "ImageNet16-200": 200,
}


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)

def randomSplit(M, N, minV, maxV):
    res = []
    while N > 0:
        l = max(minV, M - (N-1)*maxV)
        r = min(maxV, M - (N-1)*minV)
        num = random.randint(l, r)
        N -= 1
        M -= num
        res.append(num)
    print(res)
    return res

def uniform(N, k):
    """Uniform distribution of 'N' items into 'k' groups."""
    dist = []
    avg = N / k
    # Make distribution
    for i in range(k):
        dist.append(int((i + 1) * avg) - int(i * avg))
    # Return shuffled distribution
    random.shuffle(dist)
    return dist

def normal(N, k):
    """Normal distribution of 'N' items into 'k' groups."""
    dist = []
    # Make distribution
    for i in range(k):
        x = i - (k - 1) / 2
        dist.append(int(N * (np.exp(-x) / (np.exp(-x) + 1)**2)))
    # Add remainders
    remainder = N - sum(dist)
    dist = list(np.add(dist, uniform(remainder, k)))
    # Return non-shuffled distribution
    return dist


def data_organize(idxs_labels, labels):
    data_dict = {}

    labels = np.unique(labels, axis=0)
    for one in labels:
        data_dict[one] = []

    for i in range(len(idxs_labels[1, :])):
        data_dict[idxs_labels[1, i]].append(idxs_labels[0, i])
    return data_dict


def data_partition(training_data, client_dict, non_iid_level):

    number_of_clients = len(client_dict)
    idxs = np.arange(len(training_data))
    labels = training_data.targets

    # sort labels
    idxs_labels = np.vstack((idxs, labels))

    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    labels = np.unique(labels, axis=0)
    idxs = idxs_labels[0, :]
    data_dict = data_organize(idxs_labels, labels)

    if non_iid_level == 0:
        num_items = int(len(training_data)/number_of_clients)
        data_partition_profile, all_idxs = {}, [i for i in range(len(training_data))]
        for i in range(number_of_clients):
            data_partition_profile[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - data_partition_profile[i])

    else:
        data_dist = uniform(len(training_data), number_of_clients)
        data_dist.sort(reverse=True)
        print(data_dist)

        data_partition_profile, all_idxs = {}, [i for i in range(len(training_data))]

        for i in range(number_of_clients):
            pref_number = int(round(data_dist[i] * non_iid_level))

            if pref_number > len(data_dict[client_dict[i]]):
                pref_number = len(data_dict[client_dict[i]])

            data_dist[i] -= pref_number
            data_partition_profile[i] = set(np.random.choice(data_dict[client_dict[i]], pref_number, replace=False))
            all_idxs = list(set(all_idxs) - data_partition_profile[i])
            data_dict[client_dict[i]] = list(set(data_dict[client_dict[i]]) - data_partition_profile[i])

        for i in range(number_of_clients):
            rest_idxs = set(np.random.choice(all_idxs, data_dist[i], replace=False))
            data_partition_profile[i] = set.union(data_partition_profile[i], rest_idxs)
            all_idxs = list(set(all_idxs) - rest_idxs)

        for one in data_partition_profile:
            data_partition_profile[one] = list(data_partition_profile[one])

    return data_partition_profile


class CUTOUT(object):
    def __init__(self, length):
        self.length = length

    def __repr__(self):
        return "{name}(length={length})".format(
            name=self.__class__.__name__, **self.__dict__
        )

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


imagenet_pca = {
    "eigval": np.asarray([0.2175, 0.0188, 0.0045]),
    "eigvec": np.asarray(
        [
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ]
    ),
}


class Lighting(object):
    def __init__(
        self, alphastd, eigval=imagenet_pca["eigval"], eigvec=imagenet_pca["eigvec"]
    ):
        self.alphastd = alphastd
        assert eigval.shape == (3,)
        assert eigvec.shape == (3, 3)
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0.0:
            return img
        rnd = np.random.randn(3) * self.alphastd
        rnd = rnd.astype("float32")
        v = rnd
        old_dtype = np.asarray(img).dtype
        v = v * self.eigval
        v = v.reshape((3, 1))
        inc = np.dot(self.eigvec, v).reshape((3,))
        img = np.add(img, inc)
        if old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        img = Image.fromarray(img.astype(old_dtype), "RGB")
        return img

    def __repr__(self):
        return self.__class__.__name__ + "()"


def get_datasets(name, root, cutout):

    if name == "cifar10":
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif name == "cifar100":
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif name.startswith("imagenet-1k"):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif name.startswith("ImageNet16"):
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std = [x / 255 for x in [63.22, 61.26, 65.09]]
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    # Data Argumentation
    if name == "cifar10" or name == "cifar100":
        lists = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        if cutout > 0:
            lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        xshape = (1, 3, 32, 32)
    elif name.startswith("ImageNet16"):
        lists = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(16, padding=2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        if cutout > 0:
            lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        xshape = (1, 3, 16, 16)
    elif name == "tiered":
        lists = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(80, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        if cutout > 0:
            lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform = transforms.Compose(
            [
                transforms.CenterCrop(80),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        xshape = (1, 3, 32, 32)
    elif name.startswith("imagenet-1k"):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        if name == "imagenet-1k":
            xlists = [transforms.RandomResizedCrop(224)]
            xlists.append(
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
                )
            )
            xlists.append(Lighting(0.1))
        elif name == "imagenet-1k-s":
            xlists = [transforms.RandomResizedCrop(224, scale=(0.2, 1.0))]
        else:
            raise ValueError("invalid name : {:}".format(name))
        xlists.append(transforms.RandomHorizontalFlip(p=0.5))
        xlists.append(transforms.ToTensor())
        xlists.append(normalize)
        train_transform = transforms.Compose(xlists)
        test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
        xshape = (1, 3, 224, 224)
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    if name == "cifar10":
        train_data = dset.CIFAR10(
            root, train=True, transform=train_transform, download=True
        )
        test_data = dset.CIFAR10(
            root, train=False, transform=test_transform, download=True
        )
        assert len(train_data) == 50000 and len(test_data) == 10000
    elif name == "cifar100":
        train_data = dset.CIFAR100(
            root, train=True, transform=train_transform, download=True
        )
        test_data = dset.CIFAR100(
            root, train=False, transform=test_transform, download=True
        )
        assert len(train_data) == 50000 and len(test_data) == 10000
    elif name.startswith("imagenet-1k"):
        train_data = dset.ImageFolder(osp.join(root, "train"), train_transform)
        test_data = dset.ImageFolder(osp.join(root, "val"), test_transform)
        assert (
            len(train_data) == 1281167 and len(test_data) == 50000
        ), "invalid number of images : {:} & {:} vs {:} & {:}".format(
            len(train_data), len(test_data), 1281167, 50000
        )
    elif name == "ImageNet16":
        train_data = ImageNet16(root, True, train_transform)
        test_data = ImageNet16(root, False, test_transform)
        assert len(train_data) == 1281167 and len(test_data) == 50000
    elif name == "ImageNet16-120":
        train_data = ImageNet16(root, True, train_transform, 120)
        test_data = ImageNet16(root, False, test_transform, 120)
        assert len(train_data) == 151700 and len(test_data) == 6000
    elif name == "ImageNet16-150":
        train_data = ImageNet16(root, True, train_transform, 150)
        test_data = ImageNet16(root, False, test_transform, 150)
        assert len(train_data) == 190272 and len(test_data) == 7500
    elif name == "ImageNet16-200":
        train_data = ImageNet16(root, True, train_transform, 200)
        test_data = ImageNet16(root, False, test_transform, 200)
        assert len(train_data) == 254775 and len(test_data) == 10000
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    class_num = Dataset2Class[name]
    return train_data, test_data, xshape, class_num


def get_nas_search_loaders(
    train_data, valid_data, dataset, config_root, batch_size, workers
):

    valid_use = False
    if isinstance(batch_size, (list, tuple)):
        batch, test_batch = batch_size
    else:
        batch, test_batch = batch_size, batch_size
    if dataset == "cifar10" or dataset == 'cifar100':
        # split_Fpath = 'configs/nas-benchmark/cifar-split.txt'
        # cifar_split = load_config("{:}/cifar-split.txt".format(config_root), None, None)
        # train_split, valid_split = (
        #     cifar_split.train,
        #     cifar_split.valid,
        # )  # search over the proposed training and validation set
        # logger.log('Load split file from {:}'.format(split_Fpath))      # they are two disjoint groups in the original CIFAR-10 training set
        # To split data
        xvalid_data = deepcopy(train_data)
        #
        # random.seed(61)
        # np.random.seed(61)
        user_data = {}

        # set the non-iid pref
        # client_dict = {}
        # labels = train_data.targets
        # labels = np.unique(labels, axis=0)
        # pref_dist = uniform(5, len(labels))
        # print(pref_dist)
        # client_list = list(range(5))
        #
        # for i in range(len(pref_dist)):
        #     while pref_dist[i] > 0:
        #         client = np.random.choice(client_list, 1, replace=False)[0]
        #         client_dict[client] = labels[i]
        #         pref_dist[i] -= 1
        #         client_list = list(set(client_list) - set([client]))
        #
        # tep_train = data_partition(train_data, client_dict, 0.5)
        # tep_valid = data_partition(valid_data, client_dict, 0.5)
        #
        # for one in tep_train:
        #     if valid_use:
        #         # a = np.random.choice(tep[one], int(len(tep[one])/2), replace=False)
        #         user_data[one] = {'train': tep_train[one], 'test': tep_valid[one]}
        #     else:
        #         a = np.random.choice(tep_train[one], int(len(tep_train[one]) / 2), replace=False)
        #         user_data[one] = {'train': list(set(a)), 'test': list(set(tep_train[one]) - set(a)), 'valid': tep_valid[one]}
        #
        # np.save('Use_valid_{}_{}_non_iid_setting.npy'.format(valid_use, dataset), user_data)
        user_data = np.load('Use_valid_{}_{}_non_iid_setting.npy'.format(valid_use, dataset), allow_pickle=True).item()

        if hasattr(xvalid_data, "transforms"):  # to avoid a print issue
            xvalid_data.transforms = valid_data.transform
        xvalid_data.transform = deepcopy(valid_data.transform)

        search_loader = {}
        valid_loader = {}
        train_loader = {}

        for one in user_data:
            if valid_use:
                search_data = SearchDataset(dataset, [train_data, valid_data], user_data[one]['train'], user_data[one]['test'])
                valid_loader[one] = torch.utils.data.DataLoader(
                    xvalid_data,
                    batch_size=test_batch,
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(user_data[one]['test']),
                    num_workers=workers,
                    pin_memory=True,
                )
            else:
                search_data = SearchDataset(dataset, train_data, user_data[one]['train'], user_data[one]['test'])
                valid_loader[one] = torch.utils.data.DataLoader(
                    xvalid_data,
                    batch_size=test_batch,
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(user_data[one]['valid']),
                    num_workers=workers,
                    pin_memory=True,
                )


        # data loader
            search_loader[one] = torch.utils.data.DataLoader(
                search_data,
                batch_size=batch,
                shuffle=True,
                num_workers=workers,
                pin_memory=True,
            )
            train_loader[one] = torch.utils.data.DataLoader(
                train_data,
                batch_size=batch,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(user_data[one]['train']),
                num_workers=workers,
                pin_memory=True,
            )

    # elif dataset == "cifar100":
        # cifar100_test_split = load_config(
        #     "{:}/cifar100-test-split.txt".format(config_root), None, None
        # )
        # search_train_data = train_data

        # random.seed(61)
        # np.random.seed(61)
        # user_data = {}
        # client_dict = {}
        #
        # labels = search_train_data.targets
        # labels = np.unique(labels, axis=0)
        # pref_dist = uniform(5, len(labels))
        # print(pref_dist)
        # client_list = list(range(5))
        #
        # for i in range(len(pref_dist)):
        #     while pref_dist[i] > 0:
        #         client = np.random.choice(client_list, 1, replace=False)[0]
        #         client_dict[client] = labels[i]
        #         pref_dist[i] -= 1
        #         client_list = list(set(client_list) - set([client]))
        #
        # tep_train = data_partition(train_data, client_dict, 0.5)
        # tep_valid = data_partition(valid_data, client_dict, 0.5)
        #
        # for one in tep_train:
        #     # a = np.random.choice(tep[one], int(len(tep[one])/2), replace=False)
        #     user_data[one] = {'train': tep_train[one], 'test': tep_valid[one]}
        #
        # np.save('{}_non_iid_setting.npy'.format(dataset), user_data)

        # user_data = np.load('{}_non_iid_setting.npy'.format(dataset), allow_pickle=True).item()
        #
        # search_valid_data = deepcopy(valid_data)
        # search_valid_data.transform = train_data.transform
        #
        # search_loader = {}
        # for one in user_data:
        #     search_data = SearchDataset(dataset, [train_data, valid_data], user_data[one]['train'], user_data[one]['test'])
        # # data loader
        #     search_loader[one] = torch.utils.data.DataLoader(
        #         search_data,
        #         batch_size=batch,
        #         shuffle=True,
        #         num_workers=workers,
        #         pin_memory=True,
        #     )
        #

        # search_data = SearchDataset(
        #     dataset,
        #     [search_train_data, search_valid_data],
        #     list(range(len(search_train_data))),
        #     cifar100_test_split.xvalid,
        # )
        # search_loader = torch.utils.data.DataLoader(
        #     search_data,
        #     batch_size=batch,
        #     shuffle=True,
        #     num_workers=workers,
        #     pin_memory=True,
        # )
        # train_loader = torch.utils.data.DataLoader(
        #     train_data,
        #     batch_size=batch,
        #     shuffle=True,
        #     num_workers=workers,
        #     pin_memory=True,
        # )
        # valid_loader = torch.utils.data.DataLoader(
        #     valid_data,
        #     batch_size=test_batch,
        #     sampler=torch.utils.data.sampler.SubsetRandomSampler(
        #         cifar100_test_split.xvalid
        #     ),
        #     num_workers=workers,
        #     pin_memory=True,
        # )
    elif dataset == "ImageNet16-120":
        imagenet_test_split = load_config(
            "{:}/imagenet-16-120-test-split.txt".format(config_root), None, None
        )
        search_train_data = train_data
        search_valid_data = deepcopy(valid_data)
        search_valid_data.transform = train_data.transform
        search_data = SearchDataset(
            dataset,
            [search_train_data, search_valid_data],
            list(range(len(search_train_data))),
            imagenet_test_split.xvalid,
        )
        search_loader = torch.utils.data.DataLoader(
            search_data,
            batch_size=batch,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        )
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_data,
            batch_size=test_batch,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                imagenet_test_split.xvalid
            ),
            num_workers=workers,
            pin_memory=True,
        )
    else:
        raise ValueError("invalid dataset : {:}".format(dataset))
    return search_loader, train_loader, valid_loader


if __name__ == '__main__':
    train_data, test_data, xshape, class_num = dataset = get_datasets('cifar10', '/data02/dongxuanyi/.torch/cifar.python/', -1)
    data_partition(train_data, test_data, 5)

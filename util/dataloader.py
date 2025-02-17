import numpy as np
import torch.utils.data as util_data
from torchvision import transforms
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.datasets as dsets
from sklearn.model_selection import train_test_split
import os
import json
from torchvision.io import read_image
from collections import defaultdict


class RadioImageList(object):
    """
    'abd_normal', 'Airspace_opacity', 'bladder_pathology', 'bowel_abnormality',
    'Bronchiectasis', 'interstitial_lung_disease', 'liver_lesion', 'lung_normal',
    'Nodule', 'osseous_neoplasm', 'ovarian_pathology','pancreatic_lesion',
    'prostate_lesion', 'renal_lesion', 'splenic_leison', 'uterine_pathology'
    """
    def __init__(self, image_lists, flag, inference, gray=True):
        self.imgs = image_lists
        self.class_list = ['abd_normal_ct', 'Airspace_opacity', 'bladder_pathology_ct', 'bowel_abnormality_ct',
                            'Bronchiectasis', 'interstitial_lung_disease', 'liver_lesion_ct', 'lung_normal',
                            'Nodule', 'osseous_neoplasm_ct', 'ovarian_pathology_ct','pancreatic_lesion_ct',
                            'prostate_lesion_ct', 'renal_lesion_ct', 'splenic_lesion_ct', 'uterine_pathology_ct']
        self.label_list = np.arange(0, len(self.class_list)).tolist()
        self.current_labels = torch.tensor(np.array(self.get_label(self.imgs)))
        self.global_weights = self.get_global_classw()
        self.gray = gray
        if flag == "train" and not inference:
            if gray:
                self.transforms = transforms.Compose([transforms.Grayscale(),
                                                      transforms.RandomHorizontalFlip(p=0.5),
                                                      transforms.RandomVerticalFlip(p=0.5),
                                                      transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1))])
            else:
                self.transforms = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                                      transforms.RandomVerticalFlip(p=0.5),
                                                      transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1))])
        else:
            self.transforms = None
        self.resize = transforms.Resize((32, 32))
        self.toGray = transforms.Grayscale()

    def get_global_classw(self):
        y_cout = torch.bincount(self.current_labels)
        cate_w = y_cout.sum() / (y_cout * len(self.class_list))
        return cate_w

    def __getitem__(self, index):
        imgfid, target = self.imgs[index], self.current_labels[index]
        img = read_image(imgfid)
        if self.transforms is not None:
            img = self.transforms(img)
        else:
            if self.gray:
                img = self.toGray(img)
        re_img = self.resize(img)
        onehot_target = self.encode_onehot(target)
        return img/255., re_img/255., onehot_target, index, imgfid

    def normalize_img(self, img):
        mean = img.mean([1, 2], keepdim=True)
        std = img.std([1, 2], keepdim=True)
        std = std + 1e-8
        img = (img - mean) / std
        return img

    def encode_onehot(self, labels):
        one_hot = torch.zeros((len(self.class_list)))
        one_hot[labels] = 1
        return one_hot

    def __len__(self):
        return len(self.imgs)

    def get_label(self, fids):
        labels = []
        for fid in fids:
            cate_id = fid.split('retrieval/')[-1].split('/')[0]
            labels.append(self.label_list[self.class_list.index(cate_id)])
        return labels


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)


def get_radio_data(config, root_path, inference_flat=False, numworkers=1, gray=True, shuffle=True):
    dsets = {}
    dset_loaders = {}
    class_weights = {}
    image_lists = []
    for root, dirs, files in os.walk(root_path, topdown=False):
        for name in files:
            image_lists.append(os.path.join(root, name))

    Train_fid, Valid_test_fid = train_test_split(image_lists, test_size=0.3, random_state=0)
    Valid_fid, Test_fid = train_test_split(Valid_test_fid, test_size=1/3, random_state=0)

    fid_lists = [Train_fid, Valid_fid, Test_fid]
    flag = ["train", "valid", "test"]

    for i in range(len(flag)):
        dsets[flag[i]] = RadioImageList(fid_lists[i], flag=flag[i], inference=inference_flat,gray=gray)
        class_weights[flag[i]] = dsets[flag[i]].global_weights
        shuffle = True if flag[i] == 'train' else False
        print(flag[i], len(dsets[flag[i]]))
        dset_loaders[flag[i]] = util_data.DataLoader(dsets[flag[i]],
                                                      batch_size=config["batch_size"],
                                                      shuffle=shuffle, num_workers=numworkers, worker_init_fn=seed_worker)
    return dset_loaders["train"], dset_loaders["valid"], dset_loaders["test"], \
        len(dsets["train"]), len(dsets["valid"]) ,len(dsets["test"]), class_weights



class PathImageList(object):
    def __init__(self, image_lists, flag, inference):
        self.imgs = image_lists
        self.class_list = ['1', '2', '3', '4', '5','6']
        self.current_labels = torch.tensor(np.array(self.get_label(self.imgs)))
        self.global_weights = self.get_global_classw()
        if flag == "train_set" and not inference:
            self.transforms = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                                  transforms.RandomVerticalFlip(p=0.5),
                                                  transforms.ColorJitter(brightness=.1, hue=.1),
                                                  transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1))])
        else:
            self.transforms = None
        self.resize = transforms.Resize((32, 32))
        self.imgresize = transforms.Resize((224, 224))

    def __getitem__(self, index):
        imgfid, target = self.imgs[index], self.current_labels[index]
        img = read_image(imgfid)
        if self.transforms is not None:
            img = self.transforms(img)
        img = self.imgresize(img)
        onehot_target = self.encode_onehot(target, num_classes=6)
        re_img = self.resize(img)
        return img / 255., re_img/255., onehot_target, index, imgfid

    def get_global_classw(self):
        y_cout = torch.bincount(self.current_labels)
        cate_w = y_cout.sum() / (y_cout * len(self.class_list))
        return cate_w

    def encode_onehot(self,labels, num_classes=10):
        one_hot = torch.zeros((num_classes))
        one_hot[labels] = 1
        return one_hot

    def __len__(self):
        return len(self.imgs)

    def get_label(self, fids):
        labels = []
        for fid in fids:
            if 'Fat' in fid:
                labels.append(5)
            elif 'gland' in fid:
                labels.append(4)
            elif 'Necrosis' in fid:
                labels.append(3)
            elif 'Inflammatory' in fid:
                labels.append(2)
            elif 'tumor' in fid:
                labels.append(1)
            elif 'Stroma' in fid:
                labels.append(0)
        return labels



def get_path_data(config, root_path, inference_flat=False,numworkers=1):
    dsets = {}
    dset_loaders = {}
    image_lists = []
    class_weights = {}
    for root, dirs, files in os.walk(root_path, topdown=False):
        for name in files:
            image_lists.append(os.path.join(root, name))
    Train_fid, Valid_test_fid = train_test_split(image_lists, test_size=0.3, random_state=0)
    Valid_fid, Test_fid = train_test_split(Valid_test_fid, test_size=1 / 3, random_state=0)

    fid_lists = [Train_fid, Valid_fid, Test_fid]
    flag = ["train", "valid", "test"]
    for i in range(len(flag)):
        dsets[flag[i]] = PathImageList(fid_lists[i], flag=flag[i], inference=inference_flat)
        class_weights[flag[i]] = dsets[flag[i]].global_weights
        shuffle = True if flag[i] == 'train' else False
        print(flag[i], len(dsets[flag[i]]))
        dset_loaders[flag[i]] = util_data.DataLoader(dsets[flag[i]],
                                                      batch_size=config["batch_size"],
                                                      shuffle=shuffle, num_workers=numworkers, worker_init_fn=seed_worker)
    return dset_loaders["train"], dset_loaders["valid"], dset_loaders["test"], \
        len(dsets["train"]), len(dsets["valid"]) ,len(dsets["test"]), class_weights


if __name__ == "__main__":
    def get_config():
        config = {
            "alpha": 0.01,
            "resize_size": 224,
            "batch_size": 128,
            "dataset": "radioNew",
            "epoch": 300,
            "test_map": 1,
            "bit_list": [16],
            "gpus": [3],
            "device": "cuda",
            "save_path": "save/New_Retri_ViT_small_llama_pretrained/",
            "gamma": 1,
            "lambda": 0.1,
            "scale": 0.1,
            "pr_curve": False
        }
        return config
    config = get_config()
    train_loader, valid_loader, test_loader, num_train, num_valid, \
        num_test, class_weights = get_radio_data(config, root_path=r'/data/RadImageNet/retrieval/', numworkers=4,
                                                 gray=True)
    for image, label, _, fid in train_loader:
        print(image)
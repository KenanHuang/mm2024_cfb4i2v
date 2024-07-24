import os.path
import random

import numpy as np
import torch
from torch import tensor

from component.base import heapQ
import heapq as pyheapq


class CategoryDB():
    def __init__(self, class_num, max_cap=-1, data_saved_path="./flow_tmp_save/", sset="E", dset="H",reload = False):
        self.class_num = class_num
        self.max_cap = max_cap

        self._db = None
        self.npy_data_list = None
        self.init()

        assert dset in ['H', 'U']
        assert sset in ['E', 'S', 'B']

        if reload is True:
            print(f"reload:{data_saved_path}")
            self.reloadFun(data_saved_path)
        else:
            self.data_saved_path = os.path.join(data_saved_path, f"{sset}2{dset}")
            print("data_saved_path:", self.data_saved_path)
            print("database_cap:", max_cap)
            os.makedirs(self.data_saved_path, exist_ok=True)

    def reloadFun(self,data_path):
        for cls in range(self.class_num):
            npyDataPath = os.path.join(data_path,f"{cls}.npy")
            self.npy_data_list[cls] = npyDataPath

    def insert(self, label, prob, input_data: tensor):
        label = int(label)

        heap = self._db[label]
        heap.insert(prob, input_data.cpu())

    def getData(self, label, reduction="mean"):
        label = int(label)
        npyDataPath = self.npy_data_list[label]
        if npyDataPath is None: return None
        assert os.path.exists(npyDataPath) is True
        assert reduction in ['mean', 'all', 'random']
        npyData = np.load(npyDataPath) 
        tensorList = torch.from_numpy(npyData)
        
        if reduction == 'mean':
            return torch.mean(tensorList, dim=0)

        
        if reduction == 'random':
            return tensorList[random.randrange(0, tensorList.shape[0])] # ori :0->shape[0]

        if reduction == 'all':
            b = tensorList.shape[0]
            outputData = torch.zeros(self.max_cap, *tensorList.shape[1:])
            outputData[:b] = tensorList
            return outputData

    def init(self):
        self._db = [heapQ(self.max_cap) for _ in range(self.class_num)]
        self.npy_data_list = [None for _ in range(self.class_num)]

    def clear(self):
        self._db = [heapQ(self.max_cap) for _ in range(self.class_num)]
        for p in self.npy_data_list:
            if p is None: continue
            os.remove(p)
        self.npy_data_list = [None for _ in range(self.class_num)]

    def __len__(self):
        lengt = 0
        for data in self.npy_data_list:
            if data is not None:
                lengt += 1

        return lengt

    def isEmpty(self):
        for data in self.npy_data_list:
            if data is not None: return False

        return True

    def localize(self):
        self.npy_data_list = [None for _ in range(self.class_num)]
        for i in range(self.class_num):
            saveNpy = os.path.join(self.data_saved_path, f'{i}.npy')
            heap = self._db[i]
            tensorList = heap.getPureData()
            if len(tensorList) < 1: continue
            npyData = np.asarray(torch.stack(tensorList))
            np.save(saveNpy, npyData)
            self.npy_data_list[i] = saveNpy


class CategoryDBwFeat():
    def __init__(self, class_num, max_cap=-1, data_saved_path="./flow_tmp_save_wFeat/", sset="E", dset="H"):
        self.class_num = class_num
        self.max_cap = max_cap

        self._db = None
        self._db_feat = None
        self.npy_data_list = None
        self.npy_data_list_feat = None
        self.init()

        assert dset in ['H', 'U']
        assert sset in ['E', 'S', 'B']
        self.data_saved_path = os.path.join(data_saved_path, f"{sset}2{dset}")
        print("data_saved_path:", self.data_saved_path)
        os.makedirs(self.data_saved_path, exist_ok=True)

    def insert(self, label, prob, input_data: tensor, feat: tensor):
        label = int(label)

        heap = self._db[label]
        heap.insert(prob, input_data.cpu())

        heap_feat = self._db_feat[label]
        heap_feat.insert(prob, feat.cpu())

    def getData(self, label, reduction="all"):
        label = int(label)
        npyDataPath = self.npy_data_list[label]
        npyDataPathFeat = self.npy_data_list_feat[label]
        if npyDataPath is None: return None
        assert os.path.exists(npyDataPath) is True

        if npyDataPathFeat is None: return None
        assert os.path.exists(npyDataPathFeat) is True

        assert reduction in ['all']

        outputData, outputDataFeat = None, None

        npyData = np.load(npyDataPath)
        tensorList = torch.from_numpy(npyData)
        if reduction == 'all':
            outputData = tensorList


        npyDataFeat = np.load(npyDataPathFeat)
        tensorListFeat = torch.from_numpy(npyDataFeat)
        if reduction == 'all':
            outputDataFeat = tensorListFeat

        return outputData, outputDataFeat

    def init(self):
        self._db = [heapQ(self.max_cap) for _ in range(self.class_num)]
        self._db_feat = [heapQ(self.max_cap) for _ in range(self.class_num)]
        self.npy_data_list = [None for _ in range(self.class_num)]
        self.npy_data_list_feat = [None for _ in range(self.class_num)]

    def clear(self):
        self._db = [heapQ(self.max_cap) for _ in range(self.class_num)]
        for p in self.npy_data_list:
            if p is None: continue
            os.remove(p)
        self.npy_data_list = [None for _ in range(self.class_num)]

        self._db_feat = [heapQ(self.max_cap) for _ in range(self.class_num)]
        for p in self.npy_data_list_feat:
            if p is None: continue
            os.remove(p)
        self.npy_data_list_feat = [None for _ in range(self.class_num)]

    def isEmpty(self):
        tag = True
        for data in self.npy_data_list:
            if data is not None:
                tag = False
                break

        for data in self.npy_data_list_feat:
            if data is not None:
                tag = False
                break

        return tag

    def __len__(self):
        lengt = 0
        for data in self.npy_data_list:
            if data is not None:
                lengt += 1

        return lengt

    def localize(self):
        self.npy_data_list = [None for _ in range(self.class_num)]
        self.npy_data_list_feat = [None for _ in range(self.class_num)]
        for i in range(self.class_num):
            saveNpy = os.path.join(self.data_saved_path, f'{i}.npy')
            heap = self._db[i]
            tensorList = heap.getPureData()
            if len(tensorList) < 1: continue
            npyData = np.asarray(torch.stack(tensorList))
            np.save(saveNpy, npyData)
            self.npy_data_list[i] = saveNpy

            saveNpy = os.path.join(self.data_saved_path, f'{i}_feat.npy')
            heap = self._db_feat[i]
            tensorList = heap.getPureData()
            if len(tensorList) < 1: continue
            npyData = np.asarray(torch.stack(tensorList))
            np.save(saveNpy, npyData)
            self.npy_data_list_feat[i] = saveNpy


class FeatDB():
    def __init__(self, cap, data_saved_path="./feat_tmp_save/", ):
        self._db = [None for _ in range(cap)]  # idx -> feat
        self.data_saved_path = data_saved_path
        os.makedirs(data_saved_path,exist_ok=True)

    def insert(self, idx, feat: tensor):
        saveNpy = os.path.join(self.data_saved_path, f'{idx}.npy')
        feat.cpu().detach().numpy().dump(saveNpy)
        self._db[idx] = saveNpy

    def getDate(self, idx):
        npyDataPath = os.path.join(self.data_saved_path, f'{idx}.npy')
        assert os.path.exists(npyDataPath) is True
        npyData = np.load(npyDataPath,allow_pickle=True)
        featTensor = torch.from_numpy(npyData)
        return featTensor

    def __len__(self):
        lengt = 0
        for d in self._db:
            if d is not None:
                lengt += 1

        return lengt

class PseudoProbDB():
    def __init__(self,  max_cap=-1):
        print(f"Init size:{max_cap} PseudoProbDB")
        self.max_cap = max_cap

        self._db = []
        self.init()

    def insert(self, prob, sample_idx):
        sample_idx = int(sample_idx)
        pyheapq.heappush(self._db,(prob,sample_idx))

    def getData(self, topk_per = 0.5):
        topkNum = int(self.max_cap * topk_per)
        largeProbDatas = pyheapq.nlargest(n=topkNum,iterable=self._db,key=lambda x:x[0]) # get top percent pro sample
        mask = [0 for _ in range(self.max_cap)]
        for prob, idx in largeProbDatas:
            mask[idx] = 1

        return mask

    def init(self):
        self._db = []

    def clear(self):
        self.init()

    def __len__(self):
        return len(self._db)
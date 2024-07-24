import array
import os
import os.path
import random
from glob import glob

import cv2
import numpy as np
import torch
import torch.utils.data as data_utl
import typing
import einops

RGB = 'RGB'
FLOW = 'FLOW'
RGB_AND_FLOW = 'RGB_AND_FLOW'
def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))

class UnifiedDs(data_utl.Dataset):
    def __init__(self, split_file, transforms=None, class_num=101, dset='H', flow_root='', frame_root='', frame_num=8,
                 mode=RGB):
        self.data, self.dataList = self.getDatas(split_file, class_num, dset, flow_root, frame_root, frame_num)
        print(f"{dset}-", len(self.data))
        self.split_file = split_file
        self.transforms = transforms
        self.dset = dset
        self.frame_num = frame_num
        self.frame_root = frame_root
        self.flow_root = flow_root
        self.mode = mode
        assert mode in [RGB, FLOW, RGB_AND_FLOW]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (dataFileName,target,frame_num,frame_path, flow_path) where target is class_index of the target class.
        """
        # frame_num 8/16 for train 32 for test
        num_used_frame = self.frame_num  # 16#16#8

        # dataFileName,target,frame_num,frame_path, flow_path
        vid, label, nf, frame_path, flow_path, dataPath = self.data[index]

        retDatas = {"index": index}

        if self.mode in [RGB, RGB_AND_FLOW]:
            imgs = self.loadRGBData(frame_path)
            imgs = self.transforms[0](imgs)
            tf_imgs = video_to_tensor(imgs)
            if len(self.transforms) == 2:
                tf_imgs = self.transforms[1](tf_imgs)
            label = label[:, 0:num_used_frame]
            retDatas.update(
                {
                    "RGB": {"data": tf_imgs, "label": torch.from_numpy(label)},
                }
            )

        if self.mode in [FLOW, RGB_AND_FLOW]:
            flows = self.loadFlowData(flow_path)
            flows = self.transforms[0](flows)
            label_flow = label[:, 0:num_used_frame]  # flow 比视频帧数少1
            retDatas.update(
                {
                    "FLOW": {"data": video_to_tensor(flows), "label": torch.from_numpy(label_flow)},
                }
            )

        return retDatas

    def __len__(self):
        return len(self.data)

    def getDatas(self, split_file, class_num, dset, flow_root, frame_root, frame_num):
        datas = []
        dataList = []
        with open(split_file, 'r') as fd:
            lines = fd.readlines()
            for line in lines:
                # /data/datasets/lab/Stanford40/JPEGImages/brushing_teeth_008.jpg 2
                # /data/datasets/lab/EADs/EAD_image_dataset/clap/applauding_136.jpg 0
                # /data/datasets/lab/hmdb51/avi/clap/#20_Rhythm_clap_u_nm_np1_le_goo_4.avi 0
                dataPath, label = line.strip().split()
                label = int(label)  # 0

                # /data/datasets/lab/EADs/EAD_image_dataset, clap , clap_01.jpg
                pathDir, labelName, dataFileName = dataPath.rsplit('/', 2)
                if dset == 'S':
                    labelName = dataFileName.rsplit('_',1)[0]

                if '.avi' in dataFileName or '.mp4' in dataFileName:
                    dataFileName = dataFileName[:-4]

                # /root/data/datasets/lab/EADs/flow/clap/applauding_034.jpg/u
                # /root/data/datasets/lab/hmdb51/flow/clap/that70_sshowslowclap_clap_u_nm_np1_fr_med_0/u
                flow_path = os.path.join(flow_root, labelName, dataFileName)
                if os.path.exists(flow_path) is False:
                    print(f"flow path:{flow_path} is not Exists!")
                    continue

                # /root/data/datasets/lab/EADs/frame/clap/applauding_034.jpg
                # /root/data/datasets/lab/hmdb51/frame/brush_hair/sarah_brushing_her_hair_brush_hair_h_cm_np1_ri_goo_1
                frame_path = os.path.join(frame_root, labelName, dataFileName)
                if os.path.exists(frame_path) is False:
                    print(f"rgb path:{frame_path} is not Exists!")
                    continue

                targets = np.zeros((class_num, frame_num), np.float32)
                targets[int(label), :] = 1  # binary classification

                datas.append((dataFileName, targets, frame_num, frame_path, flow_path, dataPath))
                dataList.append(line)

        return datas, dataList

    def loadRGBData(self, frame_path):
        frames = glob(os.path.join(frame_path, '*.jpg'))
        frames.sort()

        frameSampleIdx = np.linspace(0, len(frames), min(self.frame_num, len(frames)), endpoint=False, dtype=np.int8)
        frameSampleIdx = list(frameSampleIdx)

        totalFrameNum = len(frameSampleIdx)
        needAppend = totalFrameNum < self.frame_num

        rgbDatas = []
        for idx in frameSampleIdx:
            img = cv2.imread(frames[idx])[:, :, [2, 1, 0]]
            w, h, c = img.shape
            if w < 226 or h < 226:
                d = 226. - min(w, h)
                sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(256, 256))  # ,fx=sc,fy=sc)
            img = (img / 255.) * 2 - 1
            rgbDatas.append(img.copy())

        if needAppend:
            copyImg = rgbDatas[-1]
            for _ in range(self.frame_num - totalFrameNum):
                rgbDatas.append(copyImg.copy())

        return np.asarray(rgbDatas, dtype=np.float32)

    def loadFlowData(self, flow_path):
        flows = glob(os.path.join(flow_path, 'u', '*.jpg'))
        totalFlowNum = len(flows)
        needAppend = totalFlowNum < self.frame_num

        flowDatas = []
        for i in range(totalFlowNum):
            imgx = cv2.imread(os.path.join(flow_path, 'u', "{:06d}.jpg".format(i)), cv2.IMREAD_GRAYSCALE)
            imgy = cv2.imread(os.path.join(flow_path, 'v', "{:06d}.jpg".format(i)), cv2.IMREAD_GRAYSCALE)

            w, h = imgx.shape
            if w < 224 or h < 224:
                d = 224. - min(w, h)
                sc = 1 + d / min(w, h)
                imgx = cv2.resize(imgx, dsize=(0, 0), fx=sc, fy=sc)
                imgy = cv2.resize(imgy, dsize=(0, 0), fx=sc, fy=sc)

            imgx = (imgx / 255.) * 2 - 1
            imgy = (imgy / 255.) * 2 - 1
            img = np.asarray([imgx, imgy]).transpose([1, 2, 0])  # 2 channel data
            flowDatas.append(img.copy())

        if needAppend:
            copyFlow = flowDatas[-1]
            for _ in range(self.frame_num - totalFlowNum):
                flowDatas.append(copyFlow.copy())

        return np.asarray(flowDatas, dtype=np.float32)


class UnifiedDsWithVideoPace(data_utl.Dataset):
    def __init__(self, split_file, transforms=None, class_num=101, dset='H', flow_root='', frame_root='', frame_num=8,
                 mode=RGB, max_sample_rate=4):
        self.data, self.dataList = self.getDatas(split_file, class_num, dset, flow_root, frame_root, frame_num)
        print(f"{dset}-", len(self.data))
        self.split_file = split_file
        self.transforms = transforms
        self.dset = dset
        self.frame_num = frame_num
        self.frame_root = frame_root
        self.flow_root = flow_root
        self.mode = mode
        self.max_sample_rate = max_sample_rate  # NOTE:video pace label :[0,max_sample_rate)!! similar to gap between frame
        assert mode in [RGB, FLOW, RGB_AND_FLOW]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (dataFileName,target,frame_num,frame_path, flow_path) where target is class_index of the target class.
        """
        # frame_num 8/16 for train 32 for test
        num_used_frame = self.frame_num  # 16#16#8

        # dataFileName,target,frame_num,frame_path, flow_path
        vid, label, nf, frame_path, flow_path, dataPath = self.data[index]

        retDatas = {"index": index}

        sr = random.randint(1, self.max_sample_rate)
        paceLabel = sr - 1  # [0,self.max_sample_rate)
        if self.mode in [RGB, RGB_AND_FLOW]:
            imgs = self.loadRGBData(frame_path=frame_path, sample_rate=sr)
            imgs = self.transforms[0](imgs)
            tf_imgs = video_to_tensor(imgs)
            if len(self.transforms) == 2:
                tf_imgs = self.transforms[1](tf_imgs)
            label = label[:, 0:num_used_frame]
            retDatas.update(
                {
                    "RGB": {"data": video_to_tensor(imgs), "label": torch.from_numpy(label),
                            "pace_label": torch.tensor(paceLabel)},
                }
            )

        if self.mode in [FLOW, RGB_AND_FLOW]:
            flows = self.loadFlowData(flow_path)
            flows = self.transforms[0](flows)
            label_flow = label[:, 0:num_used_frame]  # flow 比视频帧数少1
            retDatas.update(
                {
                    "FLOW": {"data": video_to_tensor(flows), "label": torch.from_numpy(label_flow)},
                }
            )

        return retDatas

    def __len__(self):
        return len(self.data)

    def getDatas(self, split_file, class_num, dset, flow_root, frame_root, frame_num):
        datas = []
        dataList = []
        with open(split_file, 'r') as fd:
            lines = fd.readlines()
            for line in lines:
                # /data/datasets/lab/EADs/EAD_image_dataset/clap/applauding_136.jpg 0
                # /data/datasets/lab/hmdb51/avi/clap/#20_Rhythm_clap_u_nm_np1_le_goo_4.avi 0
                dataPath, label = line.strip().split()
                label = int(label)  # 0

                # /data/datasets/lab/EADs/EAD_image_dataset, clap , clap_01.jpg
                pathDir, labelName, dataFileName = dataPath.rsplit('/', 2)
                if dset == 'S':
                    labelName = dataFileName.rsplit('_', 1)[0]

                if '.avi' in dataFileName or '.mp4' in dataFileName:
                    dataFileName = dataFileName[:-4]

                # /root/data/datasets/lab/EADs/flow/clap/applauding_034.jpg/u
                # /root/data/datasets/lab/hmdb51/flow/clap/that70_sshowslowclap_clap_u_nm_np1_fr_med_0/u
                flow_path = os.path.join(flow_root, labelName, dataFileName)
                if os.path.exists(flow_path) is False:
                    print(f"flow path:{flow_path} is not Exists!")
                    continue

                # /root/data/datasets/lab/EADs/frame/clap/applauding_034.jpg
                # /root/data/datasets/lab/hmdb51/frame/brush_hair/sarah_brushing_her_hair_brush_hair_h_cm_np1_ri_goo_1
                frame_path = os.path.join(frame_root, labelName, dataFileName)
                if os.path.exists(frame_path) is False:
                    print(f"rgb path:{frame_path} is not Exists!")
                    continue

                targets = np.zeros((class_num, frame_num), np.float32)
                targets[int(label), :] = 1  # binary classification

                datas.append((dataFileName, targets, frame_num, frame_path, flow_path, dataPath))
                dataList.append(line)

        return datas, dataList

    def getLoopRgbIdx(self, total_frames, req_frame_num, sample_rate):
        frame_idx = []
        idx = 0

        start_frame = random.randint(0,total_frames-1)
        while start_frame + idx * sample_rate >= total_frames: # 保证初始的取值不会超出数组下标
            start_frame = random.randint(0,total_frames-1)

        # for i in range(req_frame_num):
        while len(frame_idx) < req_frame_num:
            frame_idx.append(start_frame + idx * sample_rate)

            if (start_frame + (idx + 1) * sample_rate) >= total_frames:
                start_frame = 0
                idx = 0
            else:
                idx += 1

        return frame_idx

    def loadRGBData(self, frame_path, sample_rate):
        frames = glob(os.path.join(frame_path, '*.jpg'))
        frames.sort()

        # frameSampleIdx = np.linspace(0, len(frames), min(self.frame_num, len(frames)), endpoint=False, dtype=np.int8)
        # frameSampleIdx = list(frameSampleIdx)
        frameSampleIdx = self.getLoopRgbIdx(total_frames=len(frames), req_frame_num=self.frame_num,
                                            sample_rate=sample_rate)

        rgbDatas = []
        for idx in frameSampleIdx:
            # try:
            #     img = cv2.imread(frames[idx])[:, :, [2, 1, 0]]
            # except Exception as e:
            #     print(len(frames))
            #     print(idx)
            #     print(frameSampleIdx)

            img = cv2.imread(frames[idx])[:, :, [2, 1, 0]]

            w, h, c = img.shape
            if w < 226 or h < 226:
                d = 226. - min(w, h)
                sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(256, 256))  # ,fx=sc,fy=sc)
            img = (img / 255.) * 2 - 1
            rgbDatas.append(img.copy())

        return np.asarray(rgbDatas, dtype=np.float32)

    def loadFlowData(self, flow_path):
        flows = glob(os.path.join(flow_path, 'u', '*.jpg'))
        totalFlowNum = len(flows)
        needAppend = totalFlowNum < self.frame_num

        flowDatas = []
        for i in range(totalFlowNum):
            imgx = cv2.imread(os.path.join(flow_path, 'u', "{:06d}.jpg".format(i)), cv2.IMREAD_GRAYSCALE)
            imgy = cv2.imread(os.path.join(flow_path, 'v', "{:06d}.jpg".format(i)), cv2.IMREAD_GRAYSCALE)

            w, h = imgx.shape
            if w < 224 or h < 224:
                d = 224. - min(w, h)
                sc = 1 + d / min(w, h)
                imgx = cv2.resize(imgx, dsize=(0, 0), fx=sc, fy=sc)
                imgy = cv2.resize(imgy, dsize=(0, 0), fx=sc, fy=sc)

            imgx = (imgx / 255.) * 2 - 1
            imgy = (imgy / 255.) * 2 - 1
            img = np.asarray([imgx, imgy]).transpose([1, 2, 0])  # 2 channel data
            flowDatas.append(img.copy())

        if needAppend:
            copyFlow = flowDatas[-1]
            for _ in range(self.frame_num - totalFlowNum):
                flowDatas.append(copyFlow.copy())

        return np.asarray(flowDatas, dtype=np.float32)

class UnifiedDsWithRelativeVideoPace(data_utl.Dataset):
    def __init__(self, split_file, transforms=None, class_num=101, dset='H', flow_root='', frame_root='', frame_num=8,
                 mode=RGB, sample_rate=None):
        self.data, self.dataList = self.getDatas(split_file, class_num, dset, flow_root, frame_root, frame_num)
        print(f"{dset}-", len(self.data))
        self.split_file = split_file
        self.transforms = transforms
        self.dset = dset
        self.frame_num = frame_num
        self.frame_root = frame_root
        self.flow_root = flow_root
        self.mode = mode
        if isinstance(sample_rate, typing.List) and len(sample_rate) == 1:
            self.sample_rate_set = list(range(1,sample_rate[0]+1))
        elif isinstance(sample_rate,typing.List) and len(sample_rate) > 1:
            self.sample_rate_set = sample_rate
        else:
            raise Exception('Unknown type:',sample_rate)
        assert mode in [RGB, FLOW, RGB_AND_FLOW]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (dataFileName,target,frame_num,frame_path, flow_path) where target is class_index of the target class.
        """
        # frame_num 8/16 for train 32 for test
        num_used_frame = self.frame_num  # 16#16#8

        # dataFileName,target,frame_num,frame_path, flow_path
        vid, label, nf, frame_path, flow_path, dataPath = self.data[index]

        retDatas = {"index": index}

        srs = random.sample(self.sample_rate_set,k = 2)
        if self.mode in [RGB, RGB_AND_FLOW]:
            imgs1 = self.loadRGBData(frame_path=frame_path, sample_rate=srs[0])
            imgs1 = self.transforms[0](imgs1)
            imgs2 = self.loadRGBData(frame_path=frame_path, sample_rate=srs[1])
            imgs2 = self.transforms[0](imgs2)
            label = label[:, 0:num_used_frame]
            retDatas.update(
                {
                    "RGB": {"data": video_to_tensor(imgs1), "label": torch.from_numpy(label)},
                    "RGB2": {"data": video_to_tensor(imgs2)}, # resample pace rate
                }
            )

        if self.mode in [FLOW, RGB_AND_FLOW]:
            flows = self.loadFlowData(flow_path)
            flows = self.transforms[0](flows)
            label_flow = label[:, 0:num_used_frame]  # flow 比视频帧数少1
            retDatas.update(
                {
                    "FLOW": {"data": video_to_tensor(flows), "label": torch.from_numpy(label_flow)},
                }
            )

        return retDatas

    def __len__(self):
        return len(self.data)

    def getDatas(self, split_file, class_num, dset, flow_root, frame_root, frame_num):
        datas = []
        dataList = []
        with open(split_file, 'r') as fd:
            lines = fd.readlines()
            for line in lines:
                # /data/datasets/lab/EADs/EAD_image_dataset/clap/applauding_136.jpg 0
                # /data/datasets/lab/hmdb51/avi/clap/#20_Rhythm_clap_u_nm_np1_le_goo_4.avi 0
                dataPath, label = line.strip().split()
                label = int(label)  # 0

                # /data/datasets/lab/EADs/EAD_image_dataset, clap , clap_01.jpg
                pathDir, labelName, dataFileName = dataPath.rsplit('/', 2)
                if dset == 'S':
                    labelName = dataFileName.rsplit('_', 1)[0]

                if '.avi' in dataFileName or '.mp4' in dataFileName:
                    dataFileName = dataFileName[:-4]

                # /root/data/datasets/lab/EADs/flow/clap/applauding_034.jpg/u
                # /root/data/datasets/lab/hmdb51/flow/clap/that70_sshowslowclap_clap_u_nm_np1_fr_med_0/u
                flow_path = os.path.join(flow_root, labelName, dataFileName)
                if os.path.exists(flow_path) is False:
                    print(f"flow path:{flow_path} is not Exists!")
                    continue

                # /root/data/datasets/lab/EADs/frame/clap/applauding_034.jpg
                # /root/data/datasets/lab/hmdb51/frame/brush_hair/sarah_brushing_her_hair_brush_hair_h_cm_np1_ri_goo_1
                frame_path = os.path.join(frame_root, labelName, dataFileName)
                if os.path.exists(frame_path) is False:
                    print(f"rgb path:{frame_path} is not Exists!")
                    continue

                targets = np.zeros((class_num, frame_num), np.float32)
                targets[int(label), :] = 1  # binary classification

                datas.append((dataFileName, targets, frame_num, frame_path, flow_path, dataPath))
                dataList.append(line)

        return datas, dataList

    def getLoopRgbIdx(self, total_frames, req_frame_num, sample_rate):
        frame_idx = []
        idx = 0

        start_frame = random.randint(0,total_frames-1)
        while start_frame + idx * sample_rate >= total_frames: # 保证初始的取值不会超出数组下标
            start_frame = random.randint(0,total_frames-1)

        # for i in range(req_frame_num):
        while len(frame_idx) < req_frame_num:
            frame_idx.append(start_frame + idx * sample_rate)

            if (start_frame + (idx + 1) * sample_rate) >= total_frames:
                start_frame = 0
                idx = 0
            else:
                idx += 1

        return frame_idx

    def loadRGBData(self, frame_path, sample_rate):
        frames = glob(os.path.join(frame_path, '*.jpg'))
        frames.sort()

        # frameSampleIdx = np.linspace(0, len(frames), min(self.frame_num, len(frames)), endpoint=False, dtype=np.int8)
        # frameSampleIdx = list(frameSampleIdx)
        frameSampleIdx = self.getLoopRgbIdx(total_frames=len(frames), req_frame_num=self.frame_num,
                                            sample_rate=sample_rate)

        rgbDatas = []
        for idx in frameSampleIdx:
            # try:
            #     img = cv2.imread(frames[idx])[:, :, [2, 1, 0]]
            # except Exception as e:
            #     print(len(frames))
            #     print(idx)
            #     print(frameSampleIdx)

            img = cv2.imread(frames[idx])[:, :, [2, 1, 0]]

            w, h, c = img.shape
            if w < 226 or h < 226:
                d = 226. - min(w, h)
                sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(256, 256))  # ,fx=sc,fy=sc)
            img = (img / 255.) * 2 - 1
            rgbDatas.append(img.copy())

        return np.asarray(rgbDatas, dtype=np.float32)

    def loadFlowData(self, flow_path):
        flows = glob(os.path.join(flow_path, 'u', '*.jpg'))
        totalFlowNum = len(flows)
        needAppend = totalFlowNum < self.frame_num

        flowDatas = []
        for i in range(totalFlowNum):
            imgx = cv2.imread(os.path.join(flow_path, 'u', "{:06d}.jpg".format(i)), cv2.IMREAD_GRAYSCALE)
            imgy = cv2.imread(os.path.join(flow_path, 'v', "{:06d}.jpg".format(i)), cv2.IMREAD_GRAYSCALE)

            w, h = imgx.shape
            if w < 224 or h < 224:
                d = 224. - min(w, h)
                sc = 1 + d / min(w, h)
                imgx = cv2.resize(imgx, dsize=(0, 0), fx=sc, fy=sc)
                imgy = cv2.resize(imgy, dsize=(0, 0), fx=sc, fy=sc)

            imgx = (imgx / 255.) * 2 - 1
            imgy = (imgy / 255.) * 2 - 1
            img = np.asarray([imgx, imgy]).transpose([1, 2, 0])  # 2 channel data
            flowDatas.append(img.copy())

        if needAppend:
            copyFlow = flowDatas[-1]
            for _ in range(self.frame_num - totalFlowNum):
                flowDatas.append(copyFlow.copy())

        return np.asarray(flowDatas, dtype=np.float32)
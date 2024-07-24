import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow or rgb+flow')
parser.add_argument('-gpu_id', type=str, nargs='?', default='2', help="device id to run")
parser.add_argument('-save_model_path', type=str)
parser.add_argument('-save_model', action='store_true')
parser.add_argument('-load_model_path', type=str, default='')
parser.add_argument('-save_npy_path', type=str, default='./flow_tmp_save')
parser.add_argument('-root', type=str)
parser.add_argument('-droot', type=str)
parser.add_argument('-CLASS_NUM', type=int)
parser.add_argument('-frame_num', type=int, default=8)
parser.add_argument('-bs', type=int, default=16)
parser.add_argument('-dset', type=str)
parser.add_argument('-sset', type=str)
parser.add_argument('-method', type=str)
parser.add_argument('-sourcefile', type=str)
parser.add_argument('-targetfile', type=str)
parser.add_argument('-flow_r1', type=float, default=1.0)
parser.add_argument('-lr', type=float)
parser.add_argument('-vdlr', type=float,default=-1)
parser.add_argument('-update', type=float, default=1)
parser.add_argument('-loc_loss_r1', type=float,default = 1.0)
parser.add_argument('-pre', type=float, default=9, help="pretrain epoch")
parser.add_argument('-cdb_frame_sel', type=str, default='mean')
parser.add_argument('-cdb_size', type=int, default=50)
parser.add_argument('-sample_rate', type=int, default=4)
parser.add_argument('-video_pace_cls_r1', type=float, default=1.0)
parser.add_argument('-total_epoch', type=int, default=20)
parser.add_argument('-val_time', type=int, default=3)
parser.add_argument('-test_interval', type=int, default=1) # default 1->test every epoch
parser.add_argument('-post_vdpace', action='store_true') # default 1->test every epoch

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torchvision import transforms
from tool import videotransforms

from net.pytorch_i3d import InceptionI3d,VideoPacePredictor

from datasets_latest import UnifiedDs as Dataset
from datasets_latest import UnifiedDsWithVideoPace as DatasetVP

PHASE_TRAIN = 'train'
PHASE_VAL = 'val'

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def run(max_steps=20, root='/root/data/datasets/lab/EADs', droot='/root/data/datasets/lab/hmdb51',
        sourcefile='./root/data/datasets/lab/model_data/I2V/EtoH/list/sourcelistname_ead2hmdb_shared.txt',
        targetfile='/root/data/datasets/lab/model_data/I2V/EtoH/list/targetlistname.txt', batch_size=16,
        save_model_path='I3D_RGB', CLASS_NUM=13, args=None):
    from component.database import CategoryDB
    import time
    maxCap = args.cdb_size
    assert maxCap > -1
    npySavePath = args.save_npy_path if args.save_npy_path[-1] != "/" else args.save_npy_path[:-1]
    time_suffix = time.strftime("%Y%m%d%H%M%S") # generate random suffix
    fullNpySavedPath = f"{npySavePath}_{args.cdb_frame_sel}_{time_suffix}"
    cdb = CategoryDB(class_num=CLASS_NUM, max_cap=maxCap, sset=args.sset, dset=args.dset,
                     data_saved_path=fullNpySavedPath)
    # setup dataset
    CLASS_NUM = CLASS_NUM

    train_transforms = [transforms.Compose([videotransforms.RandomCrop(224),
                                            videotransforms.RandomHorizontalFlip(),
                                            ])]  # ,

    test_transforms = [transforms.Compose([videotransforms.CenterCrop(224)])]

    frameNumPrefix = '' if args.frame_num == 8 else f"_{args.frame_num}"

    src_flowRoot = os.path.join(root, 'flow' + frameNumPrefix)  # /root/data/datasets/lab/EADs/flow
    src_frameRoot = os.path.join(root, 'frame' + frameNumPrefix)  # /root/data/datasets/lab/EADs/frame

    target_flowRoot = os.path.join(droot, 'flow' + frameNumPrefix)  # /root/data/datasets/lab/hmdb51/flow
    target_frameRoot = os.path.join(droot, 'frame')  # /root/data/datasets/lab/hmdb51/frame

    valFrameNum = 32
    target_flowRoot_val = os.path.join(droot, 'flow_32')  # /root/data/datasets/lab/hmdb51/flow_32
    target_frameRoot_val = os.path.join(droot, 'frame')  # /root/data/datasets/lab/hmdb51/frame

    print(f"src_flowRoot:{src_flowRoot}")
    print(f"src_frameRoot:{src_frameRoot}")
    print(f"target_flowRoot:{target_flowRoot}")
    print(f"target_frameRoot:{target_frameRoot}")
    print(f"target_flowRoot_val:{target_flowRoot_val}")
    print(f"target_frameRoot_val:{target_frameRoot_val}")
    print(f"frame_num:{args.frame_num} flow_r1:{args.flow_r1}")
    print(f"method:{args.method}")
    print(f"pre:{args.pre}")
    print(f"lr:{args.lr}")

    vdlr = args.vdlr if args.vdlr > 0 else args.lr
    print(f"vdlr:{vdlr}")
    print(f"loc_loss_r1:{args.loc_loss_r1}")
    print(f"model saved:{save_model_path} save? {args.save_model}")
    print(f"model load from:{args.load_model_path}")
    print(f"cdb_frame_sel:{args.cdb_frame_sel}")
    print(f"Mode:{args.mode} Data:{args.sset}-{args.dset}")
    print(f"save_npy_path:{args.save_npy_path}")
    print(f"max_sample_rate:{args.sample_rate}")
    print(f"video_pace_cls_r1:{args.video_pace_cls_r1}")
    print(f"post_vdpace:{args.post_vdpace}")
    print(f"test_interval:{args.test_interval}")
    print(f"total_epoch:{max_steps}")

    dataset = Dataset(sourcefile, transforms=train_transforms, class_num=CLASS_NUM, dset=args.sset,
                      flow_root=src_flowRoot,
                      frame_root=src_frameRoot, frame_num=args.frame_num, mode=args.mode)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                                             pin_memory=True, drop_last=True)
    tgt_dataset = DatasetVP(targetfile, transforms=train_transforms, class_num=CLASS_NUM, dset=args.dset,
                          flow_root=target_flowRoot,
                          frame_root=target_frameRoot, frame_num=args.frame_num, mode=args.mode,max_sample_rate=args.sample_rate)
    tgt_dataloader = torch.utils.data.DataLoader(tgt_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                                                 drop_last=True)

    val_dataset = Dataset(targetfile, transforms=test_transforms, class_num=CLASS_NUM, dset=args.dset,
                          flow_root=target_flowRoot_val,
                          frame_root=target_frameRoot_val, frame_num=valFrameNum, mode=args.mode)  # for all evaluation we use 32 frames
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=2,
                                                 pin_memory=True, drop_last=False)

    dataloaders = {'train': dataloader, 'val': val_dataloader, 'tgt': tgt_dataloader}

    # setup the model
    i3d = InceptionI3d(400, in_channels=3)  # for flow input
    i3d.load_state_dict(torch.load('./models/rgb_imagenet.pt'))
    i3d.replace_logits(CLASS_NUM)
    i3d.cuda()

    i3d_flow = InceptionI3d(400, in_channels=2)  # for flow input
    i3d_flow.load_state_dict(torch.load('./models/flow_imagenet.pt'))
    i3d_flow.replace_logits(CLASS_NUM)
    i3d_flow.cuda()

    vid_pace_predictor = VideoPacePredictor(sample_rate_num=args.sample_rate) # input i3d feat
    vid_pace_predictor.cuda()

    lr = args.lr
    vdlr = args.vdlr if args.vdlr > 0 else lr
    # optimizer = optim.SGD(list(i3d.parameters()) + list(i3d_flow.parameters()) + list(vid_pace_predictor.parameters()), lr=lr, momentum=0.9,
    #                       weight_decay=0.0001)  # 0.0000001
    optimizer = optim.SGD(
        [
            {'params': list(i3d.parameters()) + list(i3d_flow.parameters()) ,'lr':lr},
            {'params': list(vid_pace_predictor.parameters()) ,'lr':vdlr},
        ],
        momentum=0.9,
        weight_decay=0.0001
    )
    if args.sset == 'E' and args.dset == 'H':
        ms = [30, 40]
    elif args.sset == 'B' and args.dset == 'U':
        ms = [15, 20]
    else:
        ms = [10, 15]

    print(f"Dset {args.sset}-{args.dset} decay ms:{ms}")
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, ms, gamma=0.1)

    loadStep = -1
    if args.load_model_path:
        print(f"load model from:{args.load_model_path}")
        ck = torch.load(args.load_model_path)
        i3d.load_state_dict(ck['i3d'])
        i3d_flow.load_state_dict(ck['i3d_flow'])
        optimizer.load_state_dict(ck['optimizer'])
        lsd = ck['lr_schedule']
        lr_sched.load_state_dict(lsd)
        loadStep = lsd['last_epoch']

    i3d = nn.DataParallel(i3d)
    i3d_flow = nn.DataParallel(i3d_flow)
    vid_pace_predictor = nn.DataParallel(vid_pace_predictor)

    num_steps_per_update = args.update  # accum gradient
    steps = max(loadStep, 0)
    # train it
    LOSS = torch.zeros((max_steps, dataset.__len__(), 1)).cuda()

    METHOD = args.method
    useCDB = False  # tag for warmup phase 

    bestAcc = 0  # the best Accuracy
    bestAccSf = 0 # the best Accuracy after softmax
    bestEp = -1  
    bestEpSf = -1 

    while steps < max_steps:  # for epoch in range(num_epochs):
        print('Step {}/{}'.format(steps, max_steps))
        print('-' * 10)

        useCDB = useCDB or (steps >= args.pre)
        if steps == args.pre:
            print(f"step:{steps} Use CDB:{useCDB}")

        # Each epoch has a training and validation phase
        start_test = True
        for phase in [PHASE_TRAIN, PHASE_VAL]:

            # skip val phase
            if phase == PHASE_VAL and steps > 0 and steps != (max_steps - 1) and steps % args.test_interval != 0:
                continue

            if phase == PHASE_TRAIN:
                i3d.train(True)
                i3d_flow.train(True)
                vid_pace_predictor.train(True)
            elif phase == PHASE_VAL:
                i3d.train(False)  # Set model to evaluate mode
                i3d_flow.train(False)
                vid_pace_predictor.train(False)

            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()

            # Iterate over data.
            ACC = 0
            ACC_sf = 0
            LEN = 0
            iter_tgt = iter(tgt_dataloader)
            len_it = len(iter_tgt)
            iter_tgt_num = 0

            for data in dataloaders[phase]:  # 'train' - s, 'val' - t
                iter_tgt_num += 1
                num_iter += 1
                # get the inputs
                if phase == PHASE_TRAIN:
                    index = data["index"]
                    srcData_rgb = data['RGB']
                    srcData_flow = data['FLOW']
                    rgb_inputs, rgb_labels = srcData_rgb["data"], srcData_rgb["label"]
                    flow_inputs, flow_labels = srcData_flow["data"], srcData_flow["label"]
                    # b,c,f,h,w  #b,l,f
                    if useCDB is True and cdb is not None and len(cdb) > 0:
                        # Note: replace the original flow with the retrieved flow
                        searchLabels = torch.max(torch.mean(rgb_labels, dim=-1), dim=-1)[1]  # b l -> b
                        flow_gen_arr = flow_inputs
                        for idx, sl in enumerate(searchLabels):
                            flow_tensor = cdb.getData(sl.item(), reduction=args.cdb_frame_sel)
                            if flow_tensor is None:
                                print(f"ep:{steps} fetch from db exp: label:{sl.item()}")
                            flow_gen_arr[idx] = flow_tensor if flow_tensor is not None else flow_inputs[idx]

                        flow_inputs = flow_gen_arr
                    if iter_tgt_num == len_it - 1:
                        iter_tgt_num = 1
                        iter_tgt = iter(tgt_dataloader)

                    tgtData = next(iter_tgt)
                    index_tgt = tgtData["index"]
                    tgtData_rgb = tgtData["RGB"]
                    tgtData_flow = tgtData["FLOW"]
                    rgb_inputs_tgt, flow_inputs_tgt = tgtData_rgb["data"], tgtData_flow["data"]
                    vid_pace_label = tgtData_rgb['pace_label']
                elif phase == PHASE_VAL:
                    index = data["index"]
                    tgtData_rgb = data['RGB']
                    tgtData_flow = data['FLOW']
                    rgb_inputs, rgb_labels = tgtData_rgb["data"], tgtData_rgb["label"]
                    flow_inputs, flow_labels = tgtData_flow["data"], tgtData_flow["label"]

                # wrap them in Variable
                rgb_inputs = Variable(rgb_inputs.cuda())  # b,c,f,h,w
                flow_inputs = Variable(flow_inputs.cuda())  # b,c,f-1,h,w ## 8 frames create 7 flow
                if phase == PHASE_TRAIN:
                    rgb_inputs_tgt = Variable(rgb_inputs_tgt.cuda())
                    flow_inputs_tgt = Variable(flow_inputs_tgt.cuda())
                    vid_pace_label = Variable(vid_pace_label.cuda())
                t = rgb_inputs.size(2)
                t_flow = t 
                rgb_labels = Variable(rgb_labels.cuda())
                flow_labels = Variable(flow_labels.cuda())

                if phase == PHASE_TRAIN:
                    per_frame_logits_rgb, rgb_feat_ = i3d(rgb_inputs)
                    per_frame_logits_flow, flow_feat_ = i3d_flow(flow_inputs)

                    per_frame_logits_tgt_rgb, rgb_ft_ = i3d(rgb_inputs_tgt)
                    per_frame_logits_tgt_flow, flow_ft_ = i3d_flow(flow_inputs_tgt)

                    vid_pace_logit = vid_pace_predictor(rgb_ft_)
                    vid_pace_logit = vid_pace_logit.mean(dim = -1)
                elif phase == PHASE_VAL:
                    with torch.no_grad():
                        per_frame_logits_rgb, rgb_feat_ = i3d(rgb_inputs)
                        per_frame_logits_flow, flow_feat_ = i3d_flow(flow_inputs)
                # upsample to input size
                per_frame_logits_rgb = F.upsample(per_frame_logits_rgb, t, mode='linear')
                per_frame_logits_flow = F.upsample(per_frame_logits_flow, t_flow, mode='linear')
                if phase == PHASE_TRAIN:
                    per_frame_logits_tgt_rgb = F.upsample(per_frame_logits_tgt_rgb, t, mode='linear')
                    per_frame_logits_tgt_flow = F.upsample(per_frame_logits_tgt_flow, t_flow, mode='linear')
                # compute localization loss
                loc_loss_rgb = F.binary_cross_entropy_with_logits(per_frame_logits_rgb, rgb_labels)
                loc_loss_flow = F.binary_cross_entropy_with_logits(per_frame_logits_flow, flow_labels)
                loc_loss = loc_loss_rgb + loc_loss_flow * args.flow_r1
                # compute localization loss
                tot_loc_loss = 0  # += loc_loss#.data[0]

                # compute classification loss (with max-pooling along time B x C x T)
                if phase == PHASE_TRAIN:

                    rgb_feat_ = rgb_feat_.mean(dim=2).view(rgb_feat_.size(0), -1)
                    rgb_feat_ = rgb_feat_ / (torch.norm(rgb_feat_, p=2, dim=1, keepdim=True) + 0.00001)

                    flow_feat_ = flow_feat_.mean(dim=2).view(flow_feat_.size(0), -1)
                    flow_feat_ = flow_feat_ / (torch.norm(flow_feat_, p=2, dim=1, keepdim=True) + 0.00001)

                    cls_loss_rgb = F.binary_cross_entropy_with_logits(torch.mean(per_frame_logits_rgb, dim=2),
                                                                      torch.max(rgb_labels, dim=2)[0],
                                                                      reduction='none').mean(dim=1).view(-1, 1)
                    cls_loss_flow = F.binary_cross_entropy_with_logits(torch.mean(per_frame_logits_flow, dim=2),
                                                                       torch.max(flow_labels, dim=2)[0],
                                                                       reduction='none').mean(dim=1).view(-1, 1)

                    cls_loss_vid_pace = F.cross_entropy(vid_pace_logit,vid_pace_label)

                    vdpace_r1 = args.video_pace_cls_r1
                    if steps < args.pre and args.post_vdpace is True: # 预训练阶段，而且有需要把vdpace设为0
                        vdpace_r1 = 0

                    # cls_loss = cls_loss_rgb + cls_loss_flow * args.flow_r1 + cls_loss_vid_pace * args.video_pace_cls_r1
                    cls_loss = cls_loss_rgb + cls_loss_flow * args.flow_r1 + cls_loss_vid_pace * vdpace_r1
                    tot_cls_loss += cls_loss.sum()  # /(targets.sum()+0.000001)#.data[0]
                    LOSS[steps, index] = cls_loss.data.detach()

                elif phase == PHASE_VAL:
                    cls_loss_rgb = F.binary_cross_entropy_with_logits(torch.mean(per_frame_logits_rgb, dim=2),
                                                                      torch.max(rgb_labels, dim=2)[0],
                                                                      reduction='none').mean(dim=1).view(-1, 1)
                    cls_loss_flow = F.binary_cross_entropy_with_logits(torch.mean(per_frame_logits_flow, dim=2),
                                                                       torch.max(flow_labels, dim=2)[0],
                                                                       reduction='none').mean(dim=1).view(-1, 1)

                    cls_loss = cls_loss_rgb + cls_loss_flow * args.flow_r1
                    rgb_feat_ = rgb_feat_.mean(dim=2).view(rgb_feat_.size(0), -1)
                    rgb_feat_ = rgb_feat_ / (torch.norm(rgb_feat_, p=2, dim=1, keepdim=True) + 0.00001)

                    flow_feat_ = flow_feat_.mean(dim=2).view(flow_feat_.size(0), -1)
                    flow_feat_ = flow_feat_ / (torch.norm(flow_feat_, p=2, dim=1, keepdim=True) + 0.00001)

                if phase == PHASE_VAL:
                    ACC += (torch.max(torch.mean(per_frame_logits_rgb + per_frame_logits_flow, dim=2), dim=1)[1] ==
                            torch.max(torch.max(rgb_labels, dim=2)[0], dim=1)[1]).float().sum()
                    # per_frame_logits_rgb (batch,logit,frame)
                    logit_rgb = F.softmax(per_frame_logits_rgb, dim=1)
                    logit_flow = F.softmax(per_frame_logits_flow, dim=1)
                    batch_ps_label_prob, batch_ps_label = torch.max(torch.mean(logit_rgb + logit_flow, dim=2), dim=1)
                    ACC_sf += (batch_ps_label == torch.max(torch.max(rgb_labels, dim=2)[0], dim=1)[1]).float().sum()

                    LEN += torch.max(per_frame_logits_rgb, dim=2)[0].shape[0]
                    if start_test:
                        all_output = F.softmax(torch.mean(per_frame_logits_rgb + per_frame_logits_flow, dim=2),
                                               dim=1).float().cpu()
                        all_output1 = torch.mean(per_frame_logits_rgb + per_frame_logits_flow, dim=2).float().cpu()
                        all_label = torch.max(torch.max(rgb_labels, dim=2)[0], dim=1)[1].float()
                        start_test = False
                    else:
                        all_output = torch.cat(
                            (all_output, F.softmax(torch.mean(per_frame_logits_rgb + per_frame_logits_flow, dim=2),
                                                   dim=1).float().cpu()), 0)
                        all_output1 = torch.cat((all_output1, torch.mean(per_frame_logits_rgb + per_frame_logits_flow,
                                                                         dim=2).float().cpu()), 0)
                        all_label = torch.cat((all_label, torch.max(torch.max(rgb_labels, dim=2)[0], dim=1)[1].float()),
                                              0)

                if phase == PHASE_TRAIN:
                    loss = cls_loss.mean() / num_steps_per_update
                    loss += loc_loss * args.loc_loss_r1
                    tot_loss += loss  # .data[0]

                    loss.backward()
                elif phase == PHASE_VAL:
                    # select args.frame_num in valFrameNum
                    # cdbFrameSampleIdx = np.linspace(0, valFrameNum, args.frame_num,
                    #                                 endpoint=False, dtype=np.int8) # 测试的时候是用的32帧
                    # cdbFrameSampleIdx = list(cdbFrameSampleIdx)
                    if args.frame_num == 8: #  flow num in val is 32 but 8/16 in trainning
                        cdbFrameSampleIdx = [0, 4, 8, 12, 16, 20, 24, 28]
                    elif args.frame_num == 16:
                        cdbFrameSampleIdx = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

                    if useCDB and cdb is not None:
                        for i in range(len(batch_ps_label)):
                            prob = batch_ps_label_prob[i].item()
                            label = batch_ps_label[i].item()
                            flow_input_data = flow_inputs[i]
                            cdb.insert(label, prob, flow_input_data[:, cdbFrameSampleIdx, :, :])
                    else:
                        pass

                if num_iter == num_steps_per_update and phase == 'train':
                    num_iter = 0
                    clip_gradient(optimizer, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    if steps % 1 == 0:
                        print('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss / (
                                1 * num_steps_per_update), tot_cls_loss / (1 * num_steps_per_update), tot_loss / 1))
                        tot_loss = tot_loc_loss = tot_cls_loss = 0.
            if phase == PHASE_TRAIN:  # The end of training phase
                if useCDB and cdb is not None:  # clear all flows from CFB
                    cdb.clear()

            elif phase == PHASE_VAL:  # The end of evaluation
                print(f"ep:{steps} Acc:", ACC / LEN)
                print(f"ep:{steps} Acc_sf:", ACC_sf / LEN)

                if useCDB and cdb is not None:
                    cdb.localize()  # localize all flows store in CFB

                if bestAccSf < (ACC_sf / LEN):
                    bestAccSf = ACC_sf / LEN
                    bestEpSf = steps

                if bestAcc < (ACC / LEN):
                    bestAcc = ACC / LEN
                    bestEp = steps

                if args.save_model:
                    paraDict = {
                        'i3d': i3d.module.state_dict(),
                        'i3d_flow': i3d_flow.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_schedule': lr_sched.state_dict()
                    }
                    os.makedirs(save_model_path, exist_ok=True)
                    torch.save(paraDict, os.path.join(save_model_path, f"ep{steps}-{ACC_sf / LEN}.pt"))

        steps += 1
        lr_sched.step()

    print(f"Best ep:{bestEp} acc:{bestAcc}")
    print(f"Best ep sf:{bestEpSf} acc sf:{bestAccSf}")

    return ACC / LEN, ACC_sf / LEN


if __name__ == '__main__':
    # need to add argparse
    accArr = []
    accArrSf = []
    valTimes = args.val_time
    for i in range(valTimes):
        print(f"Train on {i}", "-" * 10)
        acc,acc_sf = run(save_model_path=args.save_model_path, CLASS_NUM=args.CLASS_NUM, root=args.root, droot=args.droot,
                         sourcefile=args.sourcefile, targetfile=args.targetfile, args=args, batch_size=args.bs, max_steps=args.total_epoch)

        accArr.append(acc)
        accArrSf.append(acc_sf)

    print("Acc Arr:", accArr)
    print("AccSf Arr:", accArrSf)
    print("Avg Acc:", sum(accArr) / valTimes)
    print("Avg AccSf:", sum(accArrSf) / valTimes)
# Ctypes package used to call the forward warping C library
import argparse
import ctypes
import math
import os
import random
from ctypes import *
import tqdm

# External scripts
from flow_colors import *
from geometry import *
from tools.MiDaS.run import run,getDepthModel



# Parse input arguments
parser = argparse.ArgumentParser(description="Depthstillation options")
parser.add_argument("--input_sourcelist", dest="input_sourcelist", type=str, help="directory path of still image", required=True)
parser.add_argument("--output_dir_path", dest="output_dir_path", type=str, help="directory path of flow", required=True)
parser.add_argument("--num_motions", dest="num_motions", type=int, help="Number of motions", default=1)
parser.add_argument("--mask_type", dest="mask_type", type=str, default="H'", help="Select mask type",
                    choices=["H", "H'"])
parser.add_argument("--no_depth", dest="no_depth", action="store_true", help="Assumes constant depth")
parser.add_argument("--depth_output_path", default="./tools/MiDaS/output", type=str)
parser.add_argument("--depth_model_type", dest="depth_model_type", type=str,
                    help="type of depth estimator",default="dpt_beit_large_512")
args = parser.parse_args()

# Import warping library
clibPath = "external/forward_warping/libwarping.so"
print("clib PATH:",clibPath)
lib = ctypes.cdll.LoadLibrary(clibPath)
warp = lib.forward_warping

# cmd
# CUDA_VISIBLE_DEVICES=0 python ./get_flow_from_img_datasets.py --num_motions 8 --input_sourcelist /root/data/datasets/lab/model_data/I2V/EtoH/list/sourcelistname_ead2hmdb_shared.txt --output_dir_path /root/data/datasets/lab/EADs/frame --mask_type H

# Fix random seeds
random.seed(1024)
np.random.seed(1024)

# check If input exists
# /root/data/datasets/lab/EADs/EAD_image_dataset
sourceListPath = args.input_sourcelist
assert os.path.exists(sourceListPath) is True
print("input sourceList:", sourceListPath)

# Create directories to save outputs
outputPath = args.output_dir_path
if os.path.exists(outputPath) is False:
    os.makedirs(outputPath,exist_ok=True)
print("output path:",outputPath)

# load depth estimate model
depth_model, depth_transform, depth_net_w, depth_net_h = getDepthModel(model_path="",model_type=args.depth_model_type)

with open(sourceListPath,'r') as fd:
    lines = fd.readlines()

# Init progress bar
pbar = tqdm.tqdm(total=len(lines))


for line in lines:
    imgPath, _ = line.strip().split()
    _,labelName,imgFileName = imgPath.rsplit('/',2)

    frameOutPutPath = os.path.join(outputPath, labelName, imgFileName)
    if os.path.exists(frameOutPutPath) and len(os.listdir(frameOutPutPath)) >= args.num_motions: 
        pbar.update(1)
        continue

    print(f"Gen: {frameOutPutPath}")

    os.makedirs(frameOutPutPath, exist_ok=True)

    # Processing
    baseInput = imgPath # init img
    for idm in range(args.num_motions):
        rgb = cv2.imread(baseInput, -1)
        rgb = cv2.resize(rgb,(384,384)) # Accelerate the speed
        if len(rgb.shape) < 3:
            h, w = rgb.shape
            rgb = np.stack((rgb, rgb, rgb), -1)
        else:
            h, w, _ = rgb.shape

        depthPath = run(input_path=baseInput, output_path=args.depth_output_path, model_type=args.depth_model_type,
                        grayscale=True,model = depth_model,transform=depth_transform,net_w=depth_net_w,net_h=depth_net_h)

        # Open D0 (inverse) depth map and resize to I0
        depth = cv2.imread(depthPath, -1) / (2 ** 16 - 1)
        if depth.shape[0] != h or depth.shape[1] != w:
            depth = cv2.resize(depth, (w, h))

        # Get depth map and normalize
        depth = 1.0 / (depth + 0.005)
        depth[depth > 100] = 100

        # Set depth to constant value in case we do not want to use depth
        if args.no_depth:
            depth = depth * 0. + 1.

        # Cast I0 and D0 to pytorch tensors
        rgb = torch.from_numpy(np.expand_dims(rgb, 0))
        depth = torch.from_numpy(np.expand_dims(depth, 0)).float()

        # Fix a plausible K matrix
        K = np.array([[[0.58, 0, 0.5, 0], [0, 0.58, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]]], dtype=np.float32)

        K[:, 0, :] *= w
        K[:, 1, :] *= h
        inv_K = torch.from_numpy(np.linalg.pinv(K))
        K = torch.from_numpy(K)

        # Create objects in charge of 3D projection
        backproject_depth = BackprojectDepth(1, h, w)
        project_3d = Project3D(1, h, w)

        # Prepare p0 coordinates
        meshgrid = np.meshgrid(range(w), range(h), indexing="xy")
        p0 = np.stack(meshgrid, axis=-1).astype(np.float32)

        ## Loop over the number of motions
        # for idm in range(args.num_motions):
        # Initiate masks dictionary
        masks = {}
        # Generate random vector t # the movement of camera
        # Random sign
        scx = ((-1) ** random.randrange(2))
        scy = ((-1) ** random.randrange(2))
        scz = ((-1) ** random.randrange(2))
        # Random scalars in -0.2,0.2, excluding -0.1,0.1 to avoid zeros / very small motions
        displace = 0.01 # init 0.1
        cx = (random.random() * displace + displace) * scx
        cy = (random.random() * displace + displace) * scy
        cz = (random.random() * displace + displace) * scz
        camera_mot = [cx, cy, cz]
        # camera_mot = [0, 0, 0]

        # generate random triplet of Euler angles 
        # Random sign random.randrange(2) => [0,1]
        sax = ((-1) ** random.randrange(2))
        say = ((-1) ** random.randrange(2))
        saz = ((-1) ** random.randrange(2))
        # say = 1
        # Random angles in -pi/18,pi/18, excluding -pi/36,pi/36 to avoid zeros / very small rotations
        coef = 180.0 # init:36
        moveStep = 1e-6
        # ax = (random.random() * math.pi / coef + math.pi / coef) * sax
        ax = (random.random() * math.pi / coef + math.pi / coef) * sax
        ay = (random.random() * math.pi / coef + math.pi / coef) * say
        az = (random.random() * math.pi / coef + math.pi / coef) * saz
        camera_ang = [ax, ay, az]
        # camera_ang = [0, 0, 0]

        axisangle = torch.from_numpy(np.array([[camera_ang]], dtype=np.float32))
        translation = torch.from_numpy(np.array([[camera_mot]]))

        # Compute (R|t)
        T1 = transformation_from_parameters(axisangle, translation)

        # Back-projection
        cam_points = backproject_depth(depth, inv_K)

        # Apply transformation T_{0->1}
        p1, z1 = project_3d(cam_points, K, T1)
        z1 = z1.reshape(1, h, w)

        # Bring p1 coordinates in [0,W-1]x[0,H-1] format
        p1 = (p1 + 1) / 2
        p1[:, :, :, 0] *= w - 1
        p1[:, :, :, 1] *= h - 1

        # Create auxiliary data for warping
        dlut = torch.ones(1, h, w).float() * 1000
        safe_y = np.maximum(np.minimum(p1[:, :, :, 1].long(), h - 1), 0)
        safe_x = np.maximum(np.minimum(p1[:, :, :, 0].long(), w - 1), 0)
        warped_arr = np.zeros(h * w * 5).astype(np.uint8)
        img = rgb.reshape(-1)

        # Call forward warping routine (C code)
        warp(c_void_p(img.numpy().ctypes.data), c_void_p(safe_x[0].numpy().ctypes.data),
             c_void_p(safe_y[0].numpy().ctypes.data), c_void_p(z1.reshape(-1).numpy().ctypes.data),
             c_void_p(warped_arr.ctypes.data), c_int(h), c_int(w))
        warped_arr = warped_arr.reshape(1, h, w, 5).astype(np.uint8)

        # Warped image
        im1_raw = warped_arr[0, :, :, 0:3]

        # Validity mask H
        masks["H"] = warped_arr[0, :, :, 3:4]

        # Collision mask M
        masks["M"] = warped_arr[0, :, :, 4:5]
        # Keep all pixels that are invalid (H) or collide (M)
        masks["M"] = 1 - (masks["M"] == masks["H"]).astype(np.uint8)

        # # Dilated collision mask M'
        kernel = np.ones((3, 3), np.uint8)
        masks["M'"] = cv2.dilate(masks["M"], kernel, iterations=1)
        masks["P"] = (np.expand_dims(masks["M'"], -1) == masks["M"]).astype(np.uint8)

        # Final mask P
        masks["H'"] = masks["H"] * masks["P"]

        # Compute flow as p1-p0
        # flow_01 = p1 - p0
        #
        # # Get 16-bit flow (KITTI format) and colored flows
        # flow_16bit = cv2.cvtColor(np.concatenate((flow_01 * 64. + (2 ** 15), np.ones_like(flow_01)[:, :, :, 0:1]), -1)[0],
        #                           cv2.COLOR_BGR2RGB)
        # flow_color = flow_to_color(flow_01[0].numpy(), convert_to_bgr=True)

        im1 = cv2.inpaint(im1_raw, 1 - masks[args.mask_type], 3, cv2.INPAINT_TELEA)

        # Save images

        im1_path = os.path.join(frameOutPutPath, "%05d.jpg" % (idm))
        cv2.imwrite(im1_path, im1)
        baseInput = im1_path # generated img

        # Clear cache
        ctypes._reset_cache()

    # update progress bar
    pbar.update(1)

# Close progress bar, cya!
pbar.close()
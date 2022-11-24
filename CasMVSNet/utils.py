import numpy as np
import torchvision.utils as vutils
import torch, random
import torch.nn.functional as F


# print arguments
def print_args(args):
    print("################################  args  ################################")
    for k, v in args.__dict__.items():
        print("{0: <10}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
    print("########################################################################")


# torch.no_grad warpper for functions
def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper


# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper


@make_recursive_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type {} for tensor2float".format(type(vars)))


@make_recursive_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


@make_recursive_func
def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.to(torch.device("cuda"))
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


def save_scalars(logger, mode, scalar_dict, global_step):
    scalar_dict = tensor2float(scalar_dict)
    for key, value in scalar_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_scalar(name, value, global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_scalar(name, value[idx], global_step)


def save_images(logger, mode, images_dict, global_step):
    images_dict = tensor2numpy(images_dict)

    def preprocess(name, img):
        if not (len(img.shape) == 3 or len(img.shape) == 4):
            raise NotImplementedError("invalid img shape {}:{} in save_images".format(name, img.shape))
        if len(img.shape) == 3:
            img = img[:, np.newaxis, :, :]
        img = torch.from_numpy(img[:1])
        return vutils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True)

    for key, value in images_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_image(name, preprocess(name, value), global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_image(name, preprocess(name, value[idx]), global_step)


class DictAverageMeter(object):
    def __init__(self):
        self.data = {}
        self.count = 0

    def update(self, new_input):
        self.count += 1
        if len(self.data) == 0:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] = v
        else:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] += v

    def mean(self):
        return {k: v / self.count for k, v in self.data.items()}


# a wrapper to compute metrics for each image individually
def compute_metrics_for_each_image(metric_func):
    def wrapper(depth_est, depth_gt, mask, *args):
        batch_size = depth_gt.shape[0]
        results = []
        # compute result one by one
        for idx in range(batch_size):
            ret = metric_func(depth_est[idx], depth_gt[idx], mask[idx], *args)
            results.append(ret)
        return torch.stack(results).mean()

    return wrapper


@make_nograd_func
@compute_metrics_for_each_image
def Thres_metrics(depth_est, depth_gt, mask, thres):
    assert isinstance(thres, (int, float))
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    errors = torch.abs(depth_est - depth_gt)
    err_mask = errors > thres
    return torch.mean(err_mask.float())


# NOTE: please do not use this to build up training loss
@make_nograd_func
@compute_metrics_for_each_image
def AbsDepthError_metrics(depth_est, depth_gt, mask, thres=None):
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    error = (depth_est - depth_gt).abs()
    if thres is not None:
        error = error[(error >= float(thres[0])) & (error <= float(thres[1]))]
        if error.shape[0] == 0:
            return torch.tensor(0, device=error.device, dtype=error.dtype)
    return torch.mean(error)

import torch.distributed as dist
def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def reduce_scalar_outputs(scalar_outputs):
    world_size = get_world_size()
    if world_size < 2:
        return scalar_outputs
    with torch.no_grad():
        names = []
        scalars = []
        for k in sorted(scalar_outputs.keys()):
            names.append(k)
            scalars.append(scalar_outputs[k])
        scalars = torch.stack(scalars, dim=0)
        dist.reduce(scalars, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            scalars /= world_size
        reduced_scalars = {k: v for k, v in zip(names, scalars)}

    return reduced_scalars

import torch
from bisect import bisect_right
# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        #print("base_lr {}, warmup_factor {}, self.gamma {}, self.milesotnes {}, self.last_epoch{}".format(
        #    self.base_lrs[0], warmup_factor, self.gamma, self.milestones, self.last_epoch))
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

    
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def local_pcd(depth, intr):
    nx = depth.shape[1]  # w
    ny = depth.shape[0]  # h
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='xy')
    x = x.reshape(nx * ny)
    y = y.reshape(nx * ny)
    p2d = np.array([x, y, np.ones_like(y)])
    p3d = np.matmul(np.linalg.inv(intr), p2d)
    depth = depth.reshape(1, nx * ny)
    p3d *= depth
    p3d = np.transpose(p3d, (1, 0))
    p3d = p3d.reshape(ny, nx, 3).astype(np.float32)
    return p3d

def generate_pointcloud(rgb, depth, ply_file, intr, scale=1.0):
    """
    Generate a colored point cloud in PLY format from a color and a depth image.

    Input:
    rgb_file -- filename of color image
    depth_file -- filename of depth image
    ply_file -- filename of ply file

    """
    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
    points = []
    for v in range(rgb.shape[0]):
        for u in range(rgb.shape[1]):
            color = rgb[v, u] #rgb.getpixel((u, v))
            Z = depth[v, u] / scale
            if Z == 0: continue
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            points.append("%f %f %f %d %d %d 0\n" % (X, Y, Z, color[0], color[1], color[2]))
    file = open(ply_file, "w")
    file.write('''ply
            format ascii 1.0
            element vertex %d
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            property uchar alpha
            end_header
            %s
            ''' % (len(points), "".join(points)))
    file.close()
    print("save ply, fx:{}, fy:{}, cx:{}, cy:{}".format(fx, fy, cx, cy))

def xyzrgb_save_mask(filename,cloud_xyz,rgb,M,mask):
    H,W = mask.shape[:2]
    def validindex(x, y):
        return x>=0 and x<W and y>=0 and y<H and mask[y,x]>200
    validxy_arr = np.argwhere(mask[:,:]>200)
    vert=[]
    face=[]
    vertcolor=[]
    vert_index_map=np.zeros([H,W,1]).astype(np.int32)
    for i in range(validxy_arr.shape[0]):
        y = validxy_arr[i,0]
        x = validxy_arr[i,1]
        vert.append(cloud_xyz[y,x])
        vertcolor.append(rgb[y,x])
        vert_index_map[y,x]=i+1
    for i in range(validxy_arr.shape[0]):
        y = validxy_arr[i, 0]
        x = validxy_arr[i, 1]
        distance=10
        if validindex(x,y+1) and validindex(x+1,y) :
            if abs(cloud_xyz[y,x]).max()>0 and\
            abs(cloud_xyz[y,x]-cloud_xyz[y,x+1]).max()<distance\
            and abs(cloud_xyz[y,x]-cloud_xyz[y+1,x]).max()<distance\
            and abs(cloud_xyz[y,x+1]-cloud_xyz[y+1,x]).max()<distance :
              face.append([vert_index_map[y,x,0],vert_index_map[y+1,x,0],vert_index_map[y,x+1,0]])
        if validindex(x-1,y) and validindex(x,y-1):
            if abs(cloud_xyz[y,x]).max()>0 and\
            abs(cloud_xyz[y,x]-cloud_xyz[y,x-1]).max()<distance\
            and abs(cloud_xyz[y,x]-cloud_xyz[y-1,x]).max()<distance\
            and abs(cloud_xyz[y,x-1]-cloud_xyz[y-1,x]).max()<distance :
                face.append([vert_index_map[y,x,0],vert_index_map[y-1,x,0],vert_index_map[y,x-1,0]])

    objfile = file = open(filename,'w')
    all_points = np.array(vert)
    all_faces = np.array(face)
    # print np.min(all_faces)
    vertcolor = np.array(vertcolor)
    for i in range(all_points.shape[0]):
        p=np.dot(np.array([[all_points[i,0],all_points[i,1],all_points[i,2],1]]),M)
        s="v  "+str(p[0,0])+"  "+str(p[0,1])+"  "+\
            str(p[0,2])+"  "+str(vertcolor[i,0])+"  "+\
                str(vertcolor[i,1])+"  "+str(vertcolor[i,2])+"\n"
        file.write(s)
    for j in range(all_faces.shape[0]):
        # print all_faces[j,0]
        objfile.writelines('f ' + str(all_faces[j,0])+' '+str(all_faces[j,1])+' '+str(all_faces[j,2])+'\n')
    objfile.close()
    print(filename,"xyzrgb_saved")
def make_coordinate_grid(spatial_size, type, znorm=True):
    '''
    return [H,W,2]
    '''
    h, w = spatial_size
    x = torch.arange(w).type(type) 
    y = torch.arange(h).type(type) 
    if znorm:
        x = 2 * x / w - 1.
        y = 2 * y / h - 1.
    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)
    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)
    return meshed

def depth2cloud(depth, K):
    if len(depth.shape) == 2:
        depth = depth[:, :, None]
    h, w = depth.shape[1:3]
    fx = K[ 0, 0]
    fy = K[ 1, 1]
    cx = K[ 0, 2]
    cy = K[1, 2]
    cxy = K[0:2, 2]
    fxy = np.array([fx,fy])  # [bs,2]
    coord_abs = make_coordinate_grid([h, w],\
        depth.dtype, znorm=False).unsqueeze(0).to(depth.device)  # [bs,H,W,2]
    xy_flat = (coord_abs - cxy[ None, None, :]) / \
        (fxy[ None, None, :]+1e-10)  # [bs,H,W,2]
    cloud = torch.cat([xy_flat*depth, depth], 3)
    return cloud

def generate_meshcloud(img, depth, obj_filename,mask, K,M):
        depth=np.expand_dims(depth,axis=0)
        depth=np.expand_dims(depth,axis=-1)
        depth =torch.from_numpy(depth)
        cloud=depth2cloud(depth,K).numpy()
        cloud=cloud.squeeze(axis=0)
        xyzrgb_save_mask(obj_filename,cloud,img,M,mask)
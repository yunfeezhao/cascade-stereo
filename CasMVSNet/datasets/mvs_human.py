from torch.utils.data import Dataset
import numpy as np
import os, cv2, time, math
from PIL import Image
from datasets.data_io import *
import  pickle
import scipy.io as io
import copy, json 
# the DTU dataset preprocessed by Yao Yao (only for training)
class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, **kwargs):
        super(MVSDataset, self).__init__()
        if mode=="train":
            self.peoplenames=["wangjingjing","wangyijie","zhaoyunfei","zhengjunshen","zhonglin","zyf","wenjiayi"]#os.listdir(datapath)
        elif mode=="test":
            self.peoplenames=["zrr"]#["wangjingjing","wangyijie","wenjiayi","yefan","zhaoyunfei","zhengjunshen","zhonglin"]#os.listdir(datapath)
        
        self.start_h, self.start_w=0,0
        self.scale = 1
        self.datapath = datapath
        self.listfile ='mvs_views.txt'
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.camparam=self.readcampara()
        self.kwargs = kwargs
        print("mvsdataset kwargs", self.kwargs)
        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()
    def readcampara(self):
        cam_param={}
        for peoplename in self.peoplenames:
            optimizedcameraKM=io.loadmat(self.datapath+peoplename+"/optimizedcameraKM.mat")
            optimizedcameraKM=optimizedcameraKM['optimizedcameraKM']
            optimizedcameraK=optimizedcameraKM[0,0][0]
            optimizedcameraM=optimizedcameraKM[0,0][1]
            depth_optimized=io.loadmat(self.datapath+peoplename+"/depth_optimized.mat")
            depth_optimized=depth_optimized['depth_optimized']
            errors=depth_optimized[0,0][1]
            optimizedcameraKM={
                "extrinsic":optimizedcameraM,
                "intrinsic":optimizedcameraK,
                "systermerrors":errors
            }
            cam_param.update({peoplename:optimizedcameraKM})
        return cam_param
    def readview(self,path):
        view = []
        lines = open(path).readlines()
        for line in lines:
            a = line.split('\n')[0].split(',')
            view.append(a)
        return view    

    def build_list(self):
        metas = []
        for peoplename in self.peoplenames:
            with open(self.datapath+peoplename+"/index.txt") as f:
                imgs = f.readlines()
                imgs = [line.split('\n')[0] for line in imgs]
            views = self.readview(self.datapath+peoplename+"/"+self.listfile)#'/data3/MVFR_DATASET/Res256/meta/6_4/three_views.txt'
            # lines
            for imgn in imgs :
                for view in views:
                    ref_view = view[0]
                    src_views = view[1:self.nviews]
                    metas.append((peoplename,imgn, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, peoplename,vid):
        intrinsics = np.zeros([3,3])
        extrinsics = copy.deepcopy(self.camparam[peoplename]["extrinsic"][0,int(vid)])
        intrinsics[:,:] = copy.deepcopy(self.camparam[peoplename]["intrinsic"][0,int(vid)])
        systermerror = copy.deepcopy(self.camparam[peoplename]["systermerrors"][int(vid)])
        intrinsics[:2, :] /= 4.0
        return intrinsics, extrinsics,systermerror

    def read_img(self, filename,intrinsics):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        np_img=self.prepare_img(np_img)
        intrinsics[:2, :] /= self.scale
        intrinsics[0, 2] -= (self.start_w/4)
        intrinsics[1, 2] -= (self.start_h/4)
        return np_img,intrinsics

    def prepare_img(self, hr_img):
        #w1600-h1200-> 800-600 ; crop -> 640, 512; downsample 1/4 -> 160, 128
        if(self.scale>1):
            #downsample
            h, w = hr_img.shape[0:2]
            hr_img = cv2.resize(hr_img, (w//self.scale, h//self.scale), interpolation=cv2.INTER_NEAREST)
        if(self.start_w>0 or self.start_h>0):
            #crop
            h, w = hr_img.shape[0:2]
            target_h, target_w = h-2*self.start_h,w-2*self.start_w
            hr_img = hr_img[self.start_h: self.start_h + target_h, self.start_w: self.start_w + target_w]

            # #downsample
            # lr_img = cv2.resize(hr_img_crop, (target_w//4, target_h//4), interpolation=cv2.INTER_NEAREST)

        return hr_img

    def read_mask_hr(self, filename):
        img = cv2.imread(filename,-1)[:,:,0]
        np_img = np.array(img, dtype=np.float32)
        np_img = (np_img > 255*0.92).astype(np.float32)
        np_img = self.prepare_img(np_img)

        h, w = np_img.shape
        np_img_ms = {
            "stage1": cv2.resize(np_img, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(np_img, (w//2, h//2), interpolation=cv2.INTER_NEAREST),
            "stage3": np_img,
        }
        return np_img_ms

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def read_depth_hr(self, filename,syserror):
        # read pfm depth file
        #w1600-h1200-> 800-600 ; crop -> 640, 512; downsample 1/4 -> 160, 128
        img = cv2.imread(filename,-1)
        depth_lr = np.array(img, dtype=np.float32)-syserror
        depth_lr = self.prepare_img(depth_lr)

        h, w = depth_lr.shape
        depth_lr_ms = {
            "stage1": cv2.resize(depth_lr, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(depth_lr, (w//2, h//2), interpolation=cv2.INTER_NEAREST),
            "stage3": depth_lr,
        }
        return depth_lr_ms
    def depthfilter_numpy(self,d, mask):
        d_val = d[mask > 0]
        d_mean = np.mean(d_val)
        d_std = np.sqrt(np.var(d_val))
        offset = np.abs(d - d_mean)
        filtermask = (offset < 2 * d_std)
        ###filtermask = (offset < 2.5 * d_std)
        mask = mask * filtermask
        return mask,d_mean-2 * d_std, 2*self.interval_scale*2 * d_std/192
    def __getitem__(self, idx):
        meta = self.metas[idx]
        peoplename,imgn, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        mask = None
        depth_values = None
        proj_matrices = []

        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.datapath,peoplename+'/'+vid+"/image/"+imgn+'.jpg')

            intrinsics, extrinsics ,syserror= self.read_cam_file(peoplename,vid)
            extrinsics = extrinsics.transpose()
            img,intrinsics = self.read_img(img_filename,intrinsics)

            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics

            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                depth_filename_hr = os.path.join(self.datapath,peoplename+'/'+vid+"/depth/"+imgn+'.png')
                mask_filename_hr = os.path.join(self.datapath,peoplename+'/'+vid+"/dis_mask/"+imgn+'_mask.png')
                mask_read_ms = self.read_mask_hr(mask_filename_hr)
                depth_ms = self.read_depth_hr(depth_filename_hr,syserror)
                #mask_read_ms["stage3"],depth_min, depth_interval=self.depthfilter_numpy(depth_ms["stage3"],mask_read_ms["stage3"])

                #get depth values
                depth_min,depth_interval=1200,6
                depth_max = depth_interval * self.ndepths + depth_min
                depth_values = np.arange(depth_min, depth_max, depth_interval, dtype=np.float32)[0:192]

                mask = mask_read_ms

            imgs.append(img)

        #all
        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        #ms proj_mats
        proj_matrices = np.stack(proj_matrices)
        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 2
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 4

        proj_matrices_ms = {
            "stage1": proj_matrices,
            "stage2": stage2_pjmats,
            "stage3": stage3_pjmats
        }

        return {"imgs": imgs,
                "proj_matrices": proj_matrices_ms,
                "depth": depth_ms,
                "depth_values": depth_values,
                "filename": peoplename + '/{}/' +imgn+ '_{}'.format(view_ids[0]) + "{}",
                "mask": mask }
import traceback

import numpy as np
from PIL import Image, ImageFilter
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torch.utils.data import Dataset
from torchvision.transforms import functional as func_transforms

from libyana.transformutils import colortrans, handutils
from datasets.queries import BaseQueries, TransQueries, one_query_in 

class SeqSet(Dataset): 
    def __init__(
        self,
        pose_dataset,
        inp_res,
        scale_jittering,
        center_jittering,
        train,
        queries,
        ntokens,
        spacing,
        hue=0.15,
        saturation=0.5,
        contrast=0.5,
        brightness=0.5,
        blur_radius=0.5,
    ):
        # Dataset attributes
        self.pose_dataset = pose_dataset
        self.inp_res = tuple(inp_res) 
        
        # Sequence attributes
        self.ntokens = ntokens
        self.spacing = spacing
        

        # Color jitter attributes
        self.hue = hue
        self.contrast = contrast
        self.brightness = brightness
        self.saturation = saturation
        self.blur_radius = blur_radius


        # Training attributes
        self.train = train
        self.scale_jittering = scale_jittering
        self.center_jittering = center_jittering


        self.queries = queries



    def __len__(self):
        return len(self.pose_dataset)
    


    def get_sample(self, idx, query=None, seq_txn=None, color_augm=None, space_augm=None):
        if query is None:
            query = self.queries
        sample = {} 
          
        if BaseQueries.ACTIONIDX in query:
            sample[BaseQueries.ACTIONIDX]=self.pose_dataset.get_action_idxs(idx)
        if BaseQueries.OBJIDX in query:
            sample[BaseQueries.OBJIDX]=self.pose_dataset.get_obj_idxs(idx)

        center = np.array((480 / 2, 270 / 2))
        scale = 480

        # Get original image
        if BaseQueries.IMAGE in query or TransQueries.IMAGE in query:
            img = self.pose_dataset.get_image(idx,txn=seq_txn)
            if BaseQueries.IMAGE in query:
                sample[BaseQueries.IMAGE] = np.array(img)

        # Data augmentation
        if space_augm is not None:
            center = space_augm["center"]
            scale = space_augm["scale"]
        elif self.train :
            # Randomly jitter center
            # Center is located in square of size 2*center_jitter_factor
            # in center of cropped image
            center_jit = Uniform(low=-1, high=1).sample((2,)).numpy()
            center_offsets = self.center_jittering * scale * center_jit
            center = center + center_offsets.astype(int)

            # Scale jittering
            scale_jit = Normal(0, 1).sample().item() + 1
            scale_jittering = self.scale_jittering * scale_jit
            scale_jittering = np.clip(scale_jittering, 1 - self.scale_jittering, 1 + self.scale_jittering)
            scale = scale * scale_jittering

        space_augm = {"scale": scale, "center": center}
        sample["space_augm"] = space_augm



        # Get 2D hand joints
        if (TransQueries.JOINTS2D in query) or (TransQueries.IMAGE in query):
            affinetrans, post_rot_trans = handutils.get_affine_transform(center, scale, self.inp_res, rot=0)
            if TransQueries.AFFINETRANS in query:
                sample[TransQueries.AFFINETRANS] = affinetrans
        
        if BaseQueries.JOINTS2D in query or TransQueries.JOINTS2D in query:
            joints2d = self.pose_dataset.get_joints2d(idx)
            if BaseQueries.JOINTS2D in query:
                sample[BaseQueries.JOINTS2D] = joints2d.astype(np.float32)
        if TransQueries.JOINTS2D in query:
            rows = handutils.transform_coords(joints2d, affinetrans)
            sample[TransQueries.JOINTS2D] = np.array(rows).astype(np.float32)
        

        if BaseQueries.CAMINTR in query or TransQueries.CAMINTR in query:
            camintr = self.pose_dataset.get_camintr(idx)
            if BaseQueries.CAMINTR in query:
                sample[BaseQueries.CAMINTR] = camintr.astype(np.float32)
            if TransQueries.CAMINTR in query:
                # Rotation is applied as extr transform
                new_camintr = post_rot_trans.dot(camintr)
                sample[TransQueries.CAMINTR] = new_camintr.astype(np.float32)


        # Get 2.5D hand joints
        if BaseQueries.JOINTSABS25D in query or (TransQueries.JOINTSABS25D in query):
            joints25d=self.pose_dataset.get_abs_joints25d(idx)
            if BaseQueries.JOINTSABS25D in query:
                sample[BaseQueries.JOINTSABS25D]=joints25d.astype(np.float32)
            if TransQueries.JOINTSABS25D in query:
                joints25d_t=joints25d.astype(np.float32).copy()
                joints25d_t[:,:2]=handutils.transform_coords(joints25d_t[:,:2], affinetrans)
                sample[TransQueries.JOINTSABS25D]=joints25d_t
        

        # Get 3D hand joints
        if (
            (BaseQueries.JOINTS3D in query)
            or (TransQueries.JOINTS3D in query)
        ):
            # Center on root joint
            center3d_queries = [TransQueries.JOINTS3D, BaseQueries.JOINTS3D]
            if one_query_in(center3d_queries, query):
                joints3d = self.pose_dataset.get_joints3d(idx)

                if BaseQueries.JOINTS3D in query:
                    sample[BaseQueries.JOINTS3D] = joints3d.astype(np.float32) 

        # Get rgb image
        sample["color_augm"] = None
        if TransQueries.IMAGE in query:
            # Data augmentation
            if self.train:
                blur_radius = Uniform(low=0, high=1).sample().item() * self.blur_radius
                img = img.filter(ImageFilter.GaussianBlur(blur_radius))
                if color_augm is None:
                    bright, contrast, sat, hue = colortrans.get_color_params(
                        brightness=self.brightness,
                        saturation=self.saturation,
                        hue=self.hue,
                        contrast=self.contrast,
                    )
                else:
                    sat = color_augm["sat"]
                    contrast = color_augm["contrast"]
                    hue = color_augm["hue"]
                    bright = color_augm["bright"]
                img = colortrans.apply_jitter(
                    img, brightness=bright, saturation=sat, hue=hue, contrast=contrast
                )
                sample["color_augm"] = {"sat": sat, "bright": bright, "contrast": contrast, "hue": hue}
            else:
                sample["color_augm"] = None
            # Create buffer white image if needed
            # Transform and crop
            img = handutils.transform_img(img, affinetrans, self.inp_res)
            img = img.crop((0, 0, self.inp_res[0], self.inp_res[1]))

            # Tensorize and normalize_img
            img = func_transforms.to_tensor(img).float()
            img = func_transforms.normalize(img, [0.5, 0.5, 0.5], [1, 1, 1])
            if TransQueries.IMAGE in query:
                sample[TransQueries.IMAGE] = img
            
        
        #Get meta sample info
        sample["sample_info"]=self.pose_dataset.get_sample_info(idx)
        return sample

    def get_safesample(self, idx,seq_txn=None, color_augm=None, space_augm=None):
        try:
            sample = self.get_sample(idx, self.queries, seq_txn=seq_txn, color_augm=color_augm, space_augm=space_augm)
        except Exception:
            traceback.print_exc()
            assert False, f"Encountered error processing sample {idx}" 
        
        return sample

    def __getitem__(self, idx, verbose=False):
        fidx=self.pose_dataset.get_start_frame_idx(idx)

        seq_txn=self.pose_dataset.open_seq_lmdb(fidx)

                
        sample = self.get_safesample(fidx,seq_txn=seq_txn)
        frame_idx=self.pose_dataset.get_dataidx(fidx)


        sample["dist2query"] = 0
        sample["not_padding"] = 1
        space_augm = sample.pop("space_augm")
        color_augm = sample.pop("color_augm")

        samples = [sample]  
        cur_idx=frame_idx
        for sample_idx in range(self.ntokens-1):
            fut_idx, fut_not_padding=self.pose_dataset.get_future_frame_idx(cur_idx=cur_idx,
                                                fut_idx=cur_idx+self.spacing,
                                                spacing=self.spacing,
                                                verbose=verbose)
            
            sample_fut_frame = self.get_safesample(fut_idx,seq_txn=seq_txn, color_augm=color_augm, space_augm=space_augm)
            sample_fut_frame["dist2query"] = fut_idx-frame_idx
            sample_fut_frame["not_padding"] = fut_not_padding

            sample_fut_frame.pop("space_augm")
            sample_fut_frame.pop("color_augm")
            samples.append(sample_fut_frame)

            if fut_idx!=cur_idx:
                cur_idx=fut_idx
                
        return samples

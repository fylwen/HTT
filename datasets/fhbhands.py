import os
import lmdb

import numpy as np
from PIL import Image, ImageFile
from datasets import fhbutils
from datasets.queries import BaseQueries,TransQueries, get_trans_queries


ImageFile.LOAD_TRUNCATED_IMAGES = True


class FHBHands(object):
    def __init__(
        self,
        dataset_folder,
        split,#
        ntokens_pose,
        ntokens_action,
        spacing,
        is_shifting_window,
        split_type="actions",
    ):
        super().__init__()


        self.ntokens_pose=ntokens_pose
        self.ntokens_action=ntokens_action
        self.spacing=spacing
        self.is_shifting_window=is_shifting_window
    

        self.all_queries = [
            BaseQueries.IMAGE,           
            BaseQueries.CAMINTR,

            TransQueries.JOINTS2D, 
            TransQueries.JOINTSABS25D,

            BaseQueries.JOINTS3D,
            BaseQueries.ACTIONIDX,
            BaseQueries.OBJIDX,
        ]
             
       
        trans_queries = get_trans_queries(self.all_queries)
        self.all_queries.extend(trans_queries) 

        # Get camera info
        self.cam_extr = np.array(
            [
                [0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
                [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
                [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
                [0, 0, 0, 1],
            ]
        )
        self.cam_intr = np.array([[1395.749023, 0, 935.732544], [0, 1395.749268, 540.681030], [0, 0, 1]])

        self.reorder_idx = np.array(
            [0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19, 20]
        )
        self.name = "fhb"
        split_opts = ["actions", "objects", "subjects"]
        self.subjects = ["Subject_1", "Subject_2", "Subject_3", "Subject_4", "Subject_5", "Subject_6"]
        if split_type not in split_opts:
            raise ValueError(
                "Split for dataset {} should be in {}, got {}".format(self.name, split_opts, split_type)
            )

        self.split_type = split_type 
        

        self.root = dataset_folder
        self.info_root = os.path.join(self.root, "Subjects_info")
        self.info_split = os.path.join(self.root, "data_split_action_recognition.txt")
        self.info_video_order_for_supervision= os.path.join(self.root,'video_annotation.json') 

        self.reduce_res = True
        small_rgb = os.path.join(self.root, "Video_files_480")
        if os.path.exists(small_rgb) and self.reduce_res: 
            self.rgb_root = small_rgb
            self.reduce_factor = 1 / 4
        else:
            self.rgb_root = os.path.join(self.root, "Video_files")
            self.reduce_factor = 1
            assert False, 'Warning-reduce factor is 1'
        self.skeleton_root = os.path.join(self.root, "Hand_pose_annotation_v1")
        
        self.split = split
        self.rgb_template = "color_{:04d}.jpeg"


        #Load action labels
        path_action_object_info = os.path.join(self.root,'action_object_info.txt')
        action_object_info,object_to_idx,action_to_idx = fhbutils.get_action_object_infos(path_action_object_info)
        self.action_object_info={}
        
        self.action_object_info=action_object_info
        self.object_to_idx=object_to_idx
        self.action_to_idx=action_to_idx
        
        self.num_actions = len(self.action_object_info.keys())
        self.num_objects = len(self.object_to_idx.keys())
        
        for i,(k,v) in enumerate(self.action_object_info.items()):
            self.action_object_info[k]["action_idx"]=i
        

        # get paired links as neighboured joints
        self.links = [
            (0, 1, 2, 3, 4),
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),
        ]
 
        
        self.load_dataset()


        # Infor for rendering
        self.cam_intr[:2] = self.cam_intr[:2] * self.reduce_factor
        self.image_size = [int(1920 * self.reduce_factor), int(1080 * self.reduce_factor)] 

        if self.split=='train':
            try:
                self.env_r=lmdb.open(os.path.join(self.root,'lmdb_imgs',self.split),readonly=True,lock=False,readahead=False,meminit=False,\
                                map_size=(1024)**3,max_spare_txns=32,max_dbs=1000)
            except:
                self.env_r=None
                
        
    def load_dataset(self):
        subjects_infos = {}
        video_lens={}
        for subject in self.subjects:
            subject_info_path = os.path.join(self.info_root, "{}_info.txt".format(subject))         
            if not os.path.exists(subject_info_path):
                continue       
            subjects_infos[subject] = {}
            with open(subject_info_path, "r") as subject_f:
                raw_lines = subject_f.readlines()
                for line in raw_lines[3:]:
                    line = " ".join(line.split())
                    action, action_idx, length = line.strip().split(" ")
                    subjects_infos[subject][(action, action_idx)] = length

                    video_lens[(subject,action,action_idx)]=int(length) 

        skel_info = fhbutils.get_skeletons(self.skeleton_root, subjects_infos)
         
        with open(self.info_split, "r") as annot_f:
            lines_raw = annot_f.readlines()

        train_list, test_list, all_infos = fhbutils.get_action_train_test(lines_raw, subjects_infos) 
        
        assert self.split_type=='actions', 'FHB should use actions split'
        if self.split_type == "actions":
            if self.split == "train":
                sample_list = train_list
            elif self.split == "test":
                sample_list = test_list
            else:
                raise ValueError(
                    "Split {} not valid for fhbhands, should be [train|test|all]".format(self.split)
                )
        elif self.split_type == "subjects":
            if self.split == "train":
                subjects = ["Subject_1", "Subject_3", "Subject_4"]
            elif self.split == "test":
                subjects = ["Subject_2", "Subject_5", "Subject_6"]
            else:
                raise ValueError(f"Split {self.split} not in [train|test] for split_type subjects")
            self.subjects = subjects
            sample_list = all_infos
        elif self.split_type == "objects":
            sample_list = all_infos
        else:
            raise ValueError(
                "split_type should be in [action|objects|subjects], got {}".format(self.split_type)
            )
        if self.split_type != "subjects":
            self.subjects = ["Subject_1", "Subject_2", "Subject_3", "Subject_4", "Subject_5", "Subject_6"]
         
        image_names = []
        joints2d = []
        joints3d = []
        
        sample_infos = []

        #Add new features
        action_idxs, obj_idxs, object_loss_weights = [], [], [] 
        for subject, action_name, seq_idx, frame_idx in sample_list: 
            relative_img_path = os.path.join(subject, action_name, seq_idx, "color", self.rgb_template.format(frame_idx))#self.rgb_root, 
            skel = skel_info[subject][(action_name, seq_idx)][frame_idx]
            skel = skel[self.reorder_idx]

            skel_hom = np.concatenate([skel, np.ones([skel.shape[0], 1])], 1)
            skel_camcoords = self.cam_extr.dot(skel_hom.transpose()).transpose()[:, :3].astype(np.float32)
            
            image_names.append(relative_img_path)
            sample_infos.append(
                {
                    "subject": subject,
                    "action_name": action_name,
                    "seq_idx": seq_idx,
                    "frame_idx": frame_idx, 
                    "object_name": self.action_object_info[action_name]["object_name"]
                }
            )

            joints3d.append(skel_camcoords)
            hom_2d = np.array(self.cam_intr).dot(skel_camcoords.transpose()).transpose()
            skel2d = (hom_2d / hom_2d[:, 2:])[:, :2]
            
            joints2d.append(skel2d.astype(np.float32))

            #action related
            action_idxs.append(self.action_object_info[action_name]["action_idx"])

            #object related
            obj_idxs.append(self.action_object_info[action_name]["object_idx"])
            
        
        annotations = {
            "image_names": image_names,
            "joints2d": joints2d,
            "joints3d": joints3d,
            "sample_infos": sample_infos,

            "action_idxs":action_idxs,
            "obj_idxs":obj_idxs,

            "video_lens":video_lens,
        }
        
         
        
        # Get image paths
        self.image_names = annotations["image_names"]        

        self.joints2d = annotations["joints2d"]
        self.joints3d = annotations["joints3d"] #shd be with FPHA annotations
        
        self.sample_infos = annotations["sample_infos"]

        self.action_idxs = annotations["action_idxs"]
        self.obj_idxs=annotations["obj_idxs"] 
 
        self.video_lens=annotations["video_lens"]
        
         
        window_starts,fulls=fhbutils.get_seq_map(sample_infos=self.sample_infos,video_lens=self.video_lens,
                                                    ntokens_pose=self.ntokens_pose,
                                                    ntokens_action=self.ntokens_action,
                                                    spacing=self.spacing,
                                                    is_shifting_window=self.is_shifting_window)
        self.window_starts=window_starts
        self.fulls=fulls 
    
    def get_start_frame_idx(self, idx):
        idx=min(idx,len(self.window_starts)-1)
        return self.window_starts[idx]
            
    
    def get_dataidx(self, idx):
        idx=min(idx,len(self.fulls)-1)
        return self.fulls[idx]

    def open_seq_lmdb(self,idx):
        if self.split!='train':
            return 0,0
        idx=self.get_dataidx(idx)
        cur_seq_tag='_'.join(self.image_names[idx].split('/')[:-1])
        
        
        subdb=self.env_r.open_db(cur_seq_tag.encode('ascii'),create=False)
        txn=self.env_r.begin(write=False,db=subdb)
        return txn

    def get_image(self, idx, txn=None):
        idx = self.get_dataidx(idx)
        img_path = self.image_names[idx]
        
        if self.split=='train':
            buf=txn.get(img_path.encode('ascii'))
            img_flat=np.frombuffer(buf,dtype=np.uint8)
            img = img_flat.reshape(self.image_size[1],self.image_size[0],3).copy()
            img = Image.fromarray(img.astype(np.uint8)).convert("RGB") 
        else:
            img_path = os.path.join(self.rgb_root, img_path)
            img = Image.open(img_path).convert("RGB")
        return img
   


    def get_joints3d(self, idx):
        idx = self.get_dataidx(idx)
        joints = self.joints3d[idx]
        return joints / 1000

    def get_joints2d(self, idx):
        idx = self.get_dataidx(idx)
        joints = self.joints2d[idx] * self.reduce_factor
        return joints

 
    def get_abs_joints25d(self,idx):
        idx=self.get_dataidx(idx)
        joints=self.joints3d[idx]/1000
        joints[:,:2]=self.joints2d[idx]*self.reduce_factor     
         
        return joints


    
    

    def get_camintr(self, idx):
        idx = self.get_dataidx(idx)
        camintr = self.cam_intr
        return camintr.astype(np.float32)
 
    def get_action_idxs(self, idx):
        idx=self.get_dataidx(idx)
        action_idx=self.action_idxs[idx]
        return action_idx
    def get_obj_idxs(self,idx):
        idx=self.get_dataidx(idx)
        return self.obj_idxs[idx]

    

    def get_sample_info(self,idx):
        idx=self.get_dataidx(idx)
        sample_info=self.sample_infos[idx]
        return sample_info
    
    
    def get_future_frame_idx(self, cur_idx, fut_idx, spacing, verbose=False):
        cur_idx=self.get_dataidx(cur_idx)
        fut_idx=self.get_dataidx(fut_idx)
        cur_sample_info=self.sample_infos[cur_idx]
        fut_sample_info=self.sample_infos[fut_idx]

        if fut_sample_info["frame_idx"]-cur_sample_info["frame_idx"] != spacing:
            fut_idx=cur_idx
            not_padding=0
        else:
            not_padding=1 
        return fut_idx,not_padding


    def __len__(self):
        return len(self.window_starts)
 
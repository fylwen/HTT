from collections import defaultdict
import os

import numpy as np
from tqdm import tqdm  

 

def get_seq_map(sample_infos,video_lens,ntokens_pose, ntokens_action,spacing,is_shifting_window): 
    cur_sample = sample_infos[0] 
    pre_key = (cur_sample["subject"], cur_sample["action_name"], cur_sample["seq_idx"])
    seq_count = 0
    cur_seq_len=video_lens[pre_key] 

    window_starts = []
    full = []
    
    for sample_idx, sample_info in enumerate(sample_infos):
        cur_key = (sample_info["subject"], sample_info["action_name"], sample_info["seq_idx"])
        if pre_key != cur_key:
            pre_key=cur_key
            seq_count=0
            cur_seq_len=video_lens[pre_key] 
                
        if (not is_shifting_window and seq_count%(ntokens_action*spacing)<ntokens_pose*spacing) or \
            (is_shifting_window and seq_count%(ntokens_action*spacing)<spacing):
            window_starts.append(sample_idx)        
            
        full.append(sample_idx)
        seq_count += 1
    
    return window_starts, full
    

def get_action_train_test(lines_raw, subjects_info): 
    all_infos = []
    test_split = False
    test_samples = {}
    train_samples = {}
    for line in lines_raw[1:]:
        if line.startswith("Test"):
            test_split = True
            continue
        subject, action_name, action_seq_idx = line.split(" ")[0].split("/")
        action_idx = line.split(" ")[1].strip()  # Action classif index
        frame_nb = int(subjects_info[subject][(action_name, action_seq_idx)])
        for frame_idx in range(frame_nb): 
            sample_info = (subject, action_name, action_seq_idx, frame_idx)
            if test_split:
                test_samples[sample_info] = action_idx
            else:
                train_samples[sample_info] = action_idx
            all_infos.append(sample_info)
    test_nb = len(np.unique(list((sub, act_n, act_seq) for (sub, act_n, act_seq, _) in test_samples), axis=0))
    train_nb = len(np.unique(list((sub, act_n, act_seq) for (sub, act_n, act_seq, _) in train_samples), axis=0))  
    return train_samples, test_samples, all_infos
  


def get_skeletons(skeleton_root, subjects_info, use_cache=True): 
    skelet_dict = defaultdict(dict)
    for subject, samples in tqdm(subjects_info.items(), desc="subj"):
        for (action, seq_idx) in tqdm(samples, desc="sample"):
            skeleton_path = os.path.join(skeleton_root, subject, action, seq_idx, "skeleton.txt") 
            skeleton_vals = np.loadtxt(skeleton_path)
            if len(skeleton_vals):
                assert np.all(
                    skeleton_vals[:, 0] == list(range(skeleton_vals.shape[0]))
                ), "row idxs should match frame idx failed at {}".format(skeleton_path)
                skelet_dict[subject][(action, seq_idx)] = skeleton_vals[:, 1:].reshape(
                    skeleton_vals.shape[0], 21, -1
                )
            else: 
                skelet_dict[subject, action, seq_idx] = skeleton_vals 
    return skelet_dict

def get_action_object_infos(path_action_object_info):
    action_obj_info={}
    obj_to_idx={}
    action_to_idx={}
    with open(path_action_object_info, "r") as f:
        raw_lines = f.readlines()
        for line in raw_lines[1:]:
            line = " ".join(line.split())
            action_idx, action_name, object_name, object_pose, scenario = line.strip().split(" ")
            
            if object_name in obj_to_idx.keys():
                co_id=obj_to_idx[object_name]
            else:
                co_id=len(obj_to_idx.keys())
                obj_to_idx[object_name]=co_id
            if not (action_name) in action_to_idx.keys():
                action_to_idx[action_name]=int(action_idx)-1

            action_obj_info[action_name]={'action_idx': int(action_idx)-1,'object_idx':co_id,'object_name':object_name,
                    'object_pose_tag':object_pose,'scenario': scenario}
    
    return action_obj_info, obj_to_idx, action_to_idx

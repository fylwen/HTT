import glob
import os,cv2
import numpy as np
 
def convert_to_480_270_imgs():
    list_dirs=glob.glob('./*/*/*/cam4/')
    for cdir in list_dirs: 
        out_dir=os.path.join(cdir,'rgb480_270')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        list_imgs=glob.glob(os.path.join(cdir,'rgb','*.png'))
        for im_id in range(0,len(list_imgs)):
            path_cimg=os.path.join(cdir,'rgb','{:06d}.png'.format(im_id)) 
            cimg=cv2.imread(path_cimg)
            assert cimg is not None, 'cimg shd not be None'
            cimg=cv2.resize(cimg,(480,270))
            cv2.imwrite(os.path.join(cdir,'rgb480_270','{:06d}.png'.format(im_id)),cimg)



def get_action_idx_to_tag():
    string_idx_tag="0 background\n1 grab book\n2 grab espresso\n3 grab lotion\n4 grab spray\n5 grab milk\n6 grab cocoa\n"
    string_idx_tag+="7 grab chips\n8 grab cappuccino\n9 place book\n10 place espresso\n11 place lotion\n12 place spray\n13 place milk\n"
    string_idx_tag+="14 place cocoa\n15 place chips\n16 place cappuccino\n17 open lotion\n18 open milk\n19 open chips\n20 close lotion\n"
    string_idx_tag+="21 close milk\n22 close chips\n23 pour milk\n24 take out espresso\n25 take out cocoa\n26 take out chips\n27 take out cappuccino\n"
    string_idx_tag+="28 put in espresso\n29 put in cocoa\n30 put in cappuccino\n31 apply lotion\n32 apply spray\n33 read book\n34 read espresso\n"
    string_idx_tag+="35 spray spray\n36 squeeze lotion"
    string_idx_tag=string_idx_tag.split('\n') 
    list_idx_tag=[]
    for aid,tag in enumerate(string_idx_tag):
        action_name=' '.join(tag.split(' ')[1:])
        list_idx_tag.append(action_name) 
    return list_idx_tag

def get_object_tag_to_idx():
    string_idx_tag="0 background\n1 book\n2 espresso\n3 lotion\n4 spray\n5 milk\n6 cocoa\n7 chips\n8 cappuccino"
    string_idx_tag=string_idx_tag.split('\n') 
    dict_tag_idx={}
    for oid,tag in enumerate(string_idx_tag):
        object_name=tag.split(' ')[1]
        dict_tag_idx[object_name]=oid 
    return dict_tag_idx

def get_segments_actions_objects_info_from_files(path_dataset='./'):
    list_action_name=get_action_idx_to_tag()
    dict_object_name_to_idx=get_object_tag_to_idx()
    frame_segments={'train':[],'val':[], 'test':[]}
    for split_tag in frame_segments.keys():
        with open(os.path.join(path_dataset,'./action_labels/action_{:s}.txt'.format(split_tag)),'r') as f:
            segs=f.readlines()[1:] 
            for cline in segs:
                cline=cline.strip('\n').split(' ')
                if split_tag=='test':
                    seg_idx,action_idx=int(cline[0]),int(cline[-1])
                    start_frame_idx,end_frame_idx=int(cline[2]),int(cline[3])
                else:
                    seg_idx,action_idx=int(cline[0]),int(cline[2])
                    start_frame_idx,end_frame_idx=int(cline[3]),int(cline[4])
                tag_subject,tag_scene,tag_sequence=cline[1].split('/')
                action_name=list_action_name[action_idx]
                object_name=action_name.split(' ')[-1]
                object_idx=dict_object_name_to_idx[object_name]


                #Here for action and object, rmv bg, start from 0
                #segment_idx start from 0
                cinfo={'segment_idx':seg_idx-1,'subject':tag_subject,'scene':tag_scene,'sequence':tag_sequence,
                'action_idx':action_idx-1,'action_name':action_name, 'object_idx':object_idx-1,'object_name':object_name,
                'start_idx':start_frame_idx,'end_idx':end_frame_idx}

                frame_segments[split_tag].append(cinfo)
        
    #rmv bg
    del dict_object_name_to_idx['background']
    for k in dict_object_name_to_idx.keys():
        dict_object_name_to_idx[k]-=1
    
    dict_action_name_to_idx={}
    for aid, action_name in enumerate(list_action_name[1:]):
        dict_action_name_to_idx[action_name]=aid

    return frame_segments,dict_action_name_to_idx,dict_object_name_to_idx

    



def read_text(text_path, offset=0, half=0):
    with open(text_path, 'r') as txt_file:
        data = txt_file.readline().split(" ")
        data = list(filter(lambda x: x != "", data))

    if half:
        data_list = np.array(data)[offset:half].tolist() + np.array(data)[half+offset:].tolist()
        return np.array(data_list).reshape((-1, 3)).astype(np.float32)
    else:
        return np.array(data)[offset:].reshape((-1, 3)).astype(np.float32)

def get_skeleton(path_hand_pose):
    return read_text(path_hand_pose,offset=1,half=64)

 


def get_seq_map(frame_segments,sample_infos,ntokens_pose, ntokens_action, spacing,is_shifting_window):
    window_starts=[]
    full = []
    
    for sample_idx, sample_info in enumerate(sample_infos):
        cseg=frame_segments[sample_info['seq_idx']]
        if sample_info["frame_idx"]==cseg["start_idx"]:        
            seq_count=0
            cur_seq_len=cseg["end_idx"]-cseg["start_idx"]+1 
                
        if (not is_shifting_window and seq_count%(ntokens_action*spacing)<ntokens_pose*spacing) or \
            (is_shifting_window and seq_count%(ntokens_action*spacing)<spacing):
                window_starts.append(sample_idx)    
        full.append(sample_idx)
        seq_count += 1
    
    return window_starts, full

 

 
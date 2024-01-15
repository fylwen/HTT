import glob
import os,cv2
import numpy as np
 
from joblib import Parallel, delayed
from PIL import Image
from datasets import fhbhands, h2ohands
import lmdb
#Resize original fpha imgs to (480,270)
def resize_imgs_to_480_270_fpha(fhb_root="../fpha/"):

    fhb_rgb_src = os.path.join(fhb_root, "Video_files")
    fhb_rgb_dst = os.path.join(fhb_root, "Video_files_480")


    def convert(src, dst, out_size=(480, 270)):
        dst_folder = os.path.dirname(dst)
        os.makedirs(dst_folder, exist_ok=True)
        if not os.path.exists(dst):
            img = Image.open(src)
            dest_img = img.resize(out_size, Image.BILINEAR)
            dest_img.save(dst)

    subjects = [f"Subject_{subj_idx}" for subj_idx in range(1, 7)]
    # Gather all frame paths to convert
    for subj in subjects:
        frame_pairs = []
        subj_path = os.path.join(fhb_rgb_src, subj)
        actions = sorted(os.listdir(subj_path))
        for action in actions:
            action_path = os.path.join(subj_path, action)
            sequences = sorted(os.listdir(action_path))
            for seq in sequences:
                seq_path = os.path.join(action_path, seq, "color")
                frames = sorted(os.listdir(seq_path))
                for frame in frames:
                    frame_path_src = os.path.join(seq_path, frame)
                    frame_path_dst = os.path.join(fhb_rgb_dst, subj, action, seq, "color", frame)
                    frame_pairs.append((frame_path_src, frame_path_dst))

        # Resize all images
        nworkers=10
        print(f"Launching conversion for {len(frame_pairs)}")
        Parallel(n_jobs=nworkers, verbose=5)(
            delayed(convert)(frame_pair[0], frame_pair[1]) for frame_pair in frame_pairs
        )
#Resize original h2o imgs to (480,270)
def resize_imgs_to_480_270(h2o_root="../h2o"):
    list_dirs=glob.glob(os.path.join(h2o_root,'./*/*/*/cam4/'))
    for cdir in list_dirs:
        out_dir=os.path.join(cdir,'rgb480_270')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        list_imgs=glob.glob(os.path.join(cdir,'rgb','*.png'))
        for im_id in range(0,len(list_imgs)):
            path_cimg=os.path.join(cdir,'rgb','{:06d}.png'.format(im_id))
            cimg=cv2.imread(path_cimg)
            cimg=cv2.resize(cimg,(480,270))
            cv2.imwrite(os.path.join(cdir,'rgb480_270','{:06d}.png'.format(im_id)),cimg)


#Use lmdb to save image, and facilitate training
def convert_dataset_split_to_lmdb(dataset_name,dataset_folder,split):
    input_res = (480, 270)
    if dataset_name == "fhbhands":
        pose_dataset = fhbhands.FHBHands(dataset_folder=dataset_folder,
                                        split=split,
                                        ntokens_pose=16,
                                        ntokens_action=128,
                                        spacing=2,
                                        is_shifting_window=True,
                                        split_type="actions",)
                                                
    elif dataset_name=='h2ohands':
        pose_dataset = h2ohands.H2OHands(dataset_folder=dataset_folder,
                                        split=split,
                                        ntokens_pose=16,
                                        ntokens_action=128,
                                        spacing=2,
                                        is_shifting_window=True,
                                        split_type="actions",)



    image_names = pose_dataset.image_names
    sample_infos=pose_dataset.sample_infos


    rgb_root = pose_dataset.rgb_root
    image_path = os.path.join(rgb_root,image_names[0])
    data_size_per_img= np.array(Image.open(image_path).convert("RGB")).nbytes 
    data_size=data_size_per_img*len(image_names)

    dir_lmdb=os.path.join(dataset_folder,'lmdb_imgs',split)
    if not os.path.exists(dir_lmdb):
        os.makedirs(dir_lmdb)

    env = lmdb.open(dir_lmdb,map_size=data_size*10,max_dbs=1000)
    pre_seq_tag=''
    commit_interval=100
    for idx in range(0,len(image_names)):
        if dataset_name=='fhbhands':
            cur_seq_tag='_'.join(image_names[idx].split('/')[:-1])
        else:
            cur_seq_tag='{:04d}'.format(sample_infos[idx]["seq_idx"])
        if cur_seq_tag!=pre_seq_tag:
            pre_seq_tag=cur_seq_tag
            print(cur_seq_tag)

            if idx>0:
                txn.commit()
        
            subdb=env.open_db(cur_seq_tag.encode('ascii'))
            txn=env.begin(db=subdb,write=True)

        key_byte = image_names[idx].encode('ascii')
        image_path = os.path.join(rgb_root,image_names[idx])
        data = np.array(Image.open(image_path).convert("RGB"))
        print(idx,image_names[idx])
        txn.put(key_byte,data)

        if (idx+1)%commit_interval==0:
            txn.commit()
            txn=env.begin(db=subdb,write=True)

    txn.commit()
    env.close()
    
if __name__ == '__main__':
    # resize_imgs_to_480_270_fpha(fhb_root='/media/mldadmin/home/s123mdg31_07/Datasets/FPHAB')
    convert_dataset_split_to_lmdb('fhbhands', '~/Datasets/FPHAB')
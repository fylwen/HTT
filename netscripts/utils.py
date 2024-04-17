import os
from libyana.visutils.viz2d import visualize_joints_2d


from models.utils import torch2numpy
import numpy as np
import cv2
import matplotlib.pyplot as plt
 
 

def plot_cframe_est_gt(trj_est,trj_gt,frame_id,ax3d,title):     
    ax3d_xmin,ax3d_xmax=min(trj_est[:,:,0].min(),trj_gt[:,:,0].min()),max(trj_est[:,:,0].max(),trj_gt[:,:,0].max())
    ax3d_ymax,ax3d_ymin=min(trj_est[:,:,1].min(),trj_gt[:,:,1].min()),max(trj_est[:,:,1].max(),trj_gt[:,:,1].max())
    ax3d_zmin,ax3d_zmax=min(trj_est[:,:,2].min(),trj_gt[:,:,2].min()),max(trj_est[:,:,2].max(),trj_gt[:,:,2].max())
    ax3d.set_xlim(ax3d_xmin-0.005,ax3d_xmax+0.005)            
    ax3d.set_zlim(ax3d_ymin-0.005,ax3d_ymax+0.005)            
    ax3d.set_ylim(ax3d_zmin-0.005,ax3d_zmax+0.005)
    
    ax3d.set_xticks([])
    ax3d.set_yticks([])
    ax3d.set_zticks([]) 

    cgt,cest=trj_gt[frame_id],trj_est[frame_id]
    link= [[0, 1, 2, 3, 4],[0, 5, 6, 7, 8],[0, 9, 10, 11, 12],[0, 13, 14, 15, 16],[0, 17, 18, 19, 20]]
    for l in link:
        ax3d.plot(trj_gt[frame_id,l,0],trj_gt[frame_id,l,2],trj_gt[frame_id,l,1],alpha=0.8,c=(0,1.,0),linewidth=0.75)#0.5)
    for l in link:
        ax3d.plot(trj_est[frame_id,l,0],trj_est[frame_id,l,2],trj_est[frame_id,l,1],alpha=1.0,c=(0,0,1.),linewidth=0.75)#0.5)

def project_joints_to_2d(joints_3d,cam_intr):
    verts_hom2d = cam_intr.dot(joints_3d.transpose()).transpose()
    verts_proj = (verts_hom2d / verts_hom2d[:, 2:])[:, :2]
    return verts_proj



def draw_2d_3d_pose(batch_seq_gt_cam,batch_seq_est_cam,batch_seq_imgs,sample_id,frame_id,cam_intr, num_rows, num_cols, is_single_hand, title):
    ctrj_gt_cam,ctrj_est_cam=batch_seq_gt_cam[sample_id],batch_seq_est_cam[sample_id]
    ctrj_gt_cam2d,ctrj_est_cam2d=ctrj_gt_cam.copy(),ctrj_est_cam.copy()
    
    offset=0.025

    #img
    axi=plt.subplot2grid((num_rows,num_cols),(0,0),colspan=2) 
    axi.axis("off")
    if not (title is None):
        axi.set_title(title,fontsize=4, pad=-10)  
    
    simg=batch_seq_imgs[sample_id,frame_id].copy()
    axi.imshow(simg)
    cframe_gt_joints2d=project_joints_to_2d(ctrj_gt_cam2d[frame_id],cam_intr)
    cframe_est_joints2d=project_joints_to_2d(ctrj_est_cam2d[frame_id],cam_intr)   

    visualize_joints_2d(axi, cframe_gt_joints2d[:21], alpha=0.8,linewidth=.7,scatter=False, joint_idxs=False,color=[(0,1.,0)]*5)
    visualize_joints_2d(axi, cframe_est_joints2d[:21], alpha=1,linewidth=.7,scatter=False, joint_idxs=False,color=[(0,0,1.)]*5)

    try:
        visualize_joints_2d(axi, cframe_gt_joints2d[21:], alpha=0.8,linewidth=.7,scatter=False, joint_idxs=False,color=[(0,1.,0)]*5)
        visualize_joints_2d(axi, cframe_est_joints2d[21:], alpha=1,linewidth=.7, scatter=False, joint_idxs=False,color=[(0,0,1.)]*5)#[(1.,0.,0.)]*5)
    except:
        x=0
    
    
    dev=3 if is_single_hand else 4 
    ax3d=plt.subplot(num_rows, num_cols, dev, projection='3d') 
    if is_single_hand:
        trj_est=ctrj_est_cam[:,:21]
        trj_gt=ctrj_gt_cam[:,:21] 
        plot_cframe_est_gt(trj_est,trj_gt,frame_id,ax3d,title="Camera Space")
    else:
        trj_est=ctrj_est_cam[:,21:]
        trj_gt=ctrj_gt_cam[:,21:]
 
        plot_cframe_est_gt(trj_est,trj_gt,frame_id,ax3d,title=None)#,gt_colors,est_colors)
    ax3d.view_init(30,210)

    if not is_single_hand:
        ax3d=plt.subplot(num_rows, num_cols, dev-1, projection='3d')
        trj_est=ctrj_est_cam[:,:21] 
        trj_gt=ctrj_gt_cam[:,:21]
        plot_cframe_est_gt(trj_est,trj_gt,frame_id,ax3d,title="Camera Space")#,gt_colors,est_colors)



def vis_sample(batch_seq_gt_pose_in_cam, batch_seq_est_pose_in_cam, batch_seq_padding, cam_intr, batch_seq_imgs,
                batch_gt_action, batch_pred_action, sample_id, is_single_hand,dir_out):
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    batch_seq_gt_pose_in_cam=torch2numpy(batch_seq_gt_pose_in_cam)
    batch_seq_est_pose_in_cam=torch2numpy(batch_seq_est_pose_in_cam)
    batch_seq_padding=torch2numpy(batch_seq_padding)
    cam_intr=torch2numpy(cam_intr)
    batch_seq_imgs=torch2numpy(batch_seq_imgs)
    batch_pred_action=torch2numpy(batch_pred_action)
    batch_gt_action=torch2numpy(batch_gt_action)   
     
    batch_size, len_frames=batch_seq_gt_pose_in_cam.shape[0:2]
    
    end_frame=len_frames
    for frame_id in range(0,len_frames):
        if batch_seq_padding[sample_id,frame_id]<1e-4:
            end_frame=frame_id
            break
            
    title="Action Est: {:s}/GT: {:s} \n Pose GT in Green, Est in Blue".format(batch_pred_action[sample_id],batch_gt_action[sample_id])
    for frame_id in range(0,end_frame):
        
        num_cols=3 if is_single_hand else 4
        fig = plt.figure(figsize=(num_cols,1.2))
        draw_2d_3d_pose(batch_seq_gt_pose_in_cam,batch_seq_est_pose_in_cam,batch_seq_imgs,sample_id,frame_id,cam_intr, \
                    num_rows=1, num_cols=num_cols,is_single_hand=is_single_hand, title=title)
        fig.savefig(os.path.join(dir_out,"out_{:03d}.png".format(frame_id)), dpi=300)
        plt.close(fig)

    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')     
    cimg=cv2.imread(os.path.join(dir_out,"out_{:03d}.png".format(0)))
    videoWriter = cv2.VideoWriter(os.path.join(dir_out,'out.avi'), fourcc, 5, (cimg.shape[1],cimg.shape[0]))  
    videoWriter.write(cimg)
    for frame_id in range(1,end_frame):
        cimg=cv2.imread(os.path.join(dir_out,"out_{:03d}.png".format(frame_id)))
        videoWriter.write(cimg)
    
    videoWriter.release()


def draw_single_run(data,tag):    
    data['thresholds']*=100
    fig = plt.figure(figsize=(6, 6), facecolor='white')
    ax = fig.add_axes([0.2, 0.1, 0.7, 0.8])

    ax.plot(data['thresholds'], data['pck_curve'], linewidth=3, markersize=10, color='#0000FF')            

    print(tag,"-")
    for t, v in zip(data['thresholds'],data['pck_curve']):
        if t in [1,1.5,2,2.5,3,3.5,4,4.5,5]:
            print(t,'cm:', v)
        
    x_range = np.linspace(0,int(data['thresholds'][-1]),int(data['thresholds'][-1])+1)
    ax.set_xticks(x_range)
    ax.set_xlim(0,int(data['thresholds'][-1]))
    ax.set_xlabel('error/cm', fontsize=18, family='serif', fontstyle='italic', labelpad=2.)
    y_range = np.linspace(0,1.,6)
    ax.set_yticks(y_range)
    ax.set_ylim([0,1.01])
    ax.set_ylabel('pck', fontsize=18, family='serif', fontstyle='italic', labelpad=2.)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('serif') for label in labels]
    [label.set_fontsize(11) for label in labels]
    # fig.align_labels()
    plt.subplots_adjust(left=0.15, bottom=0.128)
    plt.tight_layout()
    
    title_tag='Auc {:.3f}, Mean EPE {:.2f} mm'.format(data['auc'], data['epe_mean']*1000)
    plt.title(title_tag, loc='center', family='serif', fontsize=20, pad=10)
    
    plt.grid(ls='--')
    plt.savefig(f'./{tag}_3Dhand.png')
    np.savez(f'./{tag}_3Dhand.npz',x=data['thresholds'],y=data['pck_curve'])
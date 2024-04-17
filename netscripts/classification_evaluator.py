import numpy as np
from models.utils import torch2numpy

class FrameClassificationEvaluator:
    """ Util class for evaluation networks.
    """

    def __init__(self, label_info):
        # init empty data storage
        self.label_info=label_info
        self.num_labels=len(self.label_info.keys()) 
        self.seq_results={}        
        self.name_labels={}
        for k,v in self.label_info.items():
            self.name_labels[v]=k 
        self.count_matrix=np.zeros((self.num_labels,self.num_labels))

    def feed(self, gt_labels, pred_labels, weights, verbose=False):   
        
        gt_labels=torch2numpy(gt_labels).flatten()
        pred_labels=torch2numpy(pred_labels).flatten()
        weights=torch2numpy(weights).flatten()
        
        for idx in range(0,gt_labels.shape[0]):
            if weights[idx]>1e-4:# and (idx+1==gt_labels.shape[0] or weights[idx+1]<1e-4):
                self.count_matrix[gt_labels[idx],pred_labels[idx]]+=1

                
                
    def get_recall_rate(self, path_to_save=None, verbose=False):        
        result={}
        
        result['total_samples']=np.sum(self.count_matrix)
        result['total_tp']=np.sum(np.array([self.count_matrix[i,i] for i in range(0,self.num_labels)]))
        result['recall_rate_mean']=result['total_tp']/result['total_samples']

        num_items_per_label=np.sum(self.count_matrix,axis=1,keepdims=True)
        num_items_per_label=np.where(num_items_per_label==0,1,num_items_per_label)
        distribution_matrix=self.count_matrix/num_items_per_label
        
        if path_to_save is not None:
            np.savez(path_to_save,action_idx_to_name=self.name_labels,distribution_matrix=distribution_matrix)
        return result
    
class SequenceClassificationEvaluator:

    def __init__(self, label_info):
        self.label_info=label_info
        self.num_labels=len(self.label_info.keys())

        self.video_seq_results={}
        self.name_labels={}
        for k,v in self.label_info.items():
            self.name_labels[v]=k
            
        self.count_matrix_video_seq=np.zeros((self.num_labels,self.num_labels))
        self.count_matrix_network_seq=np.zeros((self.num_labels,self.num_labels))

    def feed(self, gt_labels, pred_results, batch_samples, seq_len, pred_is_label=False, verbose=False):               
        gt_labels=torch2numpy(gt_labels).flatten()
        pred_results=torch2numpy(pred_results)
        if pred_is_label:
            pred_results=pred_results.flatten()

        info_subjects=batch_samples['subject'][0::seq_len]
        info_action_names=batch_samples['action_name'][0::seq_len]
        info_seq_idx=torch2numpy(batch_samples['seq_idx'][0::seq_len])
        

        

        for seq_id in range(0,len(info_subjects)):
            c_pred=pred_results[seq_id]
            c_tag=(info_subjects[seq_id],info_action_names[seq_id],info_seq_idx[seq_id],self.label_info[info_action_names[seq_id]])
 
            if not (c_tag in self.video_seq_results.keys()):
                self.video_seq_results[c_tag]=np.zeros((self.num_labels,))
            
            if pred_is_label:
                self.video_seq_results[c_tag][c_pred]+=1
                self.count_matrix_network_seq[gt_labels[seq_id],c_pred]+=1
            else:
                self.video_seq_results[c_tag]+=c_pred
                self.count_matrix_network_seq[gt_labels[seq_id]]+=c_pred
                
             
                
    def get_recall_rate(self, aggregate_sequence=True, verbose=False):        
        result={}
        if aggregate_sequence:
            for k,v in self.video_seq_results.items():
                gt_id=k[-1]
                pred_id=np.argmax(v)
                self.count_matrix_video_seq[gt_id,pred_id]+=1
                
                
            
            
            num_items_per_label=np.sum(self.count_matrix_video_seq,axis=1,keepdims=True)
            num_items_per_label=np.where(num_items_per_label==0,1,num_items_per_label)
            distribution_matrix_video_seq=self.count_matrix_video_seq/num_items_per_label

            result['video_seq_total']=np.sum(self.count_matrix_video_seq)
            result['video_seq_tp']=np.sum(np.array([self.count_matrix_video_seq[i,i] for i in range(0,self.num_labels)]))
            result['video_seq_recall_rate_mean']=result['video_seq_tp']/result['video_seq_total']


        result['total_samples']=np.sum(self.count_matrix_network_seq)
        result['total_tp']=np.sum(np.array([self.count_matrix_network_seq[i,i] for i in range(0,self.num_labels)]))
        result['recall_rate_mean']=result['total_tp']/result['total_samples']

        num_items_per_label=np.sum(self.count_matrix_network_seq,axis=1,keepdims=True)
        num_items_per_label=np.where(num_items_per_label==0,1,num_items_per_label)
        distribution_matrix_network_seq=self.count_matrix_network_seq/num_items_per_label
        
            
        return result
    

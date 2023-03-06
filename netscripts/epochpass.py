from tqdm import tqdm
import torch


from libyana.evalutils.avgmeter import AverageMeters
from netscripts import position_evaluator
from netscripts.classification_evaluator import SequenceClassificationEvaluator, FrameClassificationEvaluator
from netscripts.position_evaluator import ZimEval
from netscripts.utils import draw_single_run

def epoch_pass(
    loader,
    model,
    train,
    optimizer,
    scheduler,
    lr_decay_gamma,
    use_multiple_gpu,
    tensorboard_writer,
    dataset_action_info,
    dataset_object_info,
    ntokens,
    aggregate_sequence,
    is_single_hand,
    is_demo,
    epoch,

):
    if train:
        prefix = "train"
    else:
        prefix = "val"

    if is_single_hand:
        evaluators = {"joints3d": ZimEval(num_kp=21),"joints3d_cent":ZimEval(num_kp=20)}  
    else:
        evaluators = {"left_joints3d": ZimEval(num_kp=21),"left_joints3d_cent":ZimEval(num_kp=20), \
            "right_joints3d": ZimEval(num_kp=21),"right_joints3d_cent":ZimEval(num_kp=20)}
    
    action_evaluator =  SequenceClassificationEvaluator(dataset_action_info) 
    objlabel_evaluator = FrameClassificationEvaluator(dataset_object_info)
    
    
    model.eval()
    avg_meters = AverageMeters()
     
     
    for batch_idx, batch in enumerate(tqdm(loader)): 
        if train:
            loss, results, losses = model(batch)  
        else:
            with torch.no_grad():
                loss, results, losses = model(batch)

        if use_multiple_gpu:
            loss=loss.mean()
            for k,v in losses.items():
                if v is not None:
                    v=v.mean() 
        if train:
            if torch.isnan(loss):
                raise ValueError(f"Loss made of {losses} became nan!")
            optimizer.zero_grad()                
            loss.backward()            
            optimizer.step()
 
 
        for loss_name, loss_val in losses.items():
            if loss_val is not None:
                avg_meters.add_loss_value(loss_name, loss_val.mean().item())
                
        if not train:
            if is_single_hand:
                position_evaluator.feed_evaluators_hand(evaluators, pred=results["pred_joints3d"], 
                                                        gt=results["gt_joints3d"], 
                                                        weights=batch["not_padding"],
                                                        tag="", center_idx=0)  
            else:
                position_evaluator.feed_evaluators_hand(evaluators, pred=results["pred_joints3d"][:,:21], 
                                                        gt=results["gt_joints3d"][:,:21], 
                                                        weights=batch["not_padding"],
                                                        tag="left_", center_idx=0)  

                position_evaluator.feed_evaluators_hand(evaluators, pred=results["pred_joints3d"][:,21:], 
                                                        gt=results["gt_joints3d"][:,21:], 
                                                        weights=batch["not_padding"], 
                                                        tag="right_", center_idx=0)  
            
            
            action_evaluator.feed(gt_labels=results["action_gt_labels"],
                                    pred_results=results["action_reg_possibilities"],#results['action_pred_labels'],
                                    batch_samples=batch["sample_info"],
                                    pred_is_label=False,#True
                                    seq_len=ntokens,) 

            objlabel_evaluator.feed(gt_labels=results["obj_gt_labels"],
                                pred_labels=results["obj_pred_labels"],
                                weights=batch["not_padding"])

            
            if is_demo:
                from netscripts.utils import vis_sample
                from datasets.queries import BaseQueries 
                
                num_joints=21 if is_single_hand else 42
                batch_seq_gt_pose_in_cam=results["gt_joints3d"].view(-1,ntokens,num_joints,3)
                batch_seq_est_pose_in_cam=results["pred_joints3d"].view(-1,ntokens,num_joints,3)

                batch_seq_imgs=batch[BaseQueries.IMAGE]
                batch_seq_imgs=batch_seq_imgs.reshape((-1,ntokens,)+batch_seq_imgs.shape[1:])
                batch_seq_padding=batch['not_padding'].reshape(-1,ntokens)
                cam_intr=batch[BaseQueries.CAMINTR][0]
 

                batch_gt_action,batch_pred_action=[],[]
                for aid in results['action_gt_labels'].detach().cpu().numpy():
                    batch_gt_action.append(action_evaluator.name_labels[aid])
                for aid in results['action_pred_labels'].detach().cpu().numpy():
                    batch_pred_action.append(action_evaluator.name_labels[aid])



                vis_sample(batch_seq_gt_pose_in_cam, batch_seq_est_pose_in_cam, batch_seq_padding, cam_intr, batch_seq_imgs,
                        batch_gt_action, batch_pred_action, sample_id=0, is_single_hand=is_single_hand,dir_out='./ws/vis/')
                
                exit(0)



         
         

    save_dict = {}
    if train and lr_decay_gamma and scheduler is not None:
        save_dict['learning_rate']=scheduler.get_last_lr()[0]
        scheduler.step()
   
    
    for loss_name, avg_meter in avg_meters.average_meters.items():
        loss_val = avg_meter.avg
        save_dict[loss_name] = loss_val
    
    
    evaluator_results={}
    if not train: 
        evaluator_results = position_evaluator.parse_evaluators(evaluators)
        for eval_name, eval_res in evaluator_results.items():
            for met in ["epe_mean", "auc"]:
                loss_name = f"{eval_name}_{met}"
                # Filter nans
                if eval_res[met] == eval_res[met]:
                    save_dict[loss_name] = eval_res[met]
            draw_single_run(eval_res,eval_name)
        # Filter out Nan pck curves
        evaluator_results = {eval_name: res for eval_name, res in evaluator_results.items() if res["epe_mean"] == res["epe_mean"]}

        action_result=action_evaluator.get_recall_rate(aggregate_sequence=aggregate_sequence,verbose=False)
        for k,v in action_result.items():
            save_dict['action_'+k]=v 
        objlabel_result=objlabel_evaluator.get_recall_rate()
        for k,v in objlabel_result.items():
            save_dict['objlabel_'+k]=v
        
        
        if is_single_hand:
            print("hand- MEPE (camera space, mm)/AUC(0-80mm): {:.2f}/{:.3f}".format(save_dict["joints3d_epe_mean"]*1000,save_dict["joints3d_auc"]))
            print("hand- MEPE-RA (camera space, mm)/AUC(0-50mm): {:.2f}/{:.3f}".format(save_dict["joints3d_cent_epe_mean"]*1000,save_dict["joints3d_cent_auc"]))
        else:
            print("H2O for MEPE in camera space and Action recall rate, in main text we refer to result evaluated by the H2O Comopetition Codalab.")
            for k in ["left","right"]:
                print(k+"hand- MEPE (camera space, mm)/AUC(0-80mm): {:.2f}/{:.3f}".format(save_dict[f"{k}_joints3d_epe_mean"]*1000,save_dict[f"{k}_joints3d_auc"]))
                print(k+"hand- MEPE-RA (camera space, mm)/AUC(0-50mm): {:.2f}/{:.3f}".format(save_dict[f"{k}_joints3d_cent_epe_mean"]*1000,save_dict[f"{k}_joints3d_cent_auc"]))

        print("action recall-rate on video, TP: {:d}, Total: {:d}, recall rate {:.2f}".format(int(save_dict["action_video_seq_tp"]),int(save_dict["action_video_seq_total"]),\
                        save_dict["action_video_seq_recall_rate_mean"]*100))

    
    if not tensorboard_writer is None:
        for k,v in save_dict.items():
                if k in losses.keys() or k in ['learning_rate','total_loss']:
                    print(prefix+'/'+k,v,epoch)
                    tensorboard_writer.add_scalar(prefix+'/'+k, v, epoch)
    

    
    return save_dict, avg_meters, evaluator_results


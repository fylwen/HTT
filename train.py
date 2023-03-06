import argparse
from datetime import datetime

from matplotlib import pyplot as plt
import torch
from tqdm import tqdm

from libyana.exputils.argutils import save_args
from libyana.modelutils import modelio
from libyana.modelutils import freeze
from libyana.randomutils import setseeds

from datasets import collate
from models.htt import TemporalNet
from netscripts import epochpass
from netscripts import reloadmodel, get_dataset 
from torch.utils.tensorboard import SummaryWriter
from netscripts.get_dataset import DataLoaderX 
plt.switch_backend("agg")
print('********')
print('Lets start')

def collate_fn(seq, extend_queries=[]):
    return collate.seq_extend_flatten_collate(seq,extend_queries)
    
def main(args):
    setseeds.set_all_seeds(args.manual_seed)
    # Initialize hosting
    now = datetime.now()
    experiment_tag = args.experiment_tag
    exp_id = f"{args.cache_folder}"+experiment_tag+"/"

    # Initialize local checkpoint folder
    save_args(args, exp_id, "opt")
    board_writer=SummaryWriter(log_dir=exp_id) 
    



    print("**** Lets train on", args.train_dataset, args.train_split)
    train_dataset, _ = get_dataset.get_dataset_htt(
        args.train_dataset,
        dataset_folder=args.dataset_folder,
        split=args.train_split, 
        no_augm=False,
        scale_jittering=args.scale_jittering,
        center_jittering=args.center_jittering,
        ntokens_pose=args.ntokens_pose,
        ntokens_action=args.ntokens_action,
        spacing=args.spacing,
        is_shifting_window=False,
        split_type="actions"
    )


    loader = DataLoaderX(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn= collate_fn,
    )
    
    dataset_info=train_dataset.pose_dataset

    #Re-load pretrained weights  
    model= TemporalNet(dataset_info=dataset_info,
                is_single_hand=args.train_dataset!="h2ohands",
                transformer_num_encoder_layers_action=args.enc_action_layers,
                transformer_num_encoder_layers_pose=args.enc_pose_layers,
                transformer_d_model=args.hidden_dim,
                transformer_dropout=args.dropout,
                transformer_nhead=args.nheads,
                transformer_dim_feedforward=args.dim_feedforward,
                transformer_normalize_before=True,
                lambda_action_loss=args.lambda_action_loss,
                lambda_hand_2d=args.lambda_hand_2d, 
                lambda_hand_z=args.lambda_hand_z, 
                ntokens_pose= args.ntokens_pose,
                ntokens_action=args.ntokens_action,
                trans_factor=args.trans_factor,
                scale_factor=args.scale_factor,
                pose_loss=args.pose_loss)



                    
    if args.train_cont:
        epoch=reloadmodel.reload_model(model,args.resume_path)       
    else:
        epoch = 0
    epoch+=1
    
    #to multiple GPUs
    use_multiple_gpu= torch.cuda.device_count() > 1
    if use_multiple_gpu:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()

    freeze.freeze_batchnorm_stats(model)# Freeze batchnorm    


    print('**** Parameters to update ****')
    for i, (n,p) in enumerate(filter(lambda p: p[1].requires_grad, model.named_parameters())):
        print(i, n,p.size()) 

    
    #Optimizer
    model_params = filter(lambda p: p.requires_grad, model.parameters())   
    print(model_params) 

    
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    if args.lr_decay_gamma:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, gamma=args.lr_decay_gamma)


    if args.train_cont:
        reloadmodel.reload_optimizer(args.resume_path,optimizer,scheduler)
    
    for epoch_idx in tqdm(range(epoch, args.epochs+1), desc="epoch"):
        print(f"***Epoch #{epoch_idx}")
        epochpass.epoch_pass(
            loader,
            model,
            train=True,
            optimizer=optimizer,
            scheduler=scheduler,
            lr_decay_gamma=args.lr_decay_gamma,
            use_multiple_gpu=use_multiple_gpu,
            tensorboard_writer=board_writer,
            aggregate_sequence=False,
            is_single_hand=args.train_dataset!="h2ohands",
            dataset_action_info=dataset_info.action_to_idx,
            dataset_object_info=dataset_info.object_to_idx,
            ntokens = args.ntokens_action,
            is_demo=False,
            epoch=epoch_idx)

        if epoch_idx%args.snapshot==0:
            modelio.save_checkpoint(
                {
                    "epoch": epoch_idx, 
                    "network": "HTT",
                    "state_dict": model.module.state_dict() if use_multiple_gpu else model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler,
                },
                is_best=True,
                checkpoint=exp_id,
                snapshot=args.snapshot,)
    
    board_writer.close()

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    parser = argparse.ArgumentParser() 
    parser.add_argument('--experiment_tag',default='hello') 
    parser.add_argument('--dataset_folder',default='../fpha/')
    parser.add_argument('--cache_folder',default='./ws/ckpts/')
    parser.add_argument('--resume_path',default=None)

    #Transformer parameters
    parser.add_argument("--ntokens_pose", type=int, default=16, help="N tokens for P")
    parser.add_argument("--ntokens_action", type=int, default=128, help="N tokens for A")
    parser.add_argument("--spacing",type=int,default=2, help="Sample space for temporal sequence")
    
    # Dataset params
    parser.add_argument("--train_dataset",choices=["h2ohands", "fhbhands"],default="fhbhands",)
    parser.add_argument("--train_split", default="train", choices=["test", "train", "val"])
    
    
    parser.add_argument("--center_idx", default=0, type=int)
    parser.add_argument("--center_jittering", type=float, default=0.1, help="Controls magnitude of center jittering")
    parser.add_argument("--scale_jittering", type=float, default=0, help="Controls magnitude of scale jittering")

    # Training parameters
    parser.add_argument("--train_cont", action="store_true", help="Continue from previous training")
    parser.add_argument("--manual_seed", type=int, default=0)
    

    

    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--workers", type=int, default=16, help="Number of workers for multiprocessing")
    parser.add_argument("--pyapt_id")
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--lr_decay_gamma", type=float, default= 0.5,help="Learning rate decay factor, if 1, no decay is effectively applied")
    parser.add_argument("--lr_decay_step", type=float, default=15)
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--momentum", type=float, default=0.9)


    parser.add_argument("--trans_factor", type=float, default=100, help="Multiplier for translation prediction")
    parser.add_argument("--scale_factor", type=float, default=0.0001, help="Multiplier for scale prediction")
    
    #Transformer    
    parser.add_argument("--pose_loss", default="l1", choices=["l2", "l1"])
    parser.add_argument('--enc_pose_layers', default=2, type=int,
                        help="Number of encoding layers in P")
    parser.add_argument('--enc_action_layers', default=2, type=int,
                        help="Number of encoding layers in A")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")




    #Loss
    parser.add_argument("--lambda_action_loss",type=float, default=1, help="Weight for action/object classification")#lambda for action, lambda_3
    parser.add_argument("--lambda_hand_2d",type=float,default=1,help="Weight for hand 2D loss")#2*lambda_2, where factor 2 because of x and y
    parser.add_argument("--lambda_hand_z",type=float,default=100,help="Weight for hand z loss")#lambda_1*lambda_2


    parser.add_argument("--snapshot", type=int, default=5, help="How often to save intermediate models (epochs)" )


    args = parser.parse_args()
    for key, val in sorted(vars(args).items(), key=lambda x: x[0]):
        print(f"{key}: {val}")

    main(args)


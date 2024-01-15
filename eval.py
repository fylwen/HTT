import argparse
from datetime import datetime

from matplotlib import pyplot as plt
import torch

from libyana.exputils.argutils import save_args
from libyana.modelutils import freeze
from libyana.randomutils import setseeds
from torch.utils.tensorboard import SummaryWriter
from datasets import collate
from models.htt import TemporalNet
from netscripts import epochpass
from netscripts import reloadmodel, get_dataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

plt.switch_backend("agg")
print('********')
print('Lets start')


def collate_fn(seq, extend_queries=[]):
    return collate.seq_extend_flatten_collate(seq,extend_queries)#seq_extend_collate(seq, extend_queries)




def main(args):
    setseeds.set_all_seeds(args.manual_seed)
    # Initialize hosting
    now = datetime.now()
    
    experiment_tag = args.experiment_tag
    exp_id = f"{args.cache_folder}"+experiment_tag+"/"
    save_args(args, exp_id, "opt") 
    
    print("**** Lets eval on", args.val_dataset, args.val_split)
    val_dataset, _ = get_dataset.get_dataset_htt(
        args.val_dataset,
        dataset_folder=args.dataset_folder,
        split=args.val_split, 
        no_augm=True,
        scale_jittering=args.scale_jittering,
        center_jittering=args.center_jittering,
        ntokens_pose=args.ntokens_pose,
        ntokens_action=args.ntokens_action,
        spacing=args.spacing,
        is_shifting_window=True,
        split_type="actions"
    )


    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers),
        drop_last=False,
        collate_fn= collate_fn,
    )
    
    dataset_info=val_dataset.pose_dataset

    #Re-load pretrained weights 
    print('**** Load pretrained-weights from resume_path', args.resume_path)
    model= TemporalNet(dataset_info=dataset_info,
                is_single_hand=args.train_dataset!="h2ohands",
                transformer_num_encoder_layers_action=args.enc_action_layers,
                transformer_num_encoder_layers_pose=args.enc_pose_layers,
                transformer_d_model=args.hidden_dim,
                transformer_dropout=args.dropout,
                transformer_nhead=args.nheads,
                transformer_dim_feedforward=args.dim_feedforward,
                transformer_normalize_before=True,
                lambda_action_loss=1.,
                lambda_hand_2d=1., 
                lambda_hand_z=1., 
                ntokens_pose= args.ntokens_pose,
                ntokens_action=args.ntokens_action,
                trans_factor=args.trans_factor,
                scale_factor=args.scale_factor,
                pose_loss=args.pose_loss)

    epoch=reloadmodel.reload_model(model,args.resume_path)
    use_multiple_gpu= torch.cuda.device_count() > 1
    if use_multiple_gpu:
        assert False, "Not implement- Eval with multiple gpus!"
        #model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()

    freeze.freeze_batchnorm_stats(model)

    model_params = filter(lambda p: p.requires_grad, model.parameters())   
    
    optimizer=None
    
 
    val_save_dict, val_avg_meters, val_results = epochpass.epoch_pass(
        val_loader,
        model,
        train=False,
        optimizer=None,
        scheduler=None,
        lr_decay_gamma=0.,
        use_multiple_gpu=False,
        tensorboard_writer=None,
        aggregate_sequence=True,
        is_single_hand= args.train_dataset!="h2ohands",
        dataset_action_info=dataset_info.action_to_idx,
        dataset_object_info=dataset_info.object_to_idx,     
        ntokens=args.ntokens_action,
        is_demo=args.is_demo,
        epoch=epoch)
 
         
if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    parser = argparse.ArgumentParser()

    # Base params
    parser.add_argument('--experiment_tag',default='htt')    
    parser.add_argument('--is_demo', action="store_true", help="show demo result")  

    parser.add_argument('--dataset_folder',default='../fpha/')
    parser.add_argument('--cache_folder',default='./ws/ckpts/')
    parser.add_argument('--resume_path',default='./ws/ckpts/htt_fpha/checkpoint_45.pth')

    #Transformer parameters
    parser.add_argument("--ntokens_pose", type=int, default=16, help="N tokens for P")
    parser.add_argument("--ntokens_action", type=int, default=128, help="N tokens for A")
    parser.add_argument("--spacing",type=int,default=2, help="Sample space for temporal sequence")
    
    # Dataset params
    parser.add_argument("--train_dataset",choices=["h2ohands", "fhbhands"],default="fhbhands",)
    parser.add_argument("--val_dataset", choices=["h2ohands", "fhbhands"], default="fhbhands",) 
    parser.add_argument("--val_split", default="test", choices=["test", "train", "val"])
    
    
    
    parser.add_argument("--center_idx", default=0, type=int)
    parser.add_argument(
        "--center_jittering", type=float, default=0.1, help="Controls magnitude of center jittering"
    )
    parser.add_argument(
        "--scale_jittering", type=float, default=0, help="Controls magnitude of scale jittering"
    )



    # Training parameters
    parser.add_argument("--manual_seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers for multiprocessing")
    parser.add_argument("--epochs", type=int, default=500)
   

    parser.add_argument(
        "--trans_factor", type=float, default=100, help="Multiplier for translation prediction"
    )
    parser.add_argument(
        "--scale_factor", type=float, default=0.0001, help="Multiplier for scale prediction"
    )



    #Transformer
    parser.add_argument("--pose_loss", default="l1", choices=["l2", "l1"])
    parser.add_argument('--enc_pose_layers', default=2, type=int,
                        help="Number of encoding layers in P")
    parser.add_argument('--enc_action_layers', default=2, type=int,
                        help="Number of encoding layers in A")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")#256
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
  
  

    args = parser.parse_args()
    for key, val in sorted(vars(args).items(), key=lambda x: x[0]):
        print(f"{key}: {val}")

    main(args)

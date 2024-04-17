import numpy as np

from datasets import seqset_htt,fhbhands,h2ohands
from torch.utils.data import Dataset, DataLoader
from prefetch_generator import BackgroundGenerator
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def get_dataset_htt(
    dataset_name,
    dataset_folder,
    split,
    ntokens_pose,
    ntokens_action,
    spacing,
    is_shifting_window,
    no_augm,
    center_jittering,
    scale_jittering,
    split_type
):


    if dataset_name == "fhbhands":
        pose_dataset = fhbhands.FHBHands(split=split, 
                                        dataset_folder=dataset_folder, 
                                        ntokens_pose=ntokens_pose,
                                        ntokens_action=ntokens_action, 
                                        spacing=spacing, 
                                        is_shifting_window=is_shifting_window,
                                        split_type=split_type,)
        input_res = (480, 270)
    elif dataset_name=='h2ohands':
        pose_dataset = h2ohands.H2OHands(dataset_folder=dataset_folder,
                                    split=split,
                                    ntokens_pose=ntokens_pose,
                                    ntokens_action=ntokens_action,
                                    spacing=spacing,
                                    is_shifting_window=is_shifting_window,
                                    split_type=split_type)
        input_res = (480, 270)
    
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")




    dataset = seqset_htt.SeqSet(
        pose_dataset=pose_dataset,
        train=not no_augm,
        queries=pose_dataset.all_queries,
        center_jittering=center_jittering,
        scale_jittering=scale_jittering,
        inp_res=input_res,
        ntokens=pose_dataset.ntokens_action,
        spacing=pose_dataset.spacing,
    )


    return dataset, input_res


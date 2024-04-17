import os
import pickle
import traceback
import warnings

import torch

from libyana.modelutils import modelio


def load_opts(resume_checkpoint):
    # Identify if folder or checkpoint is provided
    is_abs_path=resume_checkpoint[0]=='/'
    if resume_checkpoint.endswith(".pth"):
        resume_checkpoint = os.path.join(*resume_checkpoint.split("/")[:-1])
    opt_path = os.path.join(resume_checkpoint, "opt.pkl")
    if is_abs_path:
        opt_path='/'+opt_path
    print(opt_path)
    with open(opt_path, "rb") as p_f:
        opts = pickle.load(p_f)
    return opts
 
def reload_model(model, resume_checkpoint,optimizer=None):
    if resume_checkpoint:
        start_epoch, _ = modelio.load_checkpoint(
            model, optimizer=optimizer, resume_path=resume_checkpoint, strict=False, as_parallel=False
        )
    else:
        start_epoch = 0
    return start_epoch


def reload_optimizer(resume_path, optimizer, scheduler=None):
    if os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path)
    try:
        missing_states = set(optimizer.state_dict().keys()) - set(checkpoint["optimizer"].keys())
        if len(missing_states) > 0:
            warnings.warn("Missing keys in optimizer ! : {}".format(missing_states))
        optimizer.load_state_dict(checkpoint["optimizer"])
    except ValueError:
        traceback.print_exc()
        warnings.warn("Couldn' load optimizer from {}".format(resume_path))

    if not scheduler is None:
        try:
            scheduler.load_state_dict(checkpoint["scheduler"].state_dict())
        except ValueError:
            traceback.print_exc()
            warnings.warn("Couldn' load scheduler from {}".format(resume_path))

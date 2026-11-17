"""Isolated ×8 data/model helpers; no changes to historical ×2/×4 implementations."""
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from sr02_degrade_benchmark import degrade_real2

def build(config,pretrained=True):
    directory=Path(config['weights_dir'])
    sys.path.insert(0,str(directory))
    from network_swinir import SwinIR
    model=SwinIR(upscale=8,in_chans=3,img_size=64,window_size=8,img_range=1.,depths=[6]*6,embed_dim=180,num_heads=[6]*6,mlp_ratio=2,upsampler='pixelshuffle',resi_connection='1conv')
    if pretrained:
        state=torch.load(directory/config['weights_file'],map_location='cpu',weights_only=True)
        model.load_state_dict(state.get('params',state),strict=True)
    return model

def prepare(path,seed,stream,step,index,augment):
    bgr=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if bgr is None:raise FileNotFoundError(path)
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    assert rgb.shape==(256,256,3)
    rng=np.random.default_rng([seed,stream,step,index])
    rotation=int(rng.integers(4)) if augment else 0
    flip=bool(rng.random()<.5) if augment else False
    target=np.rot90(rgb,rotation)
    if flip:target=target[:,::-1]
    target=np.ascontiguousarray(target)
    lr,params=degrade_real2(target,32,rng)
    assert lr.shape==(32,32,3)
    metadata=dict(seed_components=[seed,stream,step,index],rotation90=rotation,horizontal_flip=flip,degradation=params)
    return np.ascontiguousarray(lr),target,metadata

def tensor(images):
    return torch.from_numpy(np.stack(images).transpose(0,3,1,2).copy()).float()/255.

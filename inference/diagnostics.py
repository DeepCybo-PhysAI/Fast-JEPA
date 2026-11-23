from pathlib import Path
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

def load(path):
    path=Path(path)
    cfg=OmegaConf.load(path.parent/'config.json')
    if cfg.get('world_model_type') != 'fast' or cfg.get('_target_') != 'hybrid_jepa.JEPA':
        raise ValueError('Only Fast-JEPA checkpoints are supported')
    model=instantiate(cfg)
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
    return model.cuda().eval()

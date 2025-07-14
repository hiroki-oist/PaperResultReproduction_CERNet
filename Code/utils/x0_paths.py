import os
from pathlib import Path


if 'CERNET_SAVE_DIR' not in os.environ:
    raise RuntimeError("CERNET_SAVE_DIR environment variable not defined")

RESULTS_DIR = Path(os.environ['CERNET_SAVE_DIR']) / 'results'
FIGURES_DIR = Path(os.environ['CERNET_SAVE_DIR']) / 'figures'



def er_path(cfg, index=None, relative=False):
    if relative:
        save_path = cfg['er'].get('save_directory', cfg['training']['save_directory'])
    else:
        save_path = cfg['er'].get('save_directory_abs', cfg['training']['save_directory_abs'])
    suffix = 'er' if index is None else 'er_{}'.format(index)
    return Path(save_path) / suffix

def train_fig_path(cfg):
    return FIGURES_DIR / cfg['training']['save_directory']
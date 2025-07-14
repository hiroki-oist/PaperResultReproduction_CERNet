import glob
from pathlib import Path

import numpy as np
import toml

from . import x0_paths as paths


def load_cfg(cfg):
    if isinstance(cfg, TOMLConfigLoader):
        return cfg
    elif isinstance(cfg, (str, Path)):
        return TOMLConfigLoader(str(cfg))
    else:
        raise ValueError(cfg)


class TOMLConfigLoader:
    """Load a TOML config"""

    def __init__(self, cfg_path):
        self.cfg_path = cfg_path
        cfg = self._load_cfg(self.cfg_path)
        self._postprocess_cfg(cfg)

    def __getitem__(self, name):
        return self.cfg[name]

    def _toml_load(self, cfg_path):
        """Parse TOML file and convert `DynamicInlineTableDict` into dict in the results

        This is necessary because `DynamicInlineTableDict` cannot be pickled and
        therefore prevent using python multiprocessing.
        """
        def walk_dict(d):
            if isinstance(d, (list, tuple)):
                return [walk_dict(d_i) for d_i in d]
            elif isinstance(d, dict):
                return {k: walk_dict(v) for k, v in d.items()}
            else:
                return d
        return walk_dict(toml.load(cfg_path))

    def _update(self, base, d):
        """Updates a base config with another config's values

        Will recursively be called on dictionary values, and overwrite any non-dictioary value.
        """
        for section, values in d.items():
            if isinstance(values, dict):
                base.setdefault(section, {})
                self._update(base[section], values)
            else:
                base[section] = values
        return base


    def _load_cfg(self, cfg_path: str):
        cfg = self._toml_load(cfg_path)
        if 'base' in cfg:
            base_cfg = self._load_cfg(Path(cfg_path).parent / cfg['base'])
            base_cfg = self._update(base_cfg, cfg)
            return base_cfg
        else:
            return cfg

    def _postprocess_cfg(self, cfg):
        self.cfg = cfg
        # computing full paths
        # dataset_path = self.cfg['dataset']['dataset_path']

        self.cfg['dataset']['dataset_path_abs'] = self.cfg['dataset']['dataset_path']
        if not Path(self.cfg['dataset']['dataset_path_abs']).is_absolute():  # then look relative to the config directory
            self.cfg['dataset']['dataset_path_abs'] = str(Path(self.cfg_path).parent / self.cfg['dataset']['dataset_path_abs'])

        self.cfg['training']['save_directory_abs'] = self.cfg['training']['save_directory']
        if not Path(self.cfg['training']['save_directory_abs']).is_absolute():  # then relative to the ${PVRNN_SAVE_DIR}
            self.cfg['training']['save_directory_abs'] = str(paths.RESULTS_DIR / self.cfg['training']['save_directory_abs'])

        if self.cfg.get('er', {}).get('save_directory', None) is not None:
            self.cfg['er']['save_directory_abs'] = self.cfg['er']['save_directory']
            if not Path(self.cfg['er']['save_directory_abs']).is_absolute():  # then relative to the ${PVRNN_SAVE_DIR}
                self.cfg['er']['save_directory_abs'] = str(paths.RESULTS_DIR / self.cfg['er']['save_directory_abs'])

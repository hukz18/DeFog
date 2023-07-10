import hydra
import utils
import torch
import logging
from core import train
import warnings
warnings.filterwarnings('ignore', message='.*gym.*')
warnings.filterwarnings('ignore', message='.*observation.*')
warnings.filterwarnings('ignore', message='.*environment.*')
logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision('high')

@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def main(cfg):
    log_dict = utils.get_log_dict()
    for seed in cfg.seeds:
        train(cfg, seed, log_dict, -1, logger, None)
    utils.visualize_perf_drop_curve(cfg, log_dict)

if __name__ == "__main__":
    main()

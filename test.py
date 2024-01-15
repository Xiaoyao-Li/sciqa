import os
import json
import hydra
import torch
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf
from loguru import logger
import numpy as np

# from utils.misc import compute_model_dim
from utils.io import mkdir_if_not_exists
from utils.plot import Ploter
from datasets.base import create_dataset
from datasets.misc import collate_fn_general
from models.base import create_model
from models.visualizer import create_visualizer

def test(cfg: DictConfig) -> None:
    """ testing portal

    Args:
        cfg: configuration dict
    """
    if cfg.gpu is not None:
        device = f'cuda:{cfg.gpu}'
    else:
        device = 'cpu'

    ## prepare dataset for train and test
    datasets = {
        'test': create_dataset(cfg.task.dataset, 'test', cfg.slurm, cfg.charlie),
    }
    if cfg.task.visualizer.visualize:
        raise NotImplementedError('Visualizer is not implemented yet.')
    for subset, dataset in datasets.items():
        logger.info(f'Load {subset} dataset size: {len(dataset)}')
    
    collate_fn = collate_fn_general
    
    dataloaders = {
        'test': datasets['test'].get_dataloader(
            batch_size=cfg.task.test.batch_size,
            collate_fn=collate_fn,
            num_workers=cfg.task.test.num_workers,
            pin_memory=True,
            shuffle=False,
        ),
    }
    if 'test_for_vis' in datasets:
        raise NotImplementedError('Visualizer is not implemented yet.')
    
    ## create model, diffuser, and optimizer
    model = create_model(cfg.model, slurm=cfg.slurm, charlie=cfg.charlie)
    model.to(device=device)
    
    ## load ckpt
    load_ckpt(model, path=os.path.join(cfg.ckpt_dir, 'model-best.pth'))

    ## start testing
    model.eval()
    results = {}  ## format: {question_index: answer_index/None}
    for it, data in enumerate(dataloaders['test']):
        logger.info(f'Testing iter {it}..')
        for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)

        outputs = model(data)
        pred_answer = outputs['pred_answer']
        pids = data['pid']
        for pid, answer in zip(pids, pred_answer):
            results[pid] = answer
    
    ## complete not answered question
    full_test_pids = datasets['test'].pid_splits['test']
    for pid in full_test_pids:
        if pid not in results:
            results[pid] = None

    ## save results
    with open(os.path.join(cfg.result_dir, 'results.json'), 'w') as f:
        json.dump(results, f)

def load_ckpt(model: torch.nn.Module, path: str) -> None:
    """ load ckpt for current model

    Args:
        model: current model
        path: save path
    """
    assert os.path.exists(path), 'Can\'t find provided ckpt.'

    saved_state_dict = torch.load(path)['model']
    model_state_dict = model.state_dict()

    for key in model_state_dict:
        if key in saved_state_dict:
            model_state_dict[key] = saved_state_dict[key]
            logger.info(f'Load parameter {key} for current model.')
        ## model is trained with ddp
        if 'module.'+key in saved_state_dict:
            model_state_dict[key] = saved_state_dict['module.'+key]
            logger.info(f'Load parameter {key} for current model [Traind from multi GPUs].')
    
    model.load_state_dict(model_state_dict)

@hydra.main(version_base=None, config_path="./configs", config_name="test")
def main(cfg: DictConfig) -> None:

    if os.environ.get('SLURM') is not None:
        cfg.slurm = True
    if os.environ.get('CHARLIE') is not None:
        cfg.charlie = True
    ## avoide charlie and slurm at the same time
    assert not (cfg.slurm and cfg.charlie)

    ## set output logger and tensorboard
    if cfg.slurm:
        logger.remove(handler_id=0) # remove default handler
    elif cfg.charlie:
        logger.remove(handler_id=0)
    else:
        logger.remove(handler_id=0)
    logger.add(cfg.exp_dir + '/test.log')

    mkdir_if_not_exists(cfg.result_dir)

    ## Begin training progress
    logger.info('Configuration: \n' + OmegaConf.to_yaml(cfg))
    logger.info('Begin testing..')

    test(cfg) # testing portal

    ## Training is over!
    logger.info('End testing..')

if __name__ == '__main__':
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main()

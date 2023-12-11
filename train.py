import os
import hydra
import torch
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf
from loguru import logger

# from utils.misc import compute_model_dim
from utils.io import mkdir_if_not_exists
from utils.plot import Ploter
from datasets.base import create_dataset
from datasets.misc import collate_fn_general
from models.base import create_model
from models.visualizer import create_visualizer

def train(cfg: DictConfig) -> None:
    """ training portal

    Args:
        cfg: configuration dict
    """
    if cfg.gpu is not None:
        device = f'cuda:{cfg.gpu}'
    else:
        device = 'cpu'

    ## prepare dataset for train and test
    datasets = {
        'train': create_dataset(cfg.task.dataset, 'train', cfg.slurm),
        'val': create_dataset(cfg.task.dataset, 'val', cfg.slurm),
        # 'test': create_dataset(cfg.task.dataset, 'test', cfg.slurm),
    }
    if cfg.task.visualizer.visualize:
        datasets['test_for_vis'] = create_dataset(cfg.task.dataset, 'test', cfg.slurm, case_only=True)
    for subset, dataset in datasets.items():
        logger.info(f'Load {subset} dataset size: {len(dataset)}')
    
    collate_fn = collate_fn_general
    
    dataloaders = {
        'train': datasets['train'].get_dataloader(
            batch_size=cfg.task.train.batch_size,
            collate_fn=collate_fn,
            num_workers=cfg.task.train.num_workers,
            pin_memory=True,
            shuffle=True,
        ),
        'val': datasets['val'].get_dataloader(
            batch_size=cfg.task.test.batch_size,
            collate_fn=collate_fn,
            num_workers=cfg.task.test.num_workers,
            pin_memory=True,
            shuffle=True,
        ),
    }
    if 'test_for_vis' in datasets:
        raise NotImplementedError('Visualizer is not implemented yet.')
    
    ## create model, diffuser, and optimizer
    model = create_model(cfg.model, slurm=cfg.slurm)
    # diffuser = create_diffuser(model, cfg.diffuser)
    model.to(device=device)
    
    params = []
    nparams = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            params.append(p)
            nparams.append(p.nelement())
            logger.info(f'add {n} {p.shape} for optimization')
    
    params_group = [
        {'params': params, 'lr': cfg.task.lr},
    ]
    optimizer = torch.optim.Adam(params_group) # use adam optimizer defaultly
    logger.info(f'{len(params)} parameters for optimization.')
    logger.info(f'total model size is {sum(nparams)}.')

    ## create evaluator and visualizer
    # evaluator = create_evaluator(cfg.task.evaluator)
    if cfg.task.visualizer.visualize:
        visualizer = create_visualizer(cfg.task.visualizer)
    
    ## start train from scratch
    step = 0
    for epoch in range(0, cfg.task.train.num_epochs):
        ## train in epoch
        model.train()
        for it, data in enumerate(dataloaders['train']):
            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)

            optimizer.zero_grad()
            outputs = model(data)
            outputs['loss'].backward()
            optimizer.step()
            
            ## plot loss
            if (step + 1) % cfg.task.train.log_step == 0:
                total_loss = outputs['loss'].item()
                log_str = f'[TRAIN] ==> Epoch: {epoch+1:3d} | Iter: {it+1:5d} | Step: {step+1:7d} | Loss: {total_loss:.3f}'
                logger.info(log_str)
                for key in outputs:
                    val = outputs[key].item() if torch.is_tensor(outputs[key]) else outputs[key]
                    Ploter.write({
                        f'train/{key}': {'plot': True, 'value': val, 'step': step},
                        'train/epoch': {'plot': True, 'value': epoch, 'step': step},
                    })

            step += 1
        
        ## test in epoch
        if (epoch + 1) % cfg.task.eval_interval == 0:
            best_epoch = epoch
            save_ckpt(
                model=diffuser, optimizer=optimizer, epoch=best_epoch, step=step,
                path=os.path.join(cfg.ckpt_dir, f'model-{(epoch // 300 + 1) * 300}.pth')
            )

        ## test for visualize
        if cfg.task.visualizer.visualize and (epoch + 1) % cfg.task.eval_visualize == 0:
            vis_dir = os.path.join(cfg.vis_dir, f'epoch{epoch+1:3d}')
            visualizer.visualize(diffuser, dataloaders['test_for_vis'], vis_dir)

def save_ckpt(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, step: int, path: str) -> None:
    """ Save current model and corresponding data

    Args:
        model: best model
        optimizer: corresponding optimizer object
        epoch: best epoch
        step: current step
        path: save path
    """
    saved_state_dict = {}
    model_state_dict = model.state_dict()
    for key in model_state_dict:
        if 'scene_model' not in key:
            saved_state_dict[key] = model_state_dict[key]
    
    torch.save({
        'model': saved_state_dict, 
        'optimizer': optimizer.state_dict(), 
        'epoch': epoch, 'step': step,
    }, path)

@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg: DictConfig) -> None:

    if os.environ.get('SLURM') is not None:
        cfg.slurm = True

    ## set output logger and tensorboard
    if cfg.slurm:
        logger.remove(handler_id=0) # remove default handler
    logger.add(cfg.exp_dir + '/runtime.log')

    mkdir_if_not_exists(cfg.tb_dir)
    mkdir_if_not_exists(cfg.vis_dir)
    mkdir_if_not_exists(cfg.ckpt_dir)

    writer = SummaryWriter(log_dir=cfg.tb_dir)
    Ploter.setWriter(writer)

    ## Begin training progress
    logger.info('Configuration: \n' + OmegaConf.to_yaml(cfg))
    logger.info('Begin training..')

    train(cfg) # training portal

    ## Training is over!
    writer.close() # close summarywriter and flush all data to disk
    logger.info('End training..')

if __name__ == '__main__':
    main()

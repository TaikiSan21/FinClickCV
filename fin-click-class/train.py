# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 17:16:52 2022

@author: tnsak
"""

from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor, Normalize
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn import CrossEntropyLoss
import os
import dataset
import model
import yaml
import torch
from tqdm import trange

def create_dataloader(cfg, split='train'):
    label_csv = os.path.join(cfg['label_dir'],     ['label_csv'][split])
    base_trans = Compose([ToPILImage(), 
                    Resize([224, 224]), 
                    ToTensor(),
                    Normalize(mean=cfg['norm_mean'],
                                        std=cfg['norm_sd'])])
    trans_dict = {
        'train': base_trans,
        'val': base_trans,
        'test': base_trans
        }
    
    dataset = FCDataset(cfg, label_csv, trans_dict[split])
    if split == 'train':
        if cfg['weighted_sampler']:
            sp = dataset.label
            weights = np.array([len(sp)/sp.count(x) for x in range(cfg['num_classes'])])
            samp_weight= weights[sp]
            sampler = WeightedRandomSampler(samp_weight, len(samp_weight))
            shuffle = False
        else:
            shuffle = True
            sampler = None
    else:
        sampler = None
        shuffle = False
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        shuffle=shuffle,
        sampler=sampler
        )
    return dataloader

def load_model(cfg):
    '''
    creates a new model or loads existing if one found
    '''
    model = FCNet(cfg)
    model_dir = cfg['model_save_dir']
    model_states = glob.glob(model_dir + '/*.pt')

    if cfg['resume'] and len(model_states) > 0:
        # found a save state
        model_epochs = [int(m.replace(model_dir + '/','').replace('.pt','')) for m in model_states]
        start_epoch = max(model_epochs)
        # load state dict and apply weights to model
        print(f'Resuming from epoch {start_epoch}')
        state = torch.load(open(model_dir+f'/{start_epoch}.pt', 'rb'), map_location='cpu')
        model.load_state_dict(state['model'])

    else:
        # no save state found; start anew
        print('Starting new model')
        start_epoch = 0

    return model, start_epoch

def save_model(cfg, epoch, model, stats):
    # make sure save directory exists; create if not
    model_dir = cfg['model_save_dir']
    os.makedirs(model_dir, exist_ok=True)

    # get model parameters and add to stats...
    stats['model'] = model.state_dict()

    # ...and save
    torch.save(stats, open(model_dir + f'/{epoch}.pt', 'wb'))
    
    # also save config file if not present
    cfpath = model_dir + '/config.yaml'
    if not os.path.exists(cfpath):
        with open(cfpath, 'w') as f:
            yaml.dump(cfg, f)
            
def setup_optimizer(cfg, model):
    optimizer = SGD(model.parameters(),
                    lr=cfg['learning_rate'],
                    weight_decay=cfg['weight_decay'])
    return optimizer

def train(cfg, dataloader, model, optimizer):
    device = cfg['device']
    model.to(device)
    model.train()
    
    criterion = CrossEntropyLoss()
    oa_total, loss_total = 0.0, 0.0
    pb = trange(len(dataloader))
    all_pred = np.empty(0)
    all_true = np.empty(0)
    for ix, image, label in enumerate(dataloader):
        all_true = np.append(all_true, label.detach().numpy())
        image, label = image.to(device), label.to(device)
        prediction = model(image)
        optimizer.zero_grad()
        loss = criterion(prediction, label)
        
        loss.backward()
        optimizer.step()
    
        loss_total += loss.item()
        pred_label = torch.argmax(prediction, dim=1)
        all_pred = np.append(all_pred, pred_label.cpu().detach().numpy())
        oa = torch.mean((pred_label == label).float())
        oa_total += oa.item()
        
        pb.set_description(
            '[Train] Loss {.2f} OA {.2f}%'.format(
                loss_total/(ix+1),
                100*oa_total/(ix+1)
                )
            )
        pb.update(1)
    
    pb.close()
    loss_total /= len(dataloader)
    oa_total /= len(dataloader)
    
    return loss_total, oa_total, all_true, all_pred
    
def validate(cfg, dataloader, model):
    device = cfg['device']
    model.to(device)
    model.eval()
    
    criterion = CrossEntropyLoss()
    
    oa_total, loss_total = 0.0, 0.0
    pb = trange(len(dataloader))
    all_pred = np.empty(0)
    all_true = np.empty(0)
    with torch.no_grad():
        for ix, image, label in enumerate(dataloader):
            all_true = np.append(all_true, label.detach().numpy())
            image, label = image.to(device), label.to(device)
            prediction = model(image)

            loss = criterion(prediction, label)

            loss_total += loss.item()
            pred_label = torch.argmax(prediction, dim=1)
            all_pred = np.append(all_pred, pred_label.cpu().detach().numpy())
            oa = torch.mean((pred_label == label).float())
            oa_total += oa.item()
            
            pb.set_description(
                '[Val] Loss {.2f} OA {.2f}%'.format(
                    loss_total/(ix+1),
                    100*oa_total/(ix+1)
                    )
                )
            pb.update(1)
    
    pb.close()
    loss_total /= len(dataloader)
    oa_total /= len(dataloader)
    
    return loss_total, oa_total, all_true, all_pred

def main():
    parser = argparse.ArgumentParser(description='Train finclick model')
    parser.add_argument('--config', help='Path to config file', default='configs/config.yaml')
    
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, 'r'))
    init_seed(cfg['seed'])
    
    dl_train = create_dataloader(cfg, split='train')
    dl_val = create_dataloader(cfg, split='val')
    
    model, epoch= load_model(cfg)
    optim = setup_optimizer(cfg, model)
    
    num_epochs = cfg['num_epochs']
    scheduler = MultiStepLR(optim, milestones=cfg['lr_milestones'], gamma=cfg['lr_gamma'])
    # TODO: add wandb or tboard
    while epoch < num_epochs:
        epoch += 1
        train_loss, train_oa, train_true, train_pred = train(cfg, dl_train, model, optim)
        val_loss, val_oa, val_true, val_pred = val(cfg, dl_val, model)
        scheduler.step()
        stats = {
           'loss_train': train_loss,
           'loss_val': val_loss,
           'oa_train': train_oa,
           'oa_val': val_oa
            # 'cba_train': np.mean(cba_train),
            # 'cba_val': np.mean(cba_val)
        }
        save_model(cfg, epoch, model, stats)
    
    #done!
    
if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python ct_classifier/train.py").
    main()

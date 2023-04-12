import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ImitationDataset 
import os
import shutil
from omegaconf import OmegaConf
from model import ActiveDecisionModel

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

def write_log(file, content):
    with open(file, "a") as f:
        f.write(content) 

class ActiveVisionModel(object):
    def __init__(self, cfg_file) -> None:
        self.cfg_file = cfg_file
        self.cfg = OmegaConf.load(cfg_file)
        self.save_root = self.cfg.save_root
        os.makedirs(self.save_root, exist_ok=True)
        self.log = os.path.join(self.save_root, "train_log.log")
        self.device = self.cfg.device
        self.build_model()
        self.criterion = nn.MSELoss()

    def build_model(self,):
        model_cfg = self.cfg.model
        model = ActiveDecisionModel(in_channels=4)
        ckpt_file = model_cfg.get("ckpt", None)
        use_xavier_init = model_cfg.get("use_xavier_init", False)

        if use_xavier_init:
            model.apply(init_weights)
        if ckpt_file:
            ckpt = torch.load(ckpt_file, map_location=self.device)
            model.load_state_dict(ckpt)
            write_log(self.log, f"load {ckpt_file} success!")

        self.model = model.to(self.device)

    def build_dataloader(self, data_cfg):
        transform_funs = []
        transforms_cfg = OmegaConf.to_container(data_cfg.transforms)

        for f, params in transforms_cfg.items():
            if params is None:
                fun = getattr(transforms, f)()
            elif isinstance(params, list):
                fun = getattr(transforms, f)(*params)
            elif isinstance(params, dict):
                fun = getattr(transforms, f)(**params)
            else:
                raise TypeError("params must be in [null, list, dict]...")
            transform_funs.append(fun)
        transform_funs = transforms.Compose(transform_funs)
        dataset = ImitationDataset(data_cfg.root, transform_funs, data_cfg.imgsz)
        loader = DataLoader(dataset, batch_size=data_cfg.batch_size, shuffle=data_cfg.shuffle, num_workers=data_cfg.num_workers)
        return loader


    
    def build_optimizer(self, train_cfg):
        optimizer_cfg = train_cfg.optimizer
        optimizer = getattr(optim, optimizer_cfg.name)(self.model.parameters(), **optimizer_cfg.params)

        scheduler_cfg = train_cfg.scheduler
        scheduler = getattr(optim.lr_scheduler, scheduler_cfg.name)(optimizer, **scheduler_cfg.params)

        return optimizer, scheduler

    def train(self,):

        shutil.copy2(self.cfg_file, self.save_root)

        train_cfg = self.cfg.train
        epochs = train_cfg.epoch
        train_loader = self.build_dataloader(train_cfg.train_data)
        val_loader = self.build_dataloader(train_cfg.val_data)
        
        optimizer, scheduler = self.build_optimizer(train_cfg)

        max_loss = 1e5
        for epoch in range(1, 1+epochs):
            self.model.train()
            train_loss = 0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            content = f"epoch={epoch}, lr={optimizer.state_dict()['param_groups'][0]['lr']:.3f}, loss={train_loss:.3f} \n"
            print(content)
            write_log(self.log, content)

            val_loss = self.valid(val_loader)
            if val_loss <= max_loss:
                val_loss = max_loss
                save_name = os.path.join(self.save_root, "best.pth")
                torch.save(self.model.state_dict() , save_name)
                write_log(self.log, "save best model... \n")
            
            scheduler.step()
            write_log(self.log, "\n")

        save_name = os.path.join(self.save_root, "last.pth")
        torch.save(self.model.state_dict() , save_name)
        
    def valid(self, val_loader):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()

        content = f"Val_loss={test_loss:.3f} \n"
        write_log(self.log, content)
        
        return test_loss

    @staticmethod
    def inference(model, img, device, transform_cfg):
        pass





if __name__ == '__main__':

    
    active_model = ActiveVisionModel("config.yaml")
    active_model.train()





import os
import argparse
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from model.vqgan_fmri import VQGAN_fMRI
from model.nsd_data import NSDDataModule
from utils.config_utils import load_config, merge_args_and_config

def main():
    parser = argparse.ArgumentParser(description="Train VQGAN on NSD fMRI data")
    parser.add_argument("--config", type=str, default="configs/config.yml", help="Path to config file")
    
    # Parse args
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set seed
    pl.seed_everything(config['seed'])
    
    # Data
    print("Initializing Data Module...")
    data_module = NSDDataModule(config)
    
    # Model
    print("Initializing VQGAN Model...")
    
    class ModelArgs:
        def __init__(self, config_dict):
            # ... (same as before) ...
            self.embedding_dim = config_dict['vqgan']['embedding_dim']
            self.n_codes = config_dict['vqgan']['n_embeddings']
            self.n_hiddens = config_dict['vqgan']['n_hiddens']
            self.lr = float(config_dict['train']['learning_rate'])
            self.downsample = config_dict['vqgan']['downsample_factor']
            self.disc_channels = 64 
            self.disc_layers = 2    
            self.discriminator_iter_start = config_dict['vqgan']['disc_start']
            self.disc_loss_type = "hinge"
            self.image_gan_weight = 1.0 
            self.video_gan_weight = 1.0
            self.l1_weight = 4.0
            self.gan_feat_weight = 0.0
            self.perceptual_weight = 0.0 
            self.i3d_feat = False
            self.restart_thres = 1.0
            self.no_random_restart = False
            self.norm_type = "group"
            self.padding_type = "replicate"
            self.image_channels = 1 
            self.sequence_length = config_dict['data']['num_frames']
            self.resolution = config_dict['data']['fmri_resolution'][0] 
            self.batch_size = config_dict['train']['batch_size'] 
            self.gpus = 1
            self.accumulate_grad_batches = config_dict['train']['accumulate_grad_batches']

    model_args = ModelArgs(config)
    model = VQGAN_fMRI(model_args)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join("checkpoints", config['experiment_name']),
        filename='vqgan-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val/recon_loss',
        mode='min'
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['train']['epochs'],
        callbacks=[checkpoint_callback],
        accelerator=config['device'] if config['device'] != 'cpu' else 'cpu',
        devices=1, 
        # accumulate_grad_batches=config['train']['accumulate_grad_batches'], # Manual optimization handles this
        log_every_n_steps=config['train']['log_every'],
        check_val_every_n_epoch=config['train']['validation_every'],
        precision="16-mixed"
        # gradient_clip_val=... # Removed
    )
    
    # Train
    print("Starting Training...")
    trainer.fit(model, data_module)

if __name__ == "__main__":
    main()

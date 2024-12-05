import os
import time
import yaml
import random
import argparse
import logging
import importlib
from datetime import datetime
import numpy as np
from statistics import mean

import torch

from fvcore.nn import FlopCountAnalysis
from diffusion import create_diffusion


#################################################################################
#                             Helper Functions                                  #
#################################################################################
def create_log_file(log_dir='logs'):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = f"{log_dir}/evaluation_{timestamp}.log"
    return log_filename

def load_model(module_name, class_name):
    """
    Dynamically load the model class from the specified module.
    """
    module = importlib.import_module(f"{module_name}")
    model_class = getattr(module, class_name)
    return model_class

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config    

def dataset_stats(file_name):
    scale_factor = 100
    dataset_stats = {
        'track': (),
    } 
    for dataset_name, (mn, std) in dataset_stats.items():
        if dataset_name in file_name:
            return mn, std, scale_factor
    else:
        raise ValueError(f"Unrecognized dataset for file name: {file_name}")

    


#################################################################################
#                       Sampling and Evaluation Loop                            #
#################################################################################
def main(config):
    # Setup PyTorch:
    torch.manual_seed(config['sample']['seed'])
    random.seed(config['sample']['seed'])
    torch.set_grad_enabled(False)
    device = f"cuda:{config['sample']['cuda_device']}" if torch.cuda.is_available() else "cpu"
    
    # Initialize model args
    seq_length=config['model']['seq_length']
    hist_length=config['model']['hist_length']
    n_mels=config['model']['n_mels']
    
    # Initialize the model:
    # Note that parameter initialization is done within the model constructor
    model_class = load_model(config['model']['module'], config['model']['class'])
    model_name = config['model']['name']
    model = model_class[model_name](
        seq_length=seq_length,
        hist_length=hist_length,
        n_mels=n_mels,
        use_ckpt_wrapper=False,
    ).to(device)
    
    # Load a TrafficDiffuser checkpoint:
    state_dict = torch.load(config['model']['ckpt'], map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict["model"])
    model.eval()  # important!
    print('===> Model initialized !')
    
    # Create diffusion with the desired number of sampling steps 
    diffusion = create_diffusion(timestep_respacing=str(config['sample']['num_sampling_steps']))
    
    # Set up logging to file
    log_filename = create_log_file(log_dir=os.path.dirname(os.path.dirname(config['model']['ckpt'])))
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')
    
    # Print model parameters, and summary
    logging.info(f"{model_name} Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    logging.info(f"{model_name} Model summary:\n{model}\n")
    
    # Print model flops
    batch_size = 1 # to ensure flops and inference time are calculated for a single scenario
    dummy_x = torch.randn(batch_size, n_mels, seq_length, device=device)
    dummy_t = torch.randn(batch_size, device=device)
    dummy_h = torch.randn(batch_size, n_mels, hist_length, device=device)
    flops = FlopCountAnalysis(model, (dummy_x, dummy_t, dummy_h))
    gflops = flops.total() / 1e9
    logging.info(f"{model_name} GFLOPs: {gflops:.4f}\n")
    
    # Print model sampling time
    model_kwargs = dict(h=dummy_h)
    num_trials = 10
    avg_sampling_time = 0
    for _ in range(num_trials):
        torch.cuda.synchronize()
        tic = time.time()
        samples = diffusion.p_sample_loop(
                model.forward, dummy_x.shape, dummy_x, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )
        torch.cuda.synchronize()
        toc = time.time()
        avg_sampling_time += (toc - tic)
    avg_sampling_time /= num_trials
    logging.info(f"{model_name} Sampling time: {avg_sampling_time:.2f} s\n")
    print('===> Sampling time calculated !')
        
    # Choose a subset of scenarios from testset:            
    test_files = sorted(random.sample(sorted(os.listdir(config['data']['test_dir'])), config['data']['subset_size']))
    
    # Make samples directory
    samples_dir = os.path.dirname(config['model']['ckpt']).replace("checkpoints", "samples")
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    
    # Retrieve dataset stats
    stats = np.load(config['data']['stats_file'])
    mean_, std_, scale_ = stats["mean"], stats["std"], stats["scale"]  
    
    # Sample future chunks from test_files
    for filename in test_files:
        data = np.load(os.path.join(config['data']['test_dir'], filename))
        data = torch.tensor(data, dtype=torch.float32).to(device)        
        data = data.unsqueeze(0).expand(config['sample']['num_sampling'], data.size(0), data.size(1))
        model_kwargs = dict(h=data[:, :, :hist_length])
        
        # Create sampling noise:
        x = torch.randn(config['sample']['num_sampling'], n_mels, seq_length, device=device)
        
        # Sample trajectories:
        samples = diffusion.p_sample_loop(
            model.forward, x.shape, x, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )
        data_gen = torch.cat((data[:, :, :hist_length], samples), 2)
        data_gen = data_gen.cpu().numpy()
        
        # Unormalize and unscale audio data_gen
        data_gen = (data_gen / scale_) * std_ + mean_
                
        # Save sampled trajectories
        np.save(os.path.join(samples_dir, filename), data_gen)
        print(f'===> Audio chunk {filename} sampled and saved !')
        
    print(f'===> End of sampling.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_sample.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)

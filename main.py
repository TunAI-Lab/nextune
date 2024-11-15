import os

if __name__ == '__main__':
    
    # Train on multiple gpus
    #os.system("accelerate launch -m scripts.train --config configs/config_train.yaml")
    
    # Train on 1 gpu
    os.system("accelerate launch --num-processes=1 --gpu_ids 1 -m scripts.train --config configs/config_train.yaml")

    # Sample
    #os.system("accelerate launch -m scripts.sample --config configs/config_sample.yaml")
    
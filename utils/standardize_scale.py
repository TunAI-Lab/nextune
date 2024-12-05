import os
import json
import numpy as np


# --- Function: Calculate dataset statistics ---
def calculate_dataset_statistics(dataset_path):
    """
    Calculate Mean/Std values for x and y positions throughout the dataset.

    Parameters:
        dataset_path (str): Path to the processed dataset.

    Returns:
        tuple: Mean and standard deviation for the dataset.
    """
    data_list = []
    for filename in os.listdir(dataset_path):
        data = np.load(os.path.join(dataset_path, filename))
        data_list.append(data)           
    dataset = np.concatenate(data_list, axis=0)
    mean, std = np.mean(dataset, axis=0), np.std(dataset, axis=0)
    return mean, std

# --- Function: Standardize and scale scenario ---
def standardize_scale(data, mean, std, scale_factor):
    """
    Standardize and scale data using dataset-specific statistics.

    Parameters:
        data (np.ndarray): data to be standardized and scaled.
        mean (np.ndarray): Mean values for the dataset.
        std (np.ndarray): Standard deviation values for the dataset.
        scale_factor (float): Scale factor to apply after standardization.

    Returns:
        np.ndarray: Standardized and scaled data.
    """
    data = ((data - mean) / std) * scale_factor
    return data
   
   
# --- Main Pipeline ---
def main():
    """
    Main pipeline for preprocessing scenarios, calculating dataset statistics,
    and standardizing/scaling data.
    """
    dataset_path = '/data/ahmed.ghorbel/workdir/nextune/data/prep'
    output_path = '/data/ahmed.ghorbel/workdir/nextune/data/norm'
    stats_path = "/data/ahmed.ghorbel/workdir/nextune/utils/stats.npz"
    os.makedirs(output_path, exist_ok=True)
    
    scale = 100

    # Calculate and save dataset stats
    mean, std = calculate_dataset_statistics(dataset_path)
    stats = {"mean": mean, "std": std, "scale": scale}
    np.savez(stats_path, **stats)
    
    #data = np.load(stats_path)
    #mean = data["mean"]
    #std = data["std"]
    #scale = data["scale"]
    #print(f'mean:{mean}, std:{std}, scale:{scale}')

    # standardize and scale dataset    
    for filename in os.listdir(dataset_path):
        data = np.load(os.path.join(dataset_path, filename))
        data = standardize_scale(data, mean, std, scale_factor)
        np.save(os.path.join(output_path, filename), data)
        print(f"Standardized and saved {filename}")

if __name__ == "__main__":
    main()
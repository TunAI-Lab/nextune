import os
import librosa
import numpy as np
import argparse
import yaml



#################################################################################
#                             Helper Functions                                  #
#################################################################################
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

  
def preprocess_audio(file_path, target_dir, sample_rate,
                     chunk_seconds, n_mels, hop_length, n_fft):
    # Load audio file
    # Define sr=None to determine the original audio sr
    audio, sr = librosa.load(file_path, sr=sample_rate)    
    #print(f'audio original sample_rate: {sr}')    
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Total duration of each chunk (history + future)
    total_seconds = chunk_seconds
    total_samples = total_seconds * sample_rate
    
    # Break into chunks
    for i in range(0, len(audio) - total_samples + 1, total_samples):
        chunk = audio[i:i + total_samples]

        # Convert to Mel-spectrogram
        chunk_mel = librosa.feature.melspectrogram(
            y=chunk, sr=sample_rate, n_fft=n_fft,
            hop_length=hop_length, n_mels=n_mels
        )

        # Convert to decibel scale (log-scale)
        chunk_mel_db = librosa.power_to_db(chunk_mel, ref=np.max)

        # Save as npy files for quick loading during training
        save_path = os.path.join(target_dir, f"{file_name}_chunk_{i}.npy")
        np.save(save_path, chunk_mel_db)
        print(f"Processed and saved audio chunk of shape:{chunk_mel_db.shape} in {save_path}")


#################################################################################
#                             Preprocessing Loop                                #
#################################################################################
def main(config):
    # Ensure the target directory exists
    source_dir = config['data']['raw_dir']
    target_dir = config['data']['prep_dir']
    os.makedirs(target_dir, exist_ok=True)
    
    # Process each audio file in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith(".wav") or filename.endswith(".mp3"):
            file_path = os.path.join(source_dir, filename)
            preprocess_audio(
                file_path, target_dir,
                sample_rate=config['process']['sample_rate'],
                chunk_seconds=config['process']['chunk_seconds'],
                n_mels=config['process']['n_mels'],
                hop_length=config['process']['hop_length'],
                n_fft=config['process']['n_fft'],
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_process.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)
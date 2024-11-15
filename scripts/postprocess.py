import os
import librosa
import numpy as np
import argparse
import yaml
import soundfile as sf



#################################################################################
#                             Helper Functions                                  #
#################################################################################
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_as_mp3(audio_data, sample_rate, output_file):
    """Save audio data as an .mp3 file."""
    temp_wav = output_file.replace(".mp3", ".wav")
    sf.write(temp_wav, audio_data, sample_rate, format='wav')
    
    # Convert WAV to MP3 using ffmpeg
    os.system(f"ffmpeg -i {temp_wav} -q:a 0 -y {output_file}")
    os.remove(temp_wav)  # Clean up the temporary .wav file

def postprocess_audio(filename, source_dir, target_dir, sample_rate):
    """Save all audio chunks in .mp3 format."""
    input_path = os.path.join(source_dir, filename)
    audio_data = np.load(input_path)
    
    output_path = os.path.join(target_dir, filename.replace(".npy", ".mp3")) 
    save_as_mp3(audio_data, sample_rate, output_path)
    print(f"Processed and saved: {output_path}")


#################################################################################
#                             Preprocessing Loop                                #
#################################################################################
def main(config):
    # Ensure the target directory exists
    source_dir = config['data']['prep_dir']
    target_dir = config['data']['mp3_dir']
    os.makedirs(target_dir, exist_ok=True)
    
    # Process each audio file in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith(".npy"):
            postprocess_audio(
                filename, source_dir, target_dir,
                sample_rate=config['process']['sample_rate'],
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_process.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)
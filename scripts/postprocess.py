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
    os.system(f"ffmpeg -y -i {temp_wav} -q:a 0 {output_file}")
    os.remove(temp_wav)  # Clean up the temporary .wav file

def postprocess_audio(
    filename, source_dir, target_dir,
    sample_rate, hop_length, n_fft):
    """Convert an .npy file containing a Mel-spectrogram back to an audio .mp3 file."""
    input_path = os.path.join(source_dir, filename)
    mel_spectrogram = np.load(input_path)

    # Convert from log-Mel-spectrogram back to Mel-spectrogram
    mel_spectrogram = librosa.db_to_power(mel_spectrogram, ref=1.0)

    # Invert the Mel-spectrogram to audio
    audio_data = librosa.feature.inverse.mel_to_audio(
        mel_spectrogram, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
    )
    
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
                hop_length=config['process']['hop_length'],
                n_fft=config['process']['n_fft']
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_process.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)
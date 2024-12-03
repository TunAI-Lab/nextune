import os
import librosa
import numpy as np
import argparse
import yaml
from pydub import AudioSegment
import soundfile as sf


#################################################################################
#                             Helper Functions                                  #
#################################################################################
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_audio_chunk(chunk, sample_rate, save_path):
    # Convert chunk to the right format for saving
    save_path_wav = save_path.replace(".mp3", ".wav")
    sf.write(save_path_wav, chunk, sample_rate)

    # Convert WAV to MP3 using pydub
    audio_segment = AudioSegment.from_file(save_path_wav, format="wav")
    audio_segment.export(save_path, format="mp3")
    os.remove(save_path_wav)  # Clean up temporary wav file
    print(f"Audio chunk saved in {save_path}")


def preprocess_audio(file_path, target_dir, sample_rate,
                     chunk_seconds, n_mels, hop_length, n_fft):
    # Load audio file
    # Define sr=None to determine the original audio sr
    audio, sr = librosa.load(file_path, sr=sample_rate)
    print(f'Original audio has shape:{audio.shape} and sample_rate:{sr}')    
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Total duration of each chunk (history + future)
    total_samples = chunk_seconds * sample_rate
    
    # Break into chunks
    for i in range(0, len(audio) - total_samples + 1, total_samples):
        chunk = audio[i:i + total_samples]
    
        ## Convert to stft
        #stft = librosa.stft(y=chunk, n_fft=n_fft, hop_length=hop_length)
        #magnitude, phase = np.abs(stft), np.angle(stft)
        #if i == 0:
        #    print(f'magnitude shape:{magnitude.shape}, and phase shape:{phase.shape}.')
        ## Reconstruct the complex spectrogram
        #complex_spectrogram = magnitude * np.exp(1j * phase)
        #save_path = os.path.join(target_dir, f"{file_name}_chunk_{i}.mp3")
        #reconstructed_audio = librosa.istft(complex_spectrogram, hop_length=hop_length)
        #save_audio_chunk(reconstructed_audio, sample_rate, save_path)
        
        # Convert to Mel-spectrogram
        chunk_mel = librosa.feature.melspectrogram(
            y=chunk, sr=sample_rate, n_fft=n_fft,
            hop_length=hop_length, n_mels=n_mels
        )
        save_path = os.path.join(target_dir, f"{file_name}_chunk_{i}.npy")
        np.save(save_path, chunk_mel)
        print(f"Processed and saved audio chunk of shape:{chunk_mel.shape} in {save_path}")
        ## Reconstruct mp3 audio chunk from mel chunk
        #reconstructed_audio = librosa.feature.inverse.mel_to_audio(
        #    chunk_mel, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
        #)
        ## Save as npy files for quick loading during training
        #save_path = os.path.join(target_dir, f"{file_name}_chunk_{i}.mp3")
        #save_audio_chunk(reconstructed_audio, sample_rate, save_path)
        
        ## Save chunks as mp3
        #save_path = os.path.join(target_dir, f"{file_name}_chunk_{i}.mp3")
        #save_audio_chunk(chunk, sample_rate, save_path)
        #print(f"Processed and saved audio chunk of shape:{chunk.shape} in {save_path}")


#################################################################################
#                             Preprocessing Loop                                #
#################################################################################
def main(config):
    # Ensure the target directory exists
    source_dir = config['data']['raw_dir']
    target_dir = config['data']['prep_dir']
    os.makedirs(target_dir, exist_ok=True)
    
    # Process each audio file in the source directory
    for filename in sorted(os.listdir(source_dir)):
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
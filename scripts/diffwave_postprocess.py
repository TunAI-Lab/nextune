import os
import numpy as np
import torch
import torchaudio as T
import torchaudio.transforms as TT

# Data params of diffwave
params = {
    'sample_rate': 22050,
    'n_mels': 80,
    'hop_samples': 256,
    'n_fft': 1024,
}

def inverse_transform(npy_filename, output_dir):
    # Load the spectrogram
    spectrogram = np.load(npy_filename)
    spectrogram = torch.tensor(spectrogram, dtype=torch.float32)

    # Reverse normalization and log conversion
    spectrogram = spectrogram * 100 - 100 + 20
    spectrogram = torch.pow(10.0, spectrogram / 20.0)  # Convert from dB scale back to linear

    # Inverse mel transform
    inverse_mel_transform = TT.InverseMelScale(
        n_stft=params['n_fft'] // 2 + 1,
        n_mels=params['n_mels'],
        sample_rate=params['sample_rate']
    )

    # Griffin-Lim for phase reconstruction
    griffin_lim = TT.GriffinLim(
        n_fft=params['n_fft'], 
        hop_length=params['hop_samples'], 
        power=1.0, 
        n_iter=32  # Increase iterations for better phase estimation
    )

    with torch.no_grad():
        linear_spec = inverse_mel_transform(spectrogram)
        waveform = griffin_lim(linear_spec)

    # Save the reconstructed waveform
    output_path = os.path.join(output_dir, os.path.basename(npy_filename).replace(".npy", ".wav"))
    os.makedirs(output_dir, exist_ok=True)
    T.save(output_path, waveform.unsqueeze(0), params['sample_rate'])

    print(f"Saved reconstructed audio to {output_path}")
    
# Example usage:
inverse_transform("/data/ahmed.ghorbel/workdir/nextune/data/test_npys/LJ001-0001.npy", "/data/ahmed.ghorbel/workdir/nextune/data/test_npys")

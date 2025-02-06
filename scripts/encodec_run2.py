import os
import numpy as np

import torch
import torchaudio

from encodec import EncodecModel
from encodec.utils import convert_audio



# Instantiate a pretrained EnCodec model
model = EncodecModel.encodec_model_24khz()
# The number of codebooks used will be determined bythe bandwidth selected.
# E.g. for a bandwidth of 6kbps, `n_q = 8` codebooks are used.
# Supported bandwidths are 1.5kbps (n_q = 2), 3 kbps (n_q = 4), 6 kbps (n_q = 8) and 12 kbps (n_q =16) and 24kbps (n_q=32).
# For the 48 kHz model, only 3, 6, 12, and 24 kbps are supported. The number
# of codebooks for each is half that of the 24 kHz model as the frame rate is twice as much.
model.set_target_bandwidth(24.0)
datadir = "/data/ahmed.ghorbel/workdir/nextune/data/wavs/"
outdir = "/data/ahmed.ghorbel/workdir/nextune/data/codes2/"
os.makedirs(outdir, exist_ok=True)

for filename in os.listdir(datadir)[:2]:
    filepath = os.path.join(datadir, filename)
    print(filepath)

    # Load and pre-process the audio waveform
    wav, sr = torchaudio.load(filepath)
    print(sr)
    wav = convert_audio(wav[:, :200000], sr, sr, model.channels)
    wav = wav.unsqueeze(0)

    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = model.encode(wav)
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
    print(codes.shape)
    codes = codes[0]
    print('codes', codes.shape)
    print('codes type', codes.dtype)
    outpath = os.path.join(outdir, filename.split(".")[0])
    np.save(outpath, codes.cpu().numpy())
    


#--- decode and save back to wav
encoded_fr = [(codes.unsqueeze(0), None)]
audio_values = model.decode(encoded_fr)
print(audio_values.shape)
output_path = '/data/ahmed.ghorbel/workdir/nextune/scripts/output2.wav'
torchaudio.save(output_path, audio_values[0], sr)
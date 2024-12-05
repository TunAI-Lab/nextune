# TODO List


### Features
- [x] Add data preprocessing script for transforming audio samples from temporal to frequency domain.
- [x] Adapt backbone architetcure to the new data shape. 
- [x] Update train and sample scripts accordingly following the previous feature.
- [ ] verify if train and sampling works well
- [ ] update postprocess script to leverage nd array of data_gen


### Bugs


### Enhancements


### Conclusions


### Data Processing Notes
Why using a Mel-spectrogram or STFT spectrogram as audio data representation is generally the most effective format for the following reasons:
- Balanced Resolution: Spectrograms provide a good balance between time and frequency resolution, enabling the model to learn both temporal progression and spectral content.
- Dimensionality Reduction: A Mel-spectrogram reduces the frequency resolution based on perceptual importance, making it easier for a model to process while retaining essential audio features.
- Ease of Reconstruction: Converting predictions from the frequency domain back to the temporal domain is feasible with inverse STFT functions.

Example Workflow for Spectrogram-Based Diffusion:
- Convert your 15-second history and future 5-second target segments into Mel-spectrograms or STFT spectrograms.
- Normalize each spectrogram to a suitable range (e.g., [-1, 1] or [0, 1]) for model input. You can also standardize and then scale.
- Train the Diffusion Model to predict future spectrogram frames based on the given history.
- Invert the generated spectrograms back to the waveform domain using an inverse STFT or a neural vocoder.

Rule of Thumb: A hop length of 512 samples (for a 1024-sample window) is a good starting point, especially for music.

Sampling Rate Consideration:
- For audio at a standard sampling rate of 44.1 kHz or 48 kHz, a window size of 1024 samples and hop length of 512 samples works well for most music genres.
- If your audio is at a lower sample rate (e.g., 16 kHz), you may consider smaller values like window size = 512 and hop length = 256.
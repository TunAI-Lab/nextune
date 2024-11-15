# TODO List


### Features
- [x] Add data preprocessing script for transforming audio samples from temporal to frequency domain.
- [ ] Adapt backbone architetcure to the new data shape. 
- [ ] Update train and sample scripts accordingly following the previous feature.


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

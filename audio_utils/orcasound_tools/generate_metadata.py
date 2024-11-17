import functools
import os
import torchaudio
import torch
import json
import argparse
import tqdm
from sklearn.model_selection import train_test_split
import glob
import torchaudio.transforms as T
from concurrent.futures import ProcessPoolExecutor, as_completed
import random


def check_file_size(path):
    if os.path.getsize(path) > 0:
        return path
    return None


def is_loadable(path, audio_length):
    try:
        waveform, sample_rate = torchaudio.load(path, backend="ffmpeg")
        duration = waveform.size(1) / sample_rate
        if int(duration) >= audio_length:
            return path
    except Exception as e:
        print(f"Warning: Unloadable path {path}, or short than 10s")
    return None


def remove_empty_file_paths(paths, num_workers):
    non_empty_paths = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(check_file_size, path) for path in paths]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Removing Empty File Paths"):
            result = future.result()
            if result is not None:
                non_empty_paths.append(result)

    return non_empty_paths


def remove_unloadable_paths(paths, audio_length, num_workers):
    loadable_paths = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(is_loadable, path, audio_length) for path in paths]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Removing Unloadable Paths"):
            result = future.result()
            if result is not None:
                loadable_paths.append(result)

    return loadable_paths


def process_audio_files(data_dir, output_file, resampling_freq, num_workers, num_samples, gpu):
    train_metadata = []
    test_metadata = []
    sum_features = 0
    sum_squared_features = 0
    total_features_samples = 0
    total_samples = 0

    print("Gathering ts files...")
    data = glob.glob(os.path.join(data_dir, '**/*.ts'), recursive=True)
    # wav_files = glob.glob(os.path.join(data_dir, '**/*.wav'), recursive=True)
    # mp3_files = glob.glob(os.path.join(data_dir, '**/*.mp3'), recursive=True)
    # data = wav_files + mp3_files
    print(f"Found {len(data)} files.")
    if num_samples == None:
        num_samples = len(data)

    if num_samples <= len(data):
        data = random.sample(data, num_samples)
        print(f'Sampled {num_samples} random files')
    else:
        print('Num_samples is greater than the length of data, using all available data')

    print("Cleaning file paths...")
    data = remove_empty_file_paths(data, num_workers=num_workers)
    data = remove_unloadable_paths(data, args.audio_length, num_workers=num_workers)
    print(f"Number of loadable data files: {len(data)}")

    file_metadata = [{'file_path': file_path} for file_path in data]

    train_files, test_files = train_test_split(
        file_metadata,
        test_size=0.2,
        random_state=26
    )

    for file_dict in tqdm.tqdm(train_files, desc="Processing training files"):
        waveform, sample_rate = torchaudio.load(file_dict['file_path'], backend="ffmpeg")
        resampler = T.Resample(sample_rate, resampling_freq)
        if gpu:
            waveform = waveform.to('cuda')
            resampler.to('cuda')

        duration = waveform.size(1) / sample_rate
        for i in range(0, int(duration), args.audio_length):
            start_sample = i * sample_rate
            end_sample = min((i + args.audio_length) * sample_rate, waveform.size(1))

            if end_sample - start_sample < args.audio_length * sample_rate:
                continue

            segment = waveform[:, start_sample:end_sample]
            if resampling_freq is None:
                resampling_freq = sample_rate

            segment = resampler(segment)
            segment = segment - segment.mean()
            fbank = torchaudio.compliance.kaldi.fbank(segment, htk_compat=True, sample_frequency=sample_rate,
                                                      use_energy=False, window_type='hanning', num_mel_bins=128,
                                                      dither=0.0, frame_shift=10)

            sum_features += fbank.sum()
            sum_squared_features += (fbank ** 2).sum()
            total_features_samples += fbank.numel()

            train_metadata.append({
                'file_path': file_dict['file_path'],
                'start_time': i,
                'sr': sample_rate
            })
            total_samples += 1

    for file_dict in tqdm.tqdm(test_files, desc="Processing testing files"):
        waveform, sample_rate = torchaudio.load(file_dict['file_path'])
        duration = waveform.size(1) / sample_rate

        for i in range(0, int(duration), args.audio_length):
            start_sample = i * sample_rate
            end_sample = min((i + args.audio_length) * sample_rate, waveform.size(1))

            if end_sample - start_sample < args.audio_length * sample_rate:
                continue

            segment = waveform[:, start_sample:end_sample]
            if resampling_freq is None:
                resampling_freq = sample_rate

            resampler = T.Resample(sample_rate, resampling_freq)
            segment = resampler(segment)
            segment = segment - segment.mean()

            fbank = torchaudio.compliance.kaldi.fbank(segment, htk_compat=True, sample_frequency=sample_rate,
                                                      use_energy=False, window_type='hanning', num_mel_bins=128,
                                                      dither=0.0, frame_shift=10)

            sum_features += fbank.sum()
            sum_squared_features += (fbank ** 2).sum()
            total_features_samples += fbank.numel()

            test_metadata.append({
                'file_path': file_dict['file_path'],
                'start_time': i,
                'sr': sample_rate
            })
            total_samples += 1

    mean = sum_features / total_features_samples
    std = torch.sqrt(sum_squared_features / total_features_samples - mean ** 2)

    output_data = {
        'dataset_stats': {
            'mean': mean.item(),
            'std': std.item(),
            'total_samples': total_samples
        },
        'metadata': {
            'train': train_metadata,
            'test': test_metadata
        }
    }

    print(output_data['dataset_stats'])

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"Metadata file generated with {total_samples} samples across training and test sets.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metadata file for deepship dataset.")
    parser.add_argument('--data', type=str, required=True, help="Path to the directory containing deepship data.")
    parser.add_argument('--output_file', type=str, default='orcasound_metadata.json',
                        help="Path to the output JSON file.")
    parser.add_argument('--resampling_freq', default=None, type=int, help='Resampling frequency')
    parser.add_argument('--num_workers', default=None, type=int, help='Number of workers for data cleaning')
    parser.add_argument('--num_samples', default=None, type=int, help='Number of samples to process')
    parser.add_argument('--audio_length', default=10, type=int, help='Audio file length in seconds')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    parser.set_defaults(gpu=False)

    args = parser.parse_args()
    if args.gpu:
        print('Using GPU ....')

    process_audio_files(args.data, args.output_file, args.resampling_freq, args.num_workers, args.num_samples, args.gpu)
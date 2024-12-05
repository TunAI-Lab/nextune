import torch
import time
from fvcore.nn import FlopCountAnalysis
from models.backbones.model_nextune import NexTune_models
from fvcore.nn import FlopCountAnalysis
from diffusion import create_diffusion


# Test setup
model_name = 'NexTune-S'
batch_size = 1
hist_length = 469
seq_length = 469
n_mels = 256
use_ckpt_wrapper = False

num_sampling_steps = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create random torch tensors
dummy_x = torch.randn(batch_size, n_mels, seq_length, device=device)
dummy_t = torch.randn(batch_size, device=device)
dummy_h = torch.randn(batch_size, n_mels, hist_length, device=device)
            
# Initialize the models
model = NexTune_models[model_name](
    seq_length=seq_length,
    hist_length=hist_length,
    n_mels=n_mels,
    use_ckpt_wrapper=use_ckpt_wrapper,
).to(device)
#model.eval() #

# Print model parameters, and summary
print(f"{model_name} Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"{model_name} Model summary:\n{model}")

# Print model flops
flops = FlopCountAnalysis(model, (dummy_x, dummy_t, dummy_h))
gflops = flops.total() / 1e9
print(f"{model_name} GFLOPs: {gflops:.4f}")

# Print model sampling time
model_kwargs = dict(h=dummy_h)
num_trials = 10
avg_sampling_time = 0

diffusion = create_diffusion(timestep_respacing=str(num_sampling_steps))

for _ in range(num_trials):
    torch.cuda.synchronize()
    tic = time.time()
    samples = diffusion.p_sample_loop(
            model.forward, dummy_x.shape, dummy_x, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    torch.cuda.synchronize()
    toc = time.time()
    avg_sampling_time += (toc - tic)
avg_sampling_time /= num_trials
print(f"{model_name} Sampling time: {avg_sampling_time:.2f} s")
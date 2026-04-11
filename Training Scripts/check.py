import torch; 
print(f'CUDA Ready: {torch.cuda.is_available()}') 
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Compute Cap: {torch.cuda.get_device_capability(0)}')
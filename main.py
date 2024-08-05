import torch

# Torch 버전
print(torch.__version__)

# CUDA 사용 가능여부(True, False)
print(torch.cuda.is_available())

# CUDA 버전
print(torch.version.cuda)

# GPU 사용가능 개수
print(torch.cuda.device_count())

# GPU 이름
print(torch.cuda.get_device_name(0))


import torch

# PyTorch 버전 출력
print("PyTorch Version:", torch.__version__)

# CUDA 사용 가능 여부 출력
print("Is CUDA available?:", torch.cuda.is_available())

# CUDA 버전 출력
print("CUDA Version:", torch.version.cuda)

# GPU 사용 가능 개수 출력
print("Number of GPUs available:", torch.cuda.device_count())

# 첫 번째 GPU 이름 출력 (GPU가 여러 대 있는 경우 인덱스를 변경하여 다른 GPU 이름을 확인할 수 있습니다)
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")

# 랜덤 텐서 생성 및 출력
x = torch.rand(5, 3)
print("Random Tensor:", x)

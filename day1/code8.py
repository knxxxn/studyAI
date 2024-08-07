'''
가중치 초기화(weight initialization)
뉴럴 네트워크의 가중치들을 적절히 초기화하지 않으면, 학습에 어려움이 있을 수 있음
Xavier 초기화는 각 뉴런의 출력이 적절한 범위의 값을 가지도록 함
이를 통하여, 그래디언트 소실 등의 문제를 최소화함
PyTorch에서는 기본적으로 He initializer라는 초기화를 사용하며, 이 또한 효과적임
'''
import torch.nn.init as init
from torch import nn

from day1.code7 import input_size, hidden_size, num_classes


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

        # Xavier initialization
        self.xavier_init_weights()
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    def xavier_init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

model = MLP(input_size, hidden_size, num_classes)

'''
과대적합(overfitting)
과대적합이란 모델이 훈련 데이터에 너무 잘 맞춰져, 일반화 능력이 떨어지는 현상임
훈련 데이터에 대한 성능은 매우 좋아지나, 평가 데이터에 대한 성능은 낮아짐
훈련 데이터의 노이즈나 특이점까지 학습하게 됨
대체로 훈련 데이터가 충분하지 않을 경우에 과대적합이 일어나게 됨

드롭아웃(dropout)
학습 중에 무작위로 일부 뉴런을 비활성화함
모델이 특정한 특징에 과도하게 의존하는 것을 방지함
'''
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate) # Add dropout layer
        self.fc2 = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

model = MLP(input_size, hidden_size, num_classes, dropout_rate=0.5)

'''
내부 공변량 이동(internal covariate shift)
딥 뉴럴 네트워크의 학습 중, 각 층의 입력 데이터의 분포가 계속 변할 수 있음
이 경우, 각 층은 이전 층의 변화에 계속 적응해야 하며, 결국 학습이 느려지게 됨
뉴럴 네트워크의 가중치들이 변화된 입력 분포에 민감해지며 안정성을 떨어뜨림

배치 정규화(batch normalization)
배치 정규화는 각 배치의 입력을 정규화하여 내부 공변량 이동을 줄임
이를 통해, 학습 속도를 향상시킬 수 있음
또한, 가중치 초기화의 민감도를 감소시키며, 과대적합을 방지할 수 있음
'''
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size) # Add Batch Normalization
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = MLP(input_size, hidden_size, num_classes)
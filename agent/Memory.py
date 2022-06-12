import numpy as np
from collections import deque

# 経験メモリの定義
class Memory():
    # 初期化
    def __init__(self, memory_size):
        self.buffer = deque(maxlen=memory_size)

    # 経験の追加
    def add(self, experience):
        self.buffer.append(experience)

    # バッチサイズ分の経験をランダムに取得
    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[i] for i in idx]
    
    # 経験メモリのサイズ
    def __len__(self):
        return len(self.buffer)
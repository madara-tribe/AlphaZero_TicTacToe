from huber_loss import huber_loss_mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer

# 行動価値関数の定義
class QNetwork:
    # 初期化
    def __init__(self, state_size, action_size):
        # モデルの作成
        self.model = Sequential()
        self.model.add(Dense(16, activation='relu', input_dim=state_size))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))
        
        # モデルのコンパイル
        self.model.compile(loss=huber_loss_mean, optimizer=Adam(lr=0.001))
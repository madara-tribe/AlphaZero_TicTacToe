# ====================
# パラメータ更新部
# ====================

from Agent.dual_network import DN_INPUT_SHAPE
from tensorflow.keras.callbacks import LearningRateScheduler, LambdaCallback
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from pathlib import Path
import numpy as np
import pickle

## パラメータの準備
RN_EPOCHS = 100 # 学習回数
best_weight = './model/best.h5'
# 学習データの読み込み
def load_selftrain_data():
    history_path = sorted(Path('./data').glob('*.history'))[-1]
    with history_path.open(mode='rb') as f:
        return pickle.load(f)

def load_policy_value():
    history = load_selftrain_data()
    xs, y_policies, y_values = zip(*history)

    # 学習のための入力データのシェイプの変換
    a, b, c = DN_INPUT_SHAPE
    xs = np.array(xs)
    xs = xs.reshape(len(xs), c, a, b).transpose(0, 2, 3, 1)
    y_policies = np.array(y_policies)
    y_values = np.array(y_values)
    return xs, y_policies, y_values

def call_back():
    # 学習率
    def step_decay(epoch):
        x = 0.001
        if epoch >= 50: x = 0.0005
        if epoch >= 80: x = 0.00025
        return x
    lr_decay = LearningRateScheduler(step_decay)

    # 出力
    print_callback = LambdaCallback(
        on_epoch_begin=lambda epoch,logs:
                print('\rTrain {}/{}'.format(epoch + 1,RN_EPOCHS), end=''))
    return lr_decay, print_callback


# デュアルネットワークの学習
def create_latest_player():
    def train_dual_network():
        # 学習データの読み込み
        xs, y_policies, y_values = load_policy_value()

        # ベストプレイヤーのモデルの読み込み
        model = load_model(best_weight)
        # モデルのコンパイル
        model.compile(loss=['categorical_crossentropy', 'mse'], optimizer='adam')
        
        lr_decay, print_callback = call_back()
        

        # 学習の実行
        model.fit(xs, [y_policies, y_values], batch_size=128, epochs=RN_EPOCHS,
                verbose=0, callbacks=[lr_decay, print_callback])
        print('')

        # 最新プレイヤーのモデルの保存
        model.save('./model/latest.h5')

        # モデルの破棄
        K.clear_session()
        del model
    return train_dual_network

# 動作確認
if __name__ == '__main__':
    train_dual_network = create_latest_player()
    train_dual_network()

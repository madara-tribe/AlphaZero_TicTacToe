import numpy as np
import copy
import time
import serial
import cv2
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import *
from keras.utils.np_utils import to_categorical
from agent.Memory import Memory
from model.QFunction import QFunction


ser = serial.Serial('/dev/cu.usbmodem14301')
cap = cv2.VideoCapture(0)


# USB Camera
def capture(ndim=3):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    xp = int(frame.shape[1]/2)
    yp = int(frame.shape[0]/2)
    d = 400
    resize = 128
    cv2.rectangle(gray, (xp-d, yp-d), (xp+d, yp+d), color=0, thickness=10)
    cv2.imshow('gray', gray)
    gray = cv2.resize(gray[yp-d:yp + d, xp-d:xp + d],(resize, resize))
    env = np.asarray(gray, dtype=np.float32)
    if ndim == 3:
        return env[np.newaxis, :, :] 
    else:
        return env[np.newaxis, np.newaxis, :, :] 

def action_step(actions):
    r = 0
    if actions==0:
        ser.write(b"p")
    else:
        ser.write(b"i")
    time.sleep(1.0)
    r = ser.read() 
    return int(r)

# パラメータの準備
NUM_EPISODES = 50 # エピソード数
GAMMA = 1.0 

# 探索パラメータ
E_START = 1.0 # εの初期値
E_STOP = 0.01 # εの最終値
E_DECAY_RATE = 0.001 # εの減衰率
SUCCESS_REWARD = 8

# メモリパラメータ
MEMORY_SIZE = 10000 # 経験メモリのサイズ
MAX_STEPS = 30 # 最大ステップ数
BATCH_SIZE = MAX_STEPS
H = 128
W = 128
C = 1
inputs = Input([H, W, C])


class Environment:
    def __init__(self, inputs):
        # main-networkの作成
        self.main_qn = QFunction(inputs)

        # target-networkの作成
        self.target_qn = QFunction(inputs)
        # 経験メモリの作成
        self.memory = Memory(MEMORY_SIZE)

    def action_value_function(self, action_b, step_reward_b):
        target = 0
        if action_b == 1:
            target = GAMMA + step_reward_b/MAX_STEPS
        else:
            target = GAMMA + step_reward_b/MAX_STEPS
        return target

    def default(self):
        self.memory = Memory(MEMORY_SIZE)

    def run(self):
        # エピソード数分のエピソードを繰り返す
        total_step = 0 # 総ステップ数
        success_count = 0 # 成功数
        for episode in range(1, NUM_EPISODES+1):
            step = 0 # ステップ数
            R = 0
            batch_size = 0
            ser.write(b"c")

            # target-networkの更新
            self.target_qn.set_weights(self.main_qn.get_weights())
            
            # 1エピソードのループ
            for _ in range(1, MAX_STEPS+1):
                step += 1
                total_step += 1
                camera_state = capture(ndim=3)
                camera_state = camera_state.reshape(1, H, W, C)
                # εを減らす
                epsilon = E_STOP + (E_START - E_STOP)*np.exp(-E_DECAY_RATE*total_step)
                
                # ランダムな行動を選択
                if epsilon > np.random.rand():
                    action = int(np.random.randint(0, 2, 1))
                # 行動価値関数で行動を選択
                else:
                    action = np.argmax(self.main_qn.predict(camera_state))
                    pred = self.main_qn.predict(camera_state)
                # 行動に応じて状態と報酬を得る
                reward = action_step(action)
                R += reward
                self.memory.add((camera_state, action, reward, R)) 
                print('step', step, 'action', action, "R", R, 'reward', reward)
            if R >=SUCCESS_REWARD:
                success_count += 1
                
            # ニューラルネットワークの入力と出力の準備
            inputs = np.zeros((BATCH_SIZE, H, W, C)) # 入力(状態)
            targets = np.zeros((BATCH_SIZE, 2)) # 出力(行動ごとの価値)
            # バッチサイズ分の経験をランダムに取得
            minibatch = self.memory.sample(BATCH_SIZE)
            
            # ニューラルネットワークの入力と出力の生成
            for i, (state_b, action_b, reward_b, step_reward_b) in enumerate(minibatch):
                
                # 入力に状態を指定
                inputs[i] = state_b
                
                # 採った行動の価値を計算
                target = self.action_value_function(action_b, step_reward_b)
                # 出力に行動ごとの価値を指定
                targets[i] = self.main_qn.predict(state_b)
                targets[i][action_b] = target # 採った行動の価値

            # 行動価値関数の更新
            print('training....')
            self.main_qn.fit(inputs, targets, epochs=30, verbose=0)
            
            # エピソード完了時のログ表示
            print('エピソード: {}, ステップ数: {}, epsilon: {:.4f}'.format(episode, step, epsilon))
            self.default()
            # 5回連続成功で学習終了
            if success_count >= 5:
                break



cartpole_env = Environment(inputs)
cartpole_env.run()

ser.close()
cap.release()

#agent.save('agent')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
from gradient_param import get_policy, update_theta
from episode_play import play
from env import set_env
# %matplotlib inline

fig, circle = set_env()

# config
# パラメータθの初期値の準備
theta_0 = np.array([
    [np.nan, 1, 1, np.nan], # 0 上,右,下,左
    [np.nan, 1, 1, 1], # 1
    [np.nan, np.nan, np.nan, 1], # 2
    [1, np.nan, 1, np.nan], # 3
    [1, 1, np.nan, np.nan], # 4
    [np.nan, np.nan, 1, 1], # 5
    [1, 1, np.nan, np.nan], # 6
    [np.nan, np.nan, np.nan, 1]]) # 7


# パラメータθの初期値を方策に変換
pi_0 = get_policy(theta_0)
print(pi_0)

# 1エピソードの実行と履歴の確認
s_a_history = play(pi_0)
print(s_a_history)
print('1エピソードのステップ数：{}'.format(len(s_a_history)+1))

# train
stop_epsilon = 10**-4 # しきい値
theta = theta_0 # パラメータθ
pi = pi_0 # 方策

# エピソードを繰り返し実行して学習
for episode in range(10000):
    # 1エピソード実行して履歴取得
    s_a_history = play(pi)
    
    # パラメータθの更新
    theta = update_theta(theta, pi, s_a_history)
    
    # 方策の更新
    pi_new = get_policy(theta)
    
    # 方策の変化量
    pi_delta = np.sum(np.abs(pi_new-pi))
    pi = pi_new    
    
    # 出力
    print('エピソード: {}, ステップ： {}, 方策変化量: {:.4f}'.format(
        episode, len(s_a_history)-1, pi_delta))
    
    # 終了判定
    if pi_delta < stop_epsilon: # 方策の変化量がしきい値以下
        break

# アニメーションの定期処理を行う関数
def animate(i):
    state = s_a_history[i][0]
    circle.set_data((state % 3) + 0.5, 2.5 - int(state / 3))
    return circle

# アニメーションの表示
anim = animation.FuncAnimation(fig, animate, \
        frames=len(s_a_history), interval=200, repeat=False)
HTML(anim.to_jshtml())


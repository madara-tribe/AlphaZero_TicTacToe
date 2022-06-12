import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
from env import set_env
from value_function import get_policy, get_s_next, get_a_value_func, set_default_value
from sarsa_Qlearning import play
# %matplotlib inline

fig, circle = set_env()

# パラメータθの初期値の準備
theta_0 = np.array([
    [np.nan, 1, 1, np.nan], # 0
    [np.nan, 1, 1, 1], # 1
    [np.nan, np.nan, np.nan, 1], # 2
    [1, np.nan, 1, np.nan], # 3
    [1, 1, np.nan, np.nan], # 4
    [np.nan, np.nan, 1, 1], # 5
    [1, 1, np.nan, np.nan], # 6
    [np.nan, np.nan, np.nan, 1]]) # 7

# パラメータθの初期値を方策に変換
pi_0 = get_policy(theta_0)
# 価値の初期値を準備
Q = set_default_value(theta_0)
print(pi_0)
print(Q)


# train
epsilon = 0.5  # ε-greedy法のεの初期値
use_sarsa=True
# エピソードを繰り返し実行して学習
for episode in range(10):
    # ε-greedyの値を少しずつ小さくする
    epsilon = epsilon / 2

    # 1エピソード実行して履歴と行動価値関数を取得
    [s_a_history, Q] = play(Q, epsilon, pi_0, use_sarsa=use_sarsa)
    
    # 出力
    print('エピソード: {}, ステップ: {}'.format(
        episode, len(s_a_history)-1))

# アニメーションの定期処理を行う関数
def animate(i):
    state = s_a_history[i][0]
    circle.set_data((state % 3) + 0.5, 2.5 - int(state / 3))
    return circle

# アニメーションの表示
anim = animation.FuncAnimation(fig, animate, \
        frames=len(s_a_history), interval=200, repeat=False)
HTML(anim.to_jshtml())


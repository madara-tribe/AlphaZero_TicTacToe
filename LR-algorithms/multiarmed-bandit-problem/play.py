import numpy as np
import random
import math
import pandas as pd
import matplotlib.pyplot as plt
from EpsilonGreedy_Agent import EpsilonGreedyAgent
from env import SlotArm
from UCB1_Agent import UCB1_Agent
# %matplotlib inline

# シミュレーションの実行
def play(algo, arms, num_sims, num_time):
    # 履歴の準備
    times = np.zeros(num_sims * num_time) # ゲーム回数の何回目か
    rewards = np.zeros(num_sims * num_time) # 報酬

    # シミュレーション回数分ループ
    for sim in range(num_sims):
        algo.initialize(len(arms)) # アルゴリズム設定の初期化

        # ゲーム回数分ループ
        for time in range(num_time):
            # インデックスの計算
            index = sim * num_time + time

            # 履歴の計算
            times[index] = time+1
            chosen_arm = algo.policy()
            reward = arms[chosen_arm].draw()
            rewards[index] = reward

            # アルゴリズムのパラメータの更新
            algo.update(chosen_arm, reward, time+1)

    # [ゲーム回数の何回目か, 報酬]
    return [times, rewards]

# アームの準備
arms = (SlotArm(0.3), SlotArm(0.5), SlotArm(0.9))

# アルゴリズムの準備
epsilons = [0.0, 0.1, 0.2, 0.3, 0.4, 0.8]

#algos = (EpsilonGreedy(0.1),  UCB1())

for e in epsilons:
    
    algo = EpsilonGreedyAgent(e)
    # シミュレーションの実行
    results = play(algo, arms, 1000, 250)
    
    # グラフの表示
    df = pd.DataFrame({'times': results[0], 'rewards': results[1]})
    mean = df['rewards'].groupby(df['times']).mean()
    plt.plot(mean, label="epsilon={}".format(e)) 
    
# グラフの表示
plt.xlabel('Step')
plt.ylabel('Average Reward')
plt.legend(loc='best')
plt.title(algo.label())
plt.show()





# compare epsilon with UCB1
# アルゴリズムの準備
algos = (EpsilonGreedyAgent(0.1),  UCB1_Agent())

for algo in algos:
    # シミュレーションの実行
    results = play(algo, arms, 1000, 250)
    
    # グラフの表示
    df = pd.DataFrame({'times': results[0], 'rewards': results[1]})
    mean = df['rewards'].groupby(df['times']).mean()
    plt.plot(mean, label=algo.label()) 

# グラフの表示
plt.xlabel('Step')
plt.ylabel('Average Reward')
plt.legend(loc='best')
plt.show()


"""経験の蓄積と活用のバランス
if ε is 0.2:
  探索（行動）by 20% p,　活用 by 80% p 

ε-greedy方で行動するエージェント
"""
import numpy as np
import random
import math


# ε-greedyの計算処理の作成
class EpsilonGreedyAgent():
    # ε-greedyの計算処理の初期化
    def __init__(self, epsilon):
        self.epsilon = epsilon # 探索する確率

    # 試行回数と価値のリセット
    def initialize(self, n_arms):
        self.n = np.zeros(n_arms) # 各アームの試行回数
        self.v = np.zeros(n_arms) # 各アームの価値

    # 戦略（アームの選択）
    def policy(self):
        if self.epsilon > random.random():
            # ランダムにアームを選択
            return np.random.randint(0, len(self.v))
        else:
            # 価値が高いアームを選択
            return np.argmax(self.v)

    # アルゴリズムのパラメータの更新
    def update(self, chosen_arm, reward, t):
        # 選択したアームの試行回数に1加算
        self.n[chosen_arm] += 1

        # 選択したアームの価値の更新
        n = self.n[chosen_arm]
        v = self.v[chosen_arm]
        self.v[chosen_arm] = ((n-1) / float(n)) * v + (1 / float(n)) * reward

    # 文字列情報の取得
    def label(self):
        return 'ε-greedy'
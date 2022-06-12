import numpy as np
import math

# UCB1アルゴリズム
class UCB1_Agent():
    # 試行回数と成功回数と価値のリセット
    def initialize(self, n_arms): 
        self.n = np.zeros(n_arms) # 各アームの試行回数
        self.w = np.zeros(n_arms) # 各アームの成功回数
        self.v = np.zeros(n_arms) # 各アームの価値
    
    # アームの選択
    def policy(self):
        # nが全て1以上になるようにアームを選択
        for i in range(len(self.n)):
            if self.n[i] == 0:
                return i
        
        # 価値が高いアームを選択
        return np.argmax(self.v)
        
    # アルゴリズムのパラメータの更新
    def update(self, chosen_arm, reward, ｔ):
        # 選択したアームの試行回数に1加算
        self.n[chosen_arm] += 1

        # 成功時は選択したアームの成功回数に1加算
        if reward == 1.0:
            self.w[chosen_arm] += 1
        
        # 試行回数が0のアームの存在時は価値を更新しない
        for i in range(len(self.n)):
            if self.n[i] == 0:
                return
        
        # 各アームの価値の更新
        for i in range(len(self.v)):
            self.v[i] = self.w[i] / self.n[i] + (2 * math.log(t) / self.n[i]) ** 0.5
        
    # 文字列情報の取得
    def label(self):
        return 'ucb1'
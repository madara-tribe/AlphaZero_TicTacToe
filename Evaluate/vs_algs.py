import numpy as np
import random
import math

def random_action(state):
    legal_actions = state.legal_actions()
    return legal_actions[random.randint(0, len(legal_actions)-1)]

# アルファベータ法で状態価値計算
def alpha_beta(state, alpha, beta):
    # 負けは状態価値-1
    if state.is_lose():
        return -1
    
    # 引き分けは状態価値0
    if state.is_draw():
        return  0

    # 合法手の状態価値の計算    
    for action in state.legal_actions():
        score = -alpha_beta(state.next(action), -beta, -alpha)
        if score > alpha:
            alpha = score

        # 現ノードのベストスコアが親ノードを超えたら探索終了
        if alpha >= beta:
            return alpha

    # 合法手の状態価値の最大値を返す        
    return alpha

# アルファベータ法で行動選択
def alpha_beta_action(state):
    # 合法手の状態価値の計算
    best_action = 0
    alpha = -float('inf')
    for action in state.legal_actions():
        score = -alpha_beta(state.next(action), -float('inf'), -alpha)
        if score > alpha:
            best_action = action
            alpha = score

    # 合法手の状態価値の最大値を持つ行動を返す            
    return best_action

# プレイアウト（1ゲームのplay）
def playout(state):
    # 負けは状態価値-1
    if state.is_lose():
        return -1
    
    # 引き分けは状態価値0
    if state.is_draw():
        return  0
    
    # 次の状態の状態価値
    return -playout(state.next(random_action(state)))


# 最大値のインデックスを返す
def argmax(collection, key=None):
    return collection.index(max(collection))


# モンテカルロ木探索の行動選択
def advanced_mcts_action(state):
    # モンテカルロ木探索のノードの定義
    class Node:
        """ ノードの初期化（backup） """
        def __init__(self, state):
            self.state = state # 状態
            self.w = 0 # 累計価値
            self.n = 0 # 試行回数
            self.child_nodes = None  # 子ノード群

        """ 局面の価値の計算（＝評価(evaluation)）"""
        def playout_evaluation(self):
            # ゲーム終了時
            if self.state.is_done():
                # 勝敗結果で価値を取得
                value = -1 if self.state.is_lose() else 0 # 負けは-1、引き分けは0

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1
                return value

            # 子ノードが存在しない時
            if not self.child_nodes:
                # プレイアウトで価値を取得
                value = playout(self.state)

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1

                # 子ノードの展開
                if self.n == 10: 
                    self.expand()
                return value

            # 子ノードが存在する時
            else:
                # UCB1が最大の子ノードの評価で価値を取得
                value = -self.next_child_node_UCB1_Selection().playout_evaluation() 

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1
                return value

        """展開(expand if n=10, 子ノードの展開）"""
        def expand(self):
            legal_actions = self.state.legal_actions()
            self.child_nodes = []
            for action in legal_actions:
                self.child_nodes.append(Node(self.state.next(action)))

        """ 選択(selection)=>UCB1が最大の子ノードの取得"""
        def next_child_node_UCB1_Selection(self):
             # 試行回数が0の子ノードを返す
            for child_node in self.child_nodes:
                if child_node.n == 0:
                    return child_node

            # UCB1の計算
            t = 0
            for c in self.child_nodes:
                t += c.n
            ucb1_values = []
            for child_node in self.child_nodes:
                ucb1_values.append(-child_node.w/child_node.n+(2*math.log(t)/child_node.n)**0.5)

            # UCB1が最大の子ノードを返す
            return self.child_nodes[argmax(ucb1_values)]    
    
    # 現在の局面のノードの作成
    root_node = Node(state)
    root_node.expand()

    # 100回のシミュレーションを実行
    for _ in range(100):
        root_node.playout_evaluation()

    # 試行回数の最大値を持つ行動を返す
    legal_actions = state.legal_actions()
    n_list = []
    for c in root_node.child_nodes:
        n_list.append(c.n)
    return legal_actions[argmax(n_list)]
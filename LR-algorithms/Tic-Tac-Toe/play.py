from status import State
from mini_max_action import random_action, mini_max_action
from alpha_beta_action import alpha_beta_action
import random

# アルファベータ法とミニマックス法の対戦

# 状態の生成
state = State()

# ゲーム終了までのループ
while True:
    # ゲーム終了時
    if state.is_done():
        break

    # 行動の取得
    if state.is_first_player():
        action = alpha_beta_action(state) #　先手
    else:
        action = mini_max_action(state)   #　後手
        
    # 次の状態の取得
    state = state.next(action)

    # 文字列表示
    print(state)
    print()


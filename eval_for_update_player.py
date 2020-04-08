# ====================
# 新パラメータ評価部
# ====================

from Environment.state import State
from strategy.NN_MonteCarlo import select_action_by_pv_mcts
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from pathlib import Path
from shutil import copy
import numpy as np

# パラメータの準備
GAME_COUNT = 10 # 1評価あたりのゲーム数（本家は400）
BOLTZMAN_TEMP = 1.0 # ボルツマン分布の温度

# 先手プレイヤーのポイント
def first_player_point(ended_state):
    # 1:先手勝利, 0:先手敗北, 0.5:引き分け
    if ended_state.is_lose():
        return 0 if ended_state.is_first_player() else 1
    return 0.5

# 1ゲームの実行
def one_episode_play(next_actions):
    # 状態の生成
    state = State()

    # ゲーム終了までループ
    while True:
        # ゲーム終了時
        if state.is_done():
            break;

        # 行動の取得
        next_action = next_actions[0] if state.is_first_player() else next_actions[1]
        action = next_action(state)

        # 次の状態の取得
        state = state.next(action)

    # 先手プレイヤーのポイントを返す
    return first_player_point(state)

# change best player
def update_best_player():
    copy('./model/latest.h5', './model/best.h5')
    print('Change BestPlayer')


def total_episode_play():
    # load latest player
    model0 = load_model('./model/latest.h5')

    # load best layer 
    model1 = load_model('./model/best.h5')

    # select action by PV MCTS
    next_action0 = select_action_by_pv_mcts(model0, BOLTZMAN_TEMP)
    next_action1 = select_action_by_pv_mcts(model1, BOLTZMAN_TEMP)
    next_actions = (next_action0, next_action1)

    # multi play
    total_point = 0
    for i in range(GAME_COUNT):
        # 1 play
        if i % 2 == 0:
            total_point += one_episode_play(next_actions)
        else:
            total_point += 1 - one_episode_play(list(reversed(next_actions)))

        print('\rEvaluate for update player {}/{}'.format(i + 1, GAME_COUNT), end='')
    print('')

    # caluculate average point
    average_point = total_point / GAME_COUNT
    print('AveragePoint', average_point)

    # delete models
    K.clear_session()
    del model0
    del model1

    if average_point > 0.5:
        print('change best player')
        update_best_player()
        return True
    else:
        return False


if __name__ == '__main__':
    total_episode_play()

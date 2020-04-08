# ====================
# ベストプレイヤーの評価
# ====================

from Environment.state import State
from evaluate.vs_algs import random_action, alpha_beta_action, advanced_mcts_action
from strategy.NN_MonteCarlo import select_action_by_pv_mcts
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from pathlib import Path
import numpy as np

# パラメータの準備
GAME_COUNT = 10  # 1評価あたりのゲーム数

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
            break

        # 行動の取得
        next_action = next_actions[0] if state.is_first_player() else next_actions[1]
        action = next_action(state)

        # 次の状態の取得
        state = state.next(action)

    # 先手プレイヤーのポイントを返す
    return first_player_point(state)

# 任意のアルゴリズムの評価
def evaluate_algorithm_of(label, next_actions):
    # 複数回の対戦を繰り返す
    total_point = 0
    for i in range(GAME_COUNT):
        # 1ゲームの実行
        if i % 2 == 0:
            total_point += one_episode_play(next_actions)
        else:
            total_point += 1 - one_episode_play(list(reversed(next_actions)))

        # 出力
        print('\rEvaluate {}/{}'.format(i + 1, GAME_COUNT), end='')
    print('')

    # 平均ポイントの計算
    average_point = total_point / GAME_COUNT
    print(label, average_point)

# ベストプレイヤーの評価
def evaluate_best_player():
    # ベストプレイヤーのモデルの読み込み
    model = load_model('./model/best.h5')

    # PV MCTSで行動選択を行う関数の生成
    next_pv_mcts_action = select_action_by_pv_mcts(model, 0.0)

    # VSランダム
    next_actions = (next_pv_mcts_action, random_action)
    evaluate_algorithm_of('VS_Random', next_actions)

    # VSアルファベータ法
    next_actions = (next_pv_mcts_action, alpha_beta_action)
    evaluate_algorithm_of('VS_AlphaBeta', next_actions)

    # VSモンテカルロ木探索
    next_actions = (next_pv_mcts_action, advanced_mcts_action)
    evaluate_algorithm_of('VS_MCTS', next_actions)

    # モデルの破棄
    K.clear_session()
    del model
    
# 動作確認
if __name__ == '__main__':
    evaluate_best_player()

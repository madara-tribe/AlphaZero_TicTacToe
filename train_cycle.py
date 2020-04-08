# ====================
# 学習サイクルの実行
# ====================

from Agent.dual_network import dual_network
from self_play import self_play
from train_dual_network import create_latest_player
from eval_for_update_player import total_episode_play
from evaluate.evaluate_best_player import evaluate_best_player
train_network = create_latest_player()

# デュアルネットワークの作成
dual_network()

for i in range(10):
    print('Train',i,'====================')
    # セルフプレイ部
    self_play()

    # パラメータ更新部
    train_network()

    # 新パラメータ評価部
    update_best_player = total_episode_play()

    # ベストプレイヤーの評価
    if update_best_player:
        evaluate_best_player()
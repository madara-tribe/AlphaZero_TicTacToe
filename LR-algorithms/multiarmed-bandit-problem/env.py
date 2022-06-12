import random

"""ゲームの実装"""

# スロットのアームの作成
class SlotArm():
    # スロットのアームの初期化
    def __init__(self, p):
        self.p = p # コインが出る確率

    # アームを選択した時の報酬の取得
    def draw(self):
        if self.p > random.random() :
            return 1.0
        else:
            return 0.0
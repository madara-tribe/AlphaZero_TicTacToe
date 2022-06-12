import random

# ミニマックス法で状態価値計算
def cal_value_by_mini_max(state):
    # 負けは状態価値-1
    if state.is_lose():
        return -1
    
    # 引き分けは状態価値0
    if state.is_draw():
        return  0

    # 合法手の状態価値の計算
    best_score = -float('inf')
    for action in state.legal_actions():
        score = -cal_value_by_mini_max(state.next(action))
        if score > best_score:
            best_score = score
            
    # 合法手の状態価値の最大値を返す
    return best_score

# ミニマックス法で行動選択
def mini_max_action(state):
    # 合法手の状態価値の計算
    best_action = 0
    best_score = -float('inf')
    str = ['','']
    for action in state.legal_actions():
        score = -cal_value_by_mini_max(state.next(action))
        if score > best_score:
            best_action = action
            best_score  = score
            
        str[0] = '{}{:2d},'.format(str[0], action)
        str[1] = '{}{:2d},'.format(str[1], score)
    print('action:', str[0], '\nscore: ', str[1], '\n')

    # 合法手の状態価値の最大値を持つ行動を返す
    return best_action



# ランダムで行動選択
def random_action(state):
    legal_actions = state.legal_actions()
    return legal_actions[random.randint(0, len(legal_actions)-1)]
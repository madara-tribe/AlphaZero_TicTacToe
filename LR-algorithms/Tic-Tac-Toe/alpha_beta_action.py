import random

# アルファベータ法で状態価値計算
def cal_value_by_alpha_beta(state, alpha, beta):
    # 負けは状態価値-1
    if state.is_lose():
        return -1
    
    # 引き分けは状態価値0
    if state.is_draw():
        return  0

    # 合法手の状態価値の計算    
    for action in state.legal_actions():
        score = -cal_value_by_alpha_beta(state.next(action), -beta, -alpha)
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
    str = ['','']    
    for action in state.legal_actions():
        score = -cal_value_by_alpha_beta(state.next(action), -float('inf'), -alpha)
        if score > alpha:
            best_action = action
            alpha = score
            
        str[0] = '{}{:2d},'.format(str[0], action)
        str[1] = '{}{:2d},'.format(str[1], score)
    print('action:', str[0], '\nscore: ', str[1], '\n')            

    # 合法手の状態価値の最大値を持つ行動を返す
    return best_action
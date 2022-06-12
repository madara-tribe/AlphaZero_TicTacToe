import numpy as np
from value_function import get_policy, get_s_next, get_a_value_func


# 1エピソードの実行
def play(Q, epsilon, pi, use_sarsa=True):
    s = 0 # 状態
    a = a_next = get_a_value_func(s, Q, epsilon, pi) # 行動の初期値
    s_a_history = [[0, np.nan]] # 状態と行動の履歴

    # エピソード完了までループ
    while True:
        # 行動に従って次の状態の取得
        a = a_next
        s_next = get_s_next(s, a)

        # 履歴の更新
        s_a_history[-1][1] = a
        s_a_history.append([s_next, np.nan])

        # 終了判定
        if s_next == 8:
            r = 1
            a_next = np.nan
        else:
            r = 0
            # 行動価値関数Qに従って行動の取得
            a_next = get_a_value_func(s_next, Q, epsilon, pi)

        # 行動価値関数の更新
        if use_sarsa:
            Q = sarsa(s, a, r, s_next, a_next, Q)
        else:
            Q = q_learning(s, a, r, s_next, a_next, Q)

        # 終了判定
        if s_next == 8:
            break
        else:
            s = s_next

    # 履歴と行動価値関数を返す
    return [s_a_history, Q]



# Q学習による行動価値関数の更新
def q_learning(s, a, r, s_next, a_next, Q):
    eta = 0.1 # 学習係数
    gamma = 0.9 # 時間割引率
    
    if s_next == 8:
        Q[s, a] = Q[s, a] + eta * (r - Q[s, a])
    else:
        Q[s, a] = Q[s, a] + eta * (r + gamma * np.nanmax(Q[s_next, :]) - Q[s, a])
    return Q

# Sarsaによる行動価値関数の更新
def sarsa(s, a, r, s_next, a_next, Q):
    eta = 0.1 # 学習係数
    gamma = 0.9 # 時間割引率
    
    if s_next == 8:
        Q[s, a] = Q[s, a] + eta * (r - Q[s, a])
    else:
        Q[s, a] = Q[s, a] + eta * (r + gamma * Q[s_next, a_next] - Q[s, a])
    return Q
import numpy as np

# パラメータθを方策に変換
def get_policy(theta):
    # 割合の計算
    [m, n] = theta.shape
    pi = np.zeros((m, n))
    for i in range(0, m):
        pi[i, :] = theta[i, :] / np.nansum(theta[i, :])
    pi = np.nan_to_num(pi)    
    return pi


# 行動に従って次の状態を取得
def get_s_next(s, a):
    if a == 0: # 上
        return s - 3
    elif a == 1: # 右
        return s + 1
    elif a == 2: # 下
        return s + 3
    elif a == 3: # 左
        return s - 1


# ランダムまたは行動価値関数に従って行動を取得
def get_a_value_func(s, Q, epsilon, pi_0):
    if np.random.rand() < epsilon:
        # ランダムに行動を選択
        return np.random.choice([0, 1, 2, 3], p=pi_0[s])
    else:
        # 行動価値関数で行動を選択
        return np.nanargmax(Q[s])

# 行動価値関数の準備
def set_default_value(theta_0):
  [a, b] = theta_0.shape
  Q = np.random.rand(a, b) * theta_0 * 0.01
  return Q
import numpy as np


# パラメータθを方策に変換
def get_policy(theta):
    # ソフトマックス関数で変換
    [m, n] = theta.shape
    pi = np.zeros((m, n))
    exp_theta = np.exp(theta)
    for i in range(0, m):
        pi[i, :] = exp_theta[i, :] / np.nansum(exp_theta[i, :])
    pi = np.nan_to_num(pi)
    return pi


def update_theta(theta, pi, s_a_history):
    eta = 0.1 # 学習係数
    total = len(s_a_history) - 1 # ゴールまでにかかった総ステップ数
    [s_count, a_count] = theta.shape # 状態数, 行動数

    # パラメータθの変化量の計算
    delta_theta = theta.copy()
    for i in range(0, s_count):
        for j in range(0, a_count):
            if not(np.isnan(theta[i, j])):
                # ある状態である行動を採る回数
                sa_ij = [sa for sa in s_a_history if sa == [i, j]]
                n_ij = len(sa_ij)

                # ある状態でなんらかの行動を採る回数
                sa_i = [sa for sa in s_a_history if sa[0] == i]
                n_i = len(sa_i)

                # パラメータθの変化量
                delta_theta[i, j] = (n_ij - pi[i, j] * n_i) / total

    # パラメータθの更新
    return theta + eta * delta_theta
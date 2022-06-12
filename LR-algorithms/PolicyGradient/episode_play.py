import numpy as np 


# 方策に従って行動を取得
def get_a(pi, s):
    # 方策の確率に従って行動を返す
    return np.random.choice([0, 1, 2, 3], p=pi[s])


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

# 1エピソード実行して履歴取得
def play(pi):
    s = 0 # 状態
    s_a_history = [[0, np.nan]] # 状態と行動の履歴
    
    # エピソード完了までループ
    while True:
        # 方策に従って行動を取得
        a = get_a(pi, s)
        
        # 行動に従って次の状態を取得
        s_next = get_s_next(s, a)
        
        # 履歴の更新
        s_a_history[-1][1] = a     
        s_a_history.append([s_next, np.nan]) 
        
        # 終了判定
        if s_next == 8:
            break
        else:
            s = s_next
            
    return s_a_history
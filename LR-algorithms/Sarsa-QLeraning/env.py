import matplotlib.pyplot as plt

def set_env():
    # 迷路の作成
    fig = plt.figure(figsize=(3, 3))

    # 壁
    plt.plot([0, 3], [3, 3], color='k')
    plt.plot([0, 3], [0, 0], color='k')
    plt.plot([0, 0], [0, 2], color='k')
    plt.plot([3, 3], [1, 3], color='k')
    plt.plot([1, 1], [1, 2], color='k')
    plt.plot([2, 3], [2, 2], color='k')
    plt.plot([2, 1], [1, 1], color='k')
    plt.plot([2, 2], [0, 1], color='k')

    # 数字
    for i in range(3):
        for j in range(3):
            plt.text(0.5+i, 2.5-j, str(i+j*3), size=20, ha='center', va='center')

    # 円
    circle, = plt.plot([0.5], [2.5], marker='o', color='g', markersize=40)

    # 目盛りと枠の非表示
    plt.tick_params(axis='both', which='both', bottom='off', top= 'off',
            labelbottom='off', right='off', left='off', labelleft='off')
    plt.box('off')
    return fig, circle
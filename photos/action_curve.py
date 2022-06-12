
import numpy as np
import matplotlib.pyplot as plt

with np.errstate(invalid="ignore"):
    x = np.arange(-1, 15, 0.01)
    y = np.log10(x)-0.5

    y += np.random.randn(len(y)) * np.linspace(0.2, 0.8, len(y))

    # v = -np.log10(x)
    plt.xlim([-1, 15])
    plt.ylim([-5, 5])
    plt.xlabel("x")
    plt.ylabel("y", rotation=0)
    plt.gca().set_aspect("equal")
    plt.grid()
    plt.plot(x, -y)
    plt.plot(x, y)
    plt.show()


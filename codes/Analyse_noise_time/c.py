import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y = np.random.randn(100)
plt.plot(x, y)
plt.title('Random-Walk')
plt.show()

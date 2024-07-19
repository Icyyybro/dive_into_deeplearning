import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib_inline import backend_inline
from d2l import torch as d2l



def f(x):
    return 3 * x ** 2 - 4 * x

def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h

h = 0.1
for i in range(5):
    print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
    h *= 0.1


#绘制图形 f(x)=2x-3
x = np.arange(0, 3, 0.1)
d2l.plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
plt.show()



#绘制函数 f(x)=3*x*x+5*pow(e,x2)
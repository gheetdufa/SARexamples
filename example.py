
#Equation 1
import torch
from IPython.display import Image as IPImage
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
import time


def rect(x):
    ry = [1 if .5 >= abs(i) else 0 for i in x] 

    return ry
  
def sinc(x):
  return np.sin(np.pi * x) / (np.pi * x)
  
def s(t, T, K):
  return rect(t/T) * np.exp(1j*np.pi*K*(t**2))

def s_r(t, t_0, T, K):
  return s(t-t_0, T, K)

def s_out(t, t_0, T, K):
  return T * sinc(K*T*(t - t_0))
  
def phi(t, K):
  return np.pi * K * t**2

T = 7.24 * 10**-6
t = np.linspace(-T,T, 1000)
K = 5.8* 10**6 / T
t_0 = T/2

fig, axes = plt.subplots(2,3)

fig.suptitle(f"T = ${T *10 **6}, K = ${K}")

axes[0,0].plot(t, s(t, T, K).real)
axes[0,0].set_title("Time domain Reresentation")
axes[1,0].plot(t, phi(t, K))
axes[1,0].set_title("signal phase")
axes[0,1].plot(t, s_r(t,t_0, T, K))
axes[1,1].plot(t, s_out(t, t_0, T, K))
axes[1,2].plot(t, s_out(t, 0, T, K))


plt.subplots_adjust(hspace = 0.5)
plt.show()


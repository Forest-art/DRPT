import torch
import numpy as np
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


##  =A1&","&A2&","&A3&","&A4&","&A5&","&A6&","&A7&","&A8&","&A9&","&A10&","&A11&","&A12&","&A13&","&A14&","&A15&","&A16&","&A17&","&A18&","&A19&","&A20&","&A21&","&A22


font_path = '/usr/share/fonts/fonts/Arial Bold.ttf'
fm.fontManager.addfont(font_path)
# plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
new_font = fm.FontProperties(fname=font_path)



K = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
W = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
UT_AUC_K = [35.4, 32.1, 33.1, 38.5, 36.7, 37.9, 38.1, 37.7, 35.8, 36.7, 35.5]
UT_AUC_W = [35.4, 38.2, 38.7, 39.1, 38.5, 38.8, 38.4, 38.1, 38.1, 38.0, 37.9]
CGQA_AUC = [6.2, 6.1, 5.9, 6.1, 6.1, 6.5, 5.9, 6.0, 6.1, 6.2, 5.9]
plt.plot(W, UT_AUC_W, 'bo-')
# plt.plot(K, CGQA_AUC, 'ro-', label="C-QGA")



plt.xlabel("Weight(-)", font = new_font)
plt.ylabel("AUC", font = new_font)
plt.title('Hyperparameters analysis of W on UT-Zappos', font = new_font)
# plt.legend(prop = new_font)   #打上标签
plt.grid(True, linestyle='--')
plt.savefig('demos/AUC.pdf', dpi=300)
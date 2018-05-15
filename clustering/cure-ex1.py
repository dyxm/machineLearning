# Created by [Yuexiong Ding] at 2018/4/12
# 
#

import matplotlib.pyplot as plt


x = [37, 50, 28, 102, 110, 46, 109, 53, 40, 103, 37, 100]
y = [29, 38, 28, 18, 26, 27, 36, 30, 19, 49, 38, 30]

mean_x = [49.66666667, 35.5]
# mean_x2 = []
mean_y = [31.66666667, 28.5]
# mean_y2 = [28.5]

rep_x = [47.83333334, 51.33333334, 49.83333334]
rep_y = [29.33333334, 30.83333334, 34.83333334]

rep_x2 = [37.75, 36.25, 31.75]
rep_y2 = [23.75, 33.25, 28.25]

plt.scatter(x, y)
plt.scatter(mean_x, mean_y, c='red')
plt.scatter(rep_x, rep_y, c='orange')
plt.scatter(rep_x2, rep_y2, c='orange')
plt.show()



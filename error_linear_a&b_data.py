import numpy as np
import matplotlib
import seaborn as sb
import matplotlib.pyplot as plt


#  linear error data

vegetables = ["0.05", "0.04", "0.03", "0.02", "0.01", "0", "-0.01", "-0.02",
              "-0.03", "-0.04"] # x
farmers = ["0.03", "0.02", "0.01",
           "0", "-0.01", "-0.02", "-0.03", "-0.04", "-0.05", "-0.06"]

harvest = np.array([[11.22, 9.92, 9.85, 9.80, 9.82, 9.78, 9.76, 0, 11.00, 0],
                    [9.95, 9.87, 9.85, 9.80, 0, 9.69, 9.66, 9.65, 10.88, 0],
                    [9.93, 9.84, 8.54, 8.56, 8.31, 8.32, 9.46, 9.55, 0, 0],
                    [9.92, 8.59, 8.45, 7.12, 7.08, 8.21, 8.17, 9.49, 0, 0],
                    [08.69, 8.52, 7.45, 8.03, 8.14, 8.13, 0, 0, 0, 0],
                    [7.45, 7.44, 6.18, 6.14, 7.94, 7.97, 9.23, 0, 0, 0],
                    [7.46, 7.48, 7.58, 6.24, 6.53, 7.74, 0, 0, 0, 0],
                    [8.88, 8.89, 8.91, 0, 8.96, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

ax = sb.heatmap(harvest, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Time [sec]'})
ax.set_xticks(np.arange(len(farmers)))
ax.set_yticks(np.arange(len(vegetables)))
ax.set_xticklabels(farmers)
ax.set_yticklabels(vegetables)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
plt.xlabel("y [m]")
plt.ylabel("x [m]")
ax.set_title("linear error")
plt.show()

#  a/b error data

vegetables2 = ["0.0002", "0.0004", "0.0006", "0.0008", "0.0010", "0.0012", "0.0014", "0.0016", "0.0018", "0.0020"] # a
farmers2 = ["0.0002", "0.0004", "0.0006", "0.0008", "0.0010", "0.0012", "0.0014", "0.0016", "0.0018", "0.0020"] #b

harvest2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 10.47],
                    [0, 0, 0, 0, 0, 0, 9.27, 8.05, 8.07, 8.07],
                    [9.35, 9.26, 9.25, 8.07, 8.02, 8.05, 8.03, 8.07, 8.05, 8.06],
                    [9.21, 9.28, 9.23, 8.07, 8.07, 8.03, 8.01, 8.03, 8.07, 8.09],
                    [8.05, 8.15, 8.07, 9.24, 8.06, 8.06, 8.13, 8.04, 8.10, 7.99],
                    [8.03, 8.05, 0, 7.99, 8.06, 8.06, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

ax2 = sb.heatmap(harvest2, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Time [sec]'})
ax2.set_xticks(np.arange(len(farmers2)))
ax2.set_yticks(np.arange(len(vegetables2)))
ax2.set_xticklabels(farmers2)
ax2.set_yticklabels(vegetables2)

plt.xlabel("a")
plt.ylabel("b")
ax2.set_title("spiral parameters")
plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
plt.show()
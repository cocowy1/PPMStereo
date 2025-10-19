import matplotlib.pyplot as plt

# 创建一个包含三个子图的figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 子图 (a) 的数据
x_a = [0.55, 0.7, 0.85]
y_a = [12.2, 21.5, 11.1]
labels_a = ["Ours (12.2)", "DynamicStereo (21.5)", "RAFTStereo (11.1)"]
markers_a = ['*', 's', '^']
colors_a = ['red', 'saddlebrown', 'forestgreen']

# 子图 (b) 的数据
x_b = [0.55, 0.7, 0.85]
y_b = [39.7, 35.3, 16.0]
labels_b = ["Ours (39.7)", "DynamicStereo (35.3)", "RAFTStereo (16.0)"]
markers_b = ['*', 's', '^']
colors_b = ['red', 'saddlebrown', 'forestgreen']

# 子图 (c) 的数据
x_c = [10.5, 12.2, 14.0]
y_c = [2.4, 1.5, 1.3]
labels_c = ["Ours (2.4)", "DynamicStereo (1.5)", "RAFTStereo (1.3)"]
markers_c = ['*', 's', '^']
colors_c = ['red', 'saddlebrown', 'forestgreen']

# 绘制子图 (a)
axes[0].scatter(x_a, y_a, marker=markers_a[0], c=colors_a[0], s=100)
axes[0].scatter(x_a[1], y_a[1], marker=markers_a[1], c=colors_a[1], s=100)
axes[0].scatter(x_a[2], y_a[2], marker=markers_a[2], c=colors_a[2], s=100)
for i, label in enumerate(labels_a):
    axes[0].annotate(label, (x_a[i], y_a[i]), textcoords="offset points", xytext=(5, 5), ha='left')
axes[0].set_xlabel("Dynamic Replica δ₁px", fontsize=12)
axes[0].set_ylabel("Parameters (M)", fontsize=12)
axes[0].set_title("(a)", fontsize=14)

# 绘制子图 (b)
axes[1].scatter(x_b, y_b, marker=markers_b[0], c=colors_b[0], s=100)
axes[1].scatter(x_b[1], y_b[1], marker=markers_b[1], c=colors_b[1], s=100)
axes[1].scatter(x_b[2], y_b[2], marker=markers_b[2], c=colors_b[2], s=100)
for i, label in enumerate(labels_b):
    axes[1].annotate(label, (x_b[i], y_b[i]), textcoords="offset points", xytext=(5, 5), ha='left')
axes[1].set_xlabel("Dynamic Replica δ₁px", fontsize=12)
axes[1].set_ylabel("Inference GPU Memory (G)", fontsize=12)
axes[1].set_title("(b)", fontsize=14)

# 绘制子图 (c)
axes[2].scatter(x_c, y_c, marker=markers_c[0], c=colors_c[0], s=100)
axes[2].scatter(x_c[1], y_c[1], marker=markers_c[1], c=colors_c[1], s=100)
axes[2].scatter(x_c[2], y_c[2], marker=markers_c[2], c=colors_c[2], s=100)
for i, label in enumerate(labels_c):
    axes[2].annotate(label, (x_c[i], y_c[i]), textcoords="offset points", xytext=(5, 5), ha='left')
axes[2].set_xlabel("Sintel Final δ₁px", fontsize=12)
axes[2].set_ylabel("MACs (T)", fontsize=12)
axes[2].set_title("(c)", fontsize=14)

# 调整布局，显示图形
plt.tight_layout()
plt.show()
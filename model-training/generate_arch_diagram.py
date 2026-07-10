"""
生成系统架构图 — 保存到 checkpoints/architecture.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

# 中文字体
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
matplotlib.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(20, 11))
ax.set_xlim(0, 20)
ax.set_ylim(0, 11)
ax.axis('off')

# ===== 配色 =====
LAYER_COLORS = {
    'user':   {'bg': '#FFF8E1', 'border': '#F9A825'},
    'logic':  {'bg': '#E8F5E9', 'border': '#43A047'},
    'model':  {'bg': '#E3F2FD', 'border': '#1E88E5'},
}
BOX_COLORS = {
    'white':  {'bg': '#FFFFFF', 'border': '#BDBDBD'},
    'green':  {'bg': '#E8F5E9', 'border': '#43A047'},
    'blue':   {'bg': '#E3F2FD', 'border': '#1E88E5'},
    'orange': {'bg': '#FFF3E0', 'border': '#FB8C00'},
    'red':    {'bg': '#FFEBEE', 'border': '#E53935'},
}
TEXT_DARK  = '#333333'
TEXT_GRAY  = '#757575'

def add_layer_bg(x, y, w, h, label, color_key):
    """绘制层背景"""
    c = LAYER_COLORS[color_key]
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                          facecolor=c['bg'], edgecolor=c['border'],
                          linewidth=1.5, linestyle='-', alpha=0.4, zorder=0)
    ax.add_patch(rect)
    ax.text(x + 0.4, y + h - 0.35, label, fontsize=14, fontweight='bold',
            color=c['border'], zorder=1)

def box(x, y, w, h, title, subtitle='', color_key='white', fs_title=11, fs_sub=9):
    """绘制模块框"""
    c = BOX_COLORS[color_key]
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                       facecolor=c['bg'], edgecolor=c['border'],
                       linewidth=1.8, zorder=2)
    ax.add_patch(b)
    if subtitle:
        ax.text(x + w/2, y + h*0.7, title, ha='center', va='center',
                fontsize=fs_title, fontweight='bold', color=TEXT_DARK, zorder=3)
        ax.text(x + w/2, y + h*0.25, subtitle, ha='center', va='center',
                fontsize=fs_sub, color=TEXT_GRAY, zorder=3)
    else:
        ax.text(x + w/2, y + h/2, title, ha='center', va='center',
                fontsize=fs_title, fontweight='bold', color=TEXT_DARK, zorder=3)

def arrow(x1, y1, x2, y2, color='#78909C', lw=2):
    """绘制连接箭头"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle='arc3,rad=0'))

def v_arrow(x, y1, y2, color='#78909C', lw=2):
    """垂直箭头 (向下)"""
    arrow(x, y1, x, y2, color, lw)

# ===================== 图层背景 =====================
add_layer_bg(0.3, 7.6, 19.4, 3.0, '用户交互层', 'user')
add_layer_bg(0.3, 3.6, 19.4, 3.7, '服务逻辑层', 'logic')
add_layer_bg(0.3, 0.3, 19.4, 3.0, '模型与数据层', 'model')

# ===================== 第一层：用户交互 =====================
box(1.0, 8.3, 4.8, 1.6, 'Web 前端界面',
    '拖拽上传 · 结果展示 · 相似图预览 · 粘贴识别', 'orange', 12, 9)
box(7.2, 8.3, 4.8, 1.6, 'RESTful API 服务',
    'POST /predict  ·  GET /health  ·  GET /images', 'orange', 12, 9)
box(13.4, 8.3, 4.8, 1.6, 'Swagger 交互文档',
    '/docs 在线调试  ·  接口说明  ·  一键测试', 'orange', 12, 9)

# 用户层横向箭头
arrow(5.8, 9.1, 7.0, 9.1)
arrow(12.0, 9.1, 13.2, 9.1)

# ===================== 第二层：服务逻辑 =====================
# 流程线：预处理 → 推理 → 检索
box(1.0, 5.0, 3.6, 1.8, '图像预处理',
    'Resize 224×224\nImageNet 归一化\n格式校验', 'green', 11, 8)
box(5.6, 5.0, 3.6, 1.8, 'ResNet50 推理',
    'Top-5 分类输出\n置信度计算\n特征向量提取', 'green', 11, 8)
box(10.2, 5.0, 3.6, 1.8, '相似病例检索',
    'L2 归一化\n余弦相似度\nTop-K 匹配', 'green', 11, 8)

# 知识库
box(6.2, 4.0, 6.0, 0.9, '知识库匹配：36 类病害 → 防治建议 + 推荐农药',
    '', 'white', 11, 9)

# 降级路径
box(14.8, 5.0, 3.6, 1.8, 'ONNX 降级推理',
    'CPU 端侧兼容\n跨平台部署\n无 GPU 可用时', 'red', 11, 8)

# 服务层箭头
arrow(4.6, 5.9, 5.4, 5.9)
arrow(9.2, 5.9, 10.0, 5.9)
arrow(13.8, 5.9, 14.6, 5.9, '#EF5350')
# 推理 → 知识库
arrow(7.4, 5.0, 7.4, 5.1)
# 推理 → 检索（虚线，表示特征传递）
ax.annotate('特征向量', xy=(11.0, 5.2), xytext=(8.6, 5.2),
            fontsize=8, color='#78909C', ha='center', va='bottom')

# ===================== 第三层：模型与数据 =====================
box(1.0, 1.0, 5.2, 1.8, 'PyTorch 训练模型',
    'best_model.pth (271MB)\nGPU 推理  ·  训练/微调  ·  全精度', 'blue', 12, 9)
box(7.2, 1.0, 5.2, 1.8, 'ONNX 部署模型',
    'model.onnx (91MB)\n端侧部署  ·  昇腾 310  ·  跨平台', 'blue', 12, 9)
box(13.4, 1.0, 5.2, 1.8, '特征索引库',
    'feature_index.pt (248MB)\n31,541 张  ×  2,048 维  ·  L2 归一化', 'blue', 12, 9)

# ===================== 垂直连接 =====================
# 用户层 → 服务层
v_arrow(3.4, 7.6, 6.8)
v_arrow(9.6, 7.6, 6.8)
v_arrow(16.0, 7.6, 6.8)

# 服务层 → 模型层
v_arrow(3.6, 3.6, 2.8)
v_arrow(9.8, 3.6, 2.8)
v_arrow(16.0, 3.6, 2.8)

# ===================== 标注 =====================
# 数据流说明
ax.text(19.2, 5.5, '输入\n图片', fontsize=9, color=TEXT_GRAY, ha='center', fontweight='bold')
arrow(19.2, 6.0, 19.2, 6.5, '#BDBDBD', 1.5)

ax.text(19.2, 2.3, '加载\n模型', fontsize=9, color=TEXT_GRAY, ha='center', fontweight='bold')
arrow(19.2, 2.8, 19.2, 3.3, '#BDBDBD', 1.5)

# ===================== 标题 =====================
ax.text(10, 10.5, '智慧农业 AI 病虫害识别系统 — 技术架构图',
        ha='center', fontsize=22, fontweight='bold', color='#2E7D32')
ax.text(10, 10.05, 'ResNet50 · PyTorch/ONNX 双后端 · 36 类作物病害 · 相似病例检索',
        ha='center', fontsize=11, color=TEXT_GRAY)

plt.tight_layout(pad=0.5)
plt.savefig('checkpoints/architecture.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("[OK] architecture.png saved")

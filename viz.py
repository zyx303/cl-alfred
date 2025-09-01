import torch
from torchviz import make_dot
import matplotlib.pyplot as plt

def visualize_model_graph(model, feat, save_path='model_graph.png'):
    """
    使用 torchviz 可视化模型的计算图
    """
    # 确保模型在评估模式
    model.eval()
    
    # 前向传播
    with torch.no_grad():
        output = model.step(feat)
    
    # 创建计算图
    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.render(save_path, format='png')
    print(f"模型计算图已保存到: {save_path}")
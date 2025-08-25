from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
import debugpy
import os 
if os.getenv("debug"):
    debugpy.listen(5678)
    debugpy.wait_for_client()

import ai2thor.controller
from PIL import Image
import numpy as np

# 1. 以 CloudRendering 模式初始化控制器
#    - 如果是第一次运行，这里会触发下载，可能需要一些时间。
#    - scene='FloorPlan1' 是一个简单的测试场景。
print("正在以 CloudRendering 模式启动 AI2-THOR 控制器...")
try:
    controller = ai2thor.controller.Controller(
        platform=ai2thor.platform.CloudRendering,
        scene='FloorPlan1',
        gridSize=0.25,
        renderDepthImage=True,
        renderInstanceSegmentation=True,
        width=300,
        height=300
    )
    print("控制器启动成功！")
except Exception as e:
    print(f"控制器启动失败: {e}")
    exit()

# # 2. 验证模式是否正确
# print("\n--- 模式验证 ---")
# # 检查 platform 属性的类型
# is_cloud_rendering = isinstance(controller.platform, ai2thor.platform.CloudRendering)
# if is_cloud_rendering:
#     print("✅ 验证成功：控制器确实在 CloudRendering 模式下运行。")
# else:
#     print(f"❌ 验证失败：控制器正在以 {type(controller.platform)} 模式运行。")
#     controller.stop()
#     exit()

# 3. 执行一些动作来改变视角
print("\n--- 执行动作 ---")
print("动作 1: 向前移动 (MoveAhead)")
event = controller.step(action='MoveAhead')
print("动作 2: 向下看 (LookDown)")
event = controller.step(action='LookDown')


# 4. 打印最终的视角信息
print("\n--- 打印当前视角信息 ---")
# 从 event.metadata 中提取 agent 的状态
agent_metadata = event.metadata.get('agent', {})
position = agent_metadata.get('position', 'N/A')
rotation = agent_metadata.get('rotation', 'N/A')
horizon = agent_metadata.get('cameraHorizon', 'N/A')

print(f"Agent 位置 (position): {position}")
print(f"Agent 旋转 (rotation): {rotation}")
print(f"摄像头俯仰角度 (cameraHorizon): {horizon:.2f}")


# 5. 验证视觉输出并保存图像
print("\n--- 验证视觉输出 ---")
# event.frame 是一个 numpy 数组，代表 RGB 图像
current_frame = event.frame

if current_frame is not None and isinstance(current_frame, np.ndarray):
    print(f"✅ 成功获取图像帧，形状为: {current_frame.shape}, 数据类型为: {current_frame.dtype}")
    
    # 使用 Pillow 库将 numpy 数组保存为图片
    image_path = "cloudrendering_view.png"
    try:
        img = Image.fromarray(current_frame)
        img.save(image_path)
        print(f"✅ 当前视角已成功保存为图片: '{image_path}'")
    except Exception as e:
        print(f"❌ 保存图片时出错: {e}")
else:
    print("❌ 未能获取有效的图像帧。")


# 6. 停止控制器
print("\n停止控制器...")
controller.stop()
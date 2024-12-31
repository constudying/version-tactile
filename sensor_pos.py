import mujoco_py
import os
import glfw

# 设置模型文件路径
model_path = "d:/studio/项目/act-main/assets/vx300s_right.xml"

# 加载模型
model = mujoco_py.load_model_from_path(model_path)
sim = mujoco_py.MjSim(model)

# 创建渲染器
viewer = mujoco_py.MjViewer(sim)

# 初始传感器位置
sensor_pos = [0.0687, 0, 0]

def update_sensor_position():
    # 更新传感器位置
    model.site_pos[model.site_name2id('left_finger_force_sensor')] = sensor_pos
    model.site_pos[model.site_name2id('right_finger_force_sensor')] = sensor_pos

# 键盘回调函数
def key_callback(window, key, scancode, action, mods):
    global sensor_pos
    if action == glfw.PRESS or action == glfw.REPEAT:
        if key == glfw.KEY_W:
            sensor_pos[0] += 0.001
        elif key == glfw.KEY_S:
            sensor_pos[0] -= 0.001
        elif key == glfw.KEY_A:
            sensor_pos[1] -= 0.001
        elif key == glfw.KEY_D:
            sensor_pos[1] += 0.001
        elif key == glfw.KEY_Q:
            sensor_pos[2] += 0.001
        elif key == glfw.KEY_E:
            sensor_pos[2] -= 0.001
        update_sensor_position()

# 设置键盘回调
glfw.set_key_callback(viewer.window, key_callback)

# 运行模拟并渲染
while True:
    sim.step()
    viewer.render()
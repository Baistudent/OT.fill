import os

import bpy
from mathutils import Vector
from math import pi, log10

# 模板脚本：将两个带有 MTL 材质的 OBJ 模型导入场景并渲染单帧图像
# 使用时请在下方的常量配置区填写实际文件路径与渲染参数

OBJ_A_PATH = r"E:\OT.fill\data\pot\pot.obj"  # 第一个 OBJ 文件路径（需配套 MTL、纹理）
OBJ_B_PATH = r"E:\OT.fill\data\pot\2048_0.10\pot_2048_0.10.obj"  # 第二个 OBJ 文件路径（需配套 MTL、纹理）
OUTPUT_PATH = r"E:\OT.fill\data\pot\2048_0.10\render"  # 渲染输出图像路径（PNG）

CAMERA_LOCATION = (100.0, 100.0, 55.0)  # 摄像机在世界坐标中的位置
CAMERA_TARGET = (0.0, 0.0, 0.0)  # 摄像机对准的目标点坐标

LIGHT_LOCATION = (100.0, 100.0, 100.0)  # 太阳光位置（主要用于确定光照方向）
LIGHT_TARGET = (0.0, 0.0, 0.0)  # 太阳光照射的目标点
LIGHT_ENERGY = 3.0  # 太阳光能量系数（可调节整体亮度）

IMAGE_WIDTH = 2048  # 输出图像的宽度（像素）
IMAGE_HEIGHT = 2048  # 输出图像的高度（像素）
IMAGE_PERCENT = 100  # 分辨率缩放百分比（100 表示满分辨率）

RENDER_FRAMES = [0, 60, 120, 180, 240, 300, 360]  # 需要输出的关键帧列表


def clear_scene():
    """清空当前场景中的对象、网格、材质和图片数据，保证渲染环境干净。"""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        bpy.data.materials.remove(block)
    for block in bpy.data.images:
        bpy.data.images.remove(block)


def import_obj(path: str):
    """导入给定路径的 OBJ 文件并对导入对象执行归零与原点重设。"""
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.wm.obj_import(filepath=os.path.abspath(path))
    imported = list(bpy.context.selected_objects)
    place_objects_at_origin(imported)
    return imported


def place_objects_at_origin(objects):
    """将导入的对象移动到场景原点、重置原点但保留导入时的旋转。"""
    for obj in objects:
        if obj.type == "MESH":
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='MEDIAN')
            obj.location = (0.0, 0.0, 0.0)


def add_root_empty():
    """创建一个原点处的空对象，作为两个模型的父级。"""
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.empty_add(
        type='PLAIN_AXES',
        radius=1.0,
        align='WORLD',
        location=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
    )
    empty = bpy.context.object
    empty.name = "ModelsRoot"
    return empty


def parent_objects(parent_obj, children):
    """将导入的对象绑定到指定的空对象上。"""
    for child in children:
        child.parent = parent_obj


def set_objects_visibility(objects, hidden):
    """批量设置对象的渲染与视图可见性。"""
    for obj in objects:
        obj.hide_render = hidden
        obj.hide_viewport = hidden


def animate_root_rotation(root_obj):
    """为空物体设置 Z 轴旋转动画，从 0 到 360 度。"""
    scene = bpy.context.scene
    scene.frame_set(0)
    root_obj.rotation_euler = (0.0, 0.0, 0.0)
    root_obj.keyframe_insert(data_path="rotation_euler", index=-1)

    scene.frame_set(360)
    root_obj.rotation_euler = (0.0, 0.0, 2 * pi)
    root_obj.keyframe_insert(data_path="rotation_euler", index=-1)

    scene.frame_start = 0
    scene.frame_end = 360


def add_camera(location, target):
    """在指定位置创建摄像机并指向目标点。"""
    bpy.ops.object.camera_add(location=location)
    camera = bpy.context.object
    camera.name = "RenderCamera"

    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 320
    camera.rotation_euler[0] = 70 / 180 * pi
    camera.rotation_euler[1] = 0 / 180 * pi
    camera.rotation_euler[2] = 135 / 180 * pi
    bpy.context.scene.camera = camera


def add_key_light(location, target, energy):
    """创建太阳光并指向原点，模拟日光照明。"""
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.light_add(type="SUN", align="WORLD", location=location)
    light = bpy.context.object
    light.name = "KeyLight"
    light.data.energy = energy
    # 将太阳光的方向对准目标点，确保照射原点
    direction = Vector(target) - light.location
    light.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    light.data.angle = 0.5  # 太阳光角度，数值越小阴影越锐利


def configure_render(output_path: str):
    """切换到 Cycles 渲染器并设置输出路径与格式。"""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = 'GPU'
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = os.path.abspath(output_path)
    scene.render.resolution_x = IMAGE_WIDTH
    scene.render.resolution_y = IMAGE_HEIGHT
    scene.render.resolution_percentage = IMAGE_PERCENT


def resolve_output_dir(path: str) -> str:
    """根据用户提供的路径推断渲染输出目录。"""
    if not path:
        return os.getcwd()

    absolute = os.path.abspath(path)
    if os.path.isdir(absolute):
        return absolute

    directory = os.path.dirname(absolute)
    return directory if directory else os.getcwd()


def compute_psnr(image_path_a: str, image_path_b: str) -> float:
    """对两张图像计算 PSNR 值并返回结果。"""
    img_a = bpy.data.images.load(image_path_a, check_existing=False)
    img_b = bpy.data.images.load(image_path_b, check_existing=False)

    try:
        if img_a.size[0] != img_b.size[0] or img_a.size[1] != img_b.size[1]:
            raise ValueError("输入图像尺寸不一致，无法计算 PSNR。")

        pixels_a = list(img_a.pixels)
        pixels_b = list(img_b.pixels)

        if len(pixels_a) != len(pixels_b):
            raise ValueError("输入图像像素通道数量不一致，无法计算 PSNR。")

        mse = 0.0
        count = len(pixels_a)
        for value_a, value_b in zip(pixels_a, pixels_b):
            diff = value_a - value_b
            mse += diff * diff
        mse /= max(count, 1)

        if mse == 0:
            return float("inf")

        psnr = 20 * log10(1.0) - 10 * log10(mse)
        return psnr
    finally:
        bpy.data.images.remove(img_a)
        bpy.data.images.remove(img_b)


def main():
    """执行完整的场景构建、渲染配置与输出流程。"""
    # 1. 清空场景，避免残留对象影响渲染
    clear_scene()
    # 1.5 创建空物体，后续模型绑定到该父对象
    root_empty = add_root_empty()
    # 2. 导入两个 OBJ 模型（需保证相应 MTL 文件与纹理存在）
    imported_a = import_obj(OBJ_A_PATH)
    imported_b = import_obj(OBJ_B_PATH)
    parent_objects(root_empty, imported_a + imported_b)
    animate_root_rotation(root_empty)
    # 3. 布置摄像机与主光源，基础参数可按需在顶部常量区调整
    add_camera(CAMERA_LOCATION, CAMERA_TARGET)
    add_key_light(LIGHT_LOCATION, LIGHT_TARGET, LIGHT_ENERGY)
    # 4. 配置 Cycles 渲染器并指定输出文件
    output_dir = resolve_output_dir(OUTPUT_PATH)
    os.makedirs(output_dir, exist_ok=True)
    configure_render(output_dir)

    # 5. 针对每个关键帧分别渲染两张图并计算 PSNR
    obj_a_name = os.path.splitext(os.path.basename(OBJ_A_PATH))[0]
    obj_b_name = os.path.splitext(os.path.basename(OBJ_B_PATH))[0]
    scene = bpy.context.scene
    psnr_report_lines = []

    for frame in RENDER_FRAMES:
        scene.frame_set(frame)

        # 渲染仅包含 OBJ A 的图像
        set_objects_visibility(imported_a, False)
        set_objects_visibility(imported_b, True)
        output_a = os.path.join(output_dir, f"{obj_a_name}_{frame:03d}.png")
        scene.render.filepath = output_a
        bpy.ops.render.render(write_still=True)

        # 渲染仅包含 OBJ B 的图像
        set_objects_visibility(imported_a, True)
        set_objects_visibility(imported_b, False)
        output_b = os.path.join(output_dir, f"{obj_b_name}_{frame:03d}.png")
        scene.render.filepath = output_b
        bpy.ops.render.render(write_still=True)

        # 恢复两个对象的可见性
        set_objects_visibility(imported_a, False)
        set_objects_visibility(imported_b, False)

        psnr_value = compute_psnr(output_a, output_b)
        psnr_line = f"{frame:03d},{psnr_value:.4f}"
        psnr_report_lines.append(psnr_line)

    # 将 PSNR 结果写入输出目录内的文本文件
    report_path = os.path.join(output_dir, "psnr_results.txt")
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("frame,psnr_db\n")
        report_file.write("\n".join(psnr_report_lines))


if __name__ == "__main__":
    main()

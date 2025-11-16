import argparse
import os
import sys
from math import pi, log10, sqrt

import bpy
from mathutils import Vector

# 常量参数集中管理
IMAGE_WIDTH = 2048
IMAGE_HEIGHT = 2048
IMAGE_PERCENT = 100
LIGHT_ENERGY = 3.0
LIGHT_OFFSET_FACTORS = (0.5, 0.5, 0.5)
CAMERA_HEIGHT_SCALE = 0.6 * sqrt(2)
CAMERA_ORTHO_MARGIN = 1.05
GPU_DEVICE_PRIORITY = ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL", "OPENCL")


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


def set_objects_visibility(objects, hidden):
    """批量设置对象的渲染与视图可见性。"""
    for obj in objects:
        obj.hide_render = hidden
        obj.hide_viewport = hidden


def compute_scene_bounds(objects):
    """计算多个对象的联合包围盒中心与尺寸。"""
    min_corner = Vector((float("inf"), float("inf"), float("inf")))
    max_corner = Vector((float("-inf"), float("-inf"), float("-inf")))

    for obj in objects:
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            min_corner = Vector((min(min_corner.x, world_corner.x),
                                 min(min_corner.y, world_corner.y),
                                 min(min_corner.z, world_corner.z)))
            max_corner = Vector((max(max_corner.x, world_corner.x),
                                 max(max_corner.y, world_corner.y),
                                 max(max_corner.z, world_corner.z)))

    size = max_corner - min_corner
    size.x = max(size.x, 1e-3)
    size.y = max(size.y, 1e-3)
    size.z = max(size.z, 1e-3)
    center = (min_corner + max_corner) * 0.5
    return center, size


def configure_camera(center, size_vec):
    """根据包围盒尺寸自适应设置相机位置与正交比例。"""
    size_x = max(size_vec.x, 1e-3)
    size_y = max(size_vec.y, 1e-3)
    size_z = max(size_vec.z, 1e-3)
    max_dim = max(size_x, size_y, size_z)
    ortho_scale = max(size_x, size_y) * CAMERA_ORTHO_MARGIN * 1.2

    cam_x = center.x + max_dim
    cam_y = center.y + max_dim
    cam_z = center.z + CAMERA_HEIGHT_SCALE * max_dim

    bpy.ops.object.camera_add(location=(cam_x, cam_y, cam_z))
    camera = bpy.context.object
    camera.name = "RenderCamera"
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = ortho_scale
    direction = Vector((0.0, 0.0, 0.0)) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = camera


def configure_light(center, size):
    """创建太阳光并根据包围盒尺寸放置。"""
    light_x = center.x + LIGHT_OFFSET_FACTORS[0] * size.x
    light_y = center.y + LIGHT_OFFSET_FACTORS[1] * size.y
    light_z = center.z + LIGHT_OFFSET_FACTORS[2] * size.z

    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.light_add(type="SUN", align="WORLD", location=(light_x, light_y, light_z))
    light = bpy.context.object
    light.name = "KeyLight"
    light.data.energy = LIGHT_ENERGY
    direction = Vector((0.0, 0.0, 0.0)) - light.location
    light.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    light.data.angle = 0.5

def enable_cycles_gpu():
    """尝试启用 GPU 设备；若失败则回落到 CPU。"""
    prefs = bpy.context.preferences.addons["cycles"].preferences
    get_devices = getattr(prefs, "get_devices", None)
    refresh_devices = getattr(prefs, "refresh_devices", None)
    if callable(get_devices):
        get_devices()
    elif callable(refresh_devices):
        refresh_devices()

    requested = os.environ.get("BLENDER_CYCLES_DEVICE", "").upper()
    priority = []
    if requested:
        priority.append(requested)
    priority.extend(dtype for dtype in GPU_DEVICE_PRIORITY if dtype != requested)

    for dtype in priority:
        try:
            prefs.compute_device_type = dtype
        except Exception:
            continue

        gpu_found = False
        for device in prefs.devices:
            is_gpu = device.type != "CPU"
            device.use = is_gpu
            gpu_found = gpu_found or is_gpu

        if gpu_found:
            return dtype

    prefs.compute_device_type = "NONE"
    for device in prefs.devices:
        device.use = device.type == "CPU"
    return "CPU"


def configure_render(output_path: str):
    """切换到 Cycles 渲染器、启用 GPU 并设置输出路径。"""
    scene = bpy.context.scene
    # scene.render.engine = "CYCLES"
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    device_kind = enable_cycles_gpu()
    scene.cycles.device = 'GPU' if device_kind != "CPU" else 'CPU'
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.filepath = os.path.abspath(output_path)
    scene.render.resolution_x = IMAGE_WIDTH
    scene.render.resolution_y = IMAGE_HEIGHT
    scene.render.resolution_percentage = IMAGE_PERCENT

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


def render_single_view(scene, visible_group, hidden_group, filepath):
    """切换可见性并渲染单张图片。"""
    set_objects_visibility(visible_group, False)
    set_objects_visibility(hidden_group, True)
    scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)


def parse_args():
    """只解析 -- 之后传入脚本的 OBJ 参数，兼容直接 python 运行。"""
    if "--" in sys.argv:
        script_args = sys.argv[sys.argv.index("--") + 1:]
    else:
        script_args = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Render two OBJ files with a shared camera setup.")
    parser.add_argument("obj_a", help="Path to the first OBJ file.")
    parser.add_argument("obj_b", help="Path to the second OBJ file.")
    return parser.parse_args(script_args)


def main():
    args = parse_args()
    output_dir = os.path.dirname(os.path.abspath(args.obj_a)) or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    scene = bpy.context.scene
    clear_scene()
    imported_a = import_obj(args.obj_a)
    imported_b = import_obj(args.obj_b)

    center, size = compute_scene_bounds(imported_a + imported_b)
    configure_camera(center, size)
    configure_light(center, size)
    configure_render(output_dir)

    name_a = os.path.splitext(os.path.basename(args.obj_a))[0]
    name_b = os.path.splitext(os.path.basename(args.obj_b))[0]
    output_a = os.path.join(output_dir, f"render_{name_a}.png")
    output_b = os.path.join(output_dir, f"render_{name_b}.png")

    render_single_view(scene, imported_a, imported_b, output_a)
    render_single_view(scene, imported_b, imported_a, output_b)

    set_objects_visibility(imported_a, False)
    set_objects_visibility(imported_b, False)

    psnr_value = compute_psnr(output_a, output_b)
    folder_name = os.path.basename(os.path.normpath(os.path.dirname(args.obj_a))) or "root"
    print(f"{folder_name}_psnr : {psnr_value:.4f} dB")


if __name__ == "__main__":
    main()

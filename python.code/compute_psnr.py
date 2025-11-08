import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class Face:
    vt_indices: Tuple[int, int, int]
    material: Optional[str]


@dataclass
class OBJData:
    vt: np.ndarray  # (N, 2)
    faces: List[Face]
    textures: Dict[Optional[str], np.ndarray]  # material -> texture array (H, W, 3) float32


def parse_mtl(mtl_path: Path) -> Dict[str, Path]:
    material_to_texture: Dict[str, Path] = {}
    current: Optional[str] = None
    try:
        with mtl_path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if not parts:
                    continue
                tag = parts[0].lower()
                if tag == "newmtl" and len(parts) >= 2:
                    current = parts[1]
                elif tag == "map_kd" and current is not None:
                    tex_path = " ".join(parts[1:])  # allow spaces in path
                    material_to_texture[current] = (mtl_path.parent / tex_path).resolve()
    except FileNotFoundError:
        pass
    return material_to_texture


def load_texture(texture_path: Path) -> np.ndarray:
    img = Image.open(texture_path).convert("RGB")
    return np.asarray(img).astype(np.float32)


def parse_obj(obj_path: Path) -> OBJData:
    vt_list: List[Tuple[float, float]] = []
    faces: List[Face] = []
    material_to_texture_path: Dict[str, Path] = {}
    current_material: Optional[str] = None
    mtllibs: List[Path] = []

    with obj_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            tag = parts[0].lower()

            if tag == "vt" and len(parts) >= 3:
                u = float(parts[1])
                v = float(parts[2])
                vt_list.append((u, v))
            elif tag == "f" and len(parts) >= 4:
                indices = []
                for comp in parts[1:]:
                    items = comp.split("/")
                    if len(items) < 2 or items[1] == "":
                        raise ValueError(f"Face '{line}' 缺少纹理坐标 vt")
                    vt_idx = int(items[1]) - 1
                    indices.append(vt_idx)
                # triangulate if polygon
                for i in range(1, len(indices) - 1):
                    faces.append(Face(
                        vt_indices=(indices[0], indices[i], indices[i + 1]),
                        material=current_material,
                    ))
            elif tag == "usemtl" and len(parts) >= 2:
                current_material = parts[1]
            elif tag == "mtllib" and len(parts) >= 2:
                for name in parts[1:]:
                    mtllibs.append((obj_path.parent / name).resolve())

    # parse all referenced MTL files
    for mtl_path in mtllibs:
        material_to_texture_path.update(parse_mtl(mtl_path))

    textures: Dict[Optional[str], np.ndarray] = {}
    for material, tex_path in material_to_texture_path.items():
        if tex_path.exists():
            textures[material] = load_texture(tex_path)
        else:
            raise FileNotFoundError(f"纹理文件不存在: {tex_path}")

    # Some faces may use no material but we still allow them by setting None key with default
    if None not in textures:
        textures[None] = np.full((1, 1, 3), 0.0, dtype=np.float32)

    vt_array = np.asarray(vt_list, dtype=np.float64)
    return OBJData(vt=vt_array, faces=faces, textures=textures)


def cross2d(a: np.ndarray, b: np.ndarray) -> float:
    return a[0] * b[1] - a[1] * b[0]


def barycentric(point: np.ndarray, tri: np.ndarray) -> Optional[np.ndarray]:
    a, b, c = tri
    v0 = b - a
    v1 = c - a
    v2 = point - a
    denom = cross2d(v0, v1)
    if abs(denom) < 1e-12:
        return None
    inv_denom = 1.0 / denom
    w1 = cross2d(v2, v1) * inv_denom
    w2 = cross2d(v0, v2) * inv_denom
    w0 = 1.0 - w1 - w2
    return np.array([w0, w1, w2], dtype=np.float64)


def bilinear_sample(texture: np.ndarray, u: float, v: float) -> np.ndarray:
    h, w, _ = texture.shape
    if w == 1 and h == 1:
        return texture[0, 0]
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    x = u * (w - 1)
    y = (1.0 - v) * (h - 1)
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    dx = x - x0
    dy = y - y0

    top = (1.0 - dx) * texture[y0, x0] + dx * texture[y0, x1]
    bottom = (1.0 - dx) * texture[y1, x0] + dx * texture[y1, x1]
    return (1.0 - dy) * top + dy * bottom


def face_error(uv_ori: np.ndarray, uv_res: np.ndarray,
                tex_ori: np.ndarray, tex_res: np.ndarray) -> Tuple[float, int]:
    h_ori, w_ori, _ = tex_ori.shape
    if w_ori <= 1 or h_ori <= 1:
        return 0.0, 0

    u_vals = np.clip(uv_ori[:, 0], 0.0, 1.0)
    v_vals = np.clip(uv_ori[:, 1], 0.0, 1.0)

    x_min = int(max(0, math.floor(u_vals.min() * (w_ori - 1))))
    x_max = int(min(w_ori - 1, math.ceil(u_vals.max() * (w_ori - 1))))
    if x_max < x_min:
        return 0.0, 0

    y_min = int(max(0, math.floor((1.0 - v_vals.max()) * (h_ori - 1))))
    y_max = int(min(h_ori - 1, math.ceil((1.0 - v_vals.min()) * (h_ori - 1))))
    if y_max < y_min:
        return 0.0, 0

    # Generate all integer pixel coordinates in bounding box
    xs = np.linspace(x_min, x_max, x_max - x_min + 1)
    ys = np.linspace(y_min, y_max, y_max - y_min + 1)
    xv, yv = np.meshgrid(xs, ys)
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)

    u = xv / (w_ori - 1)
    v = 1.0 - yv / (h_ori - 1)
    points = np.stack((u, v), axis=1)  # (N, 2)

    a = uv_ori[0]
    v0 = uv_ori[1] - a
    v1 = uv_ori[2] - a
    denom = cross2d(v0, v1)
    if abs(denom) < 1e-12:
        return 0.0, 0

    v0x, v0y = v0
    v1x, v1y = v1
    v2 = points - a
    v2x = v2[:, 0]
    v2y = v2[:, 1]
    inv_denom = 1.0 / denom
    w1 = (v2x * v1y - v1x * v2y) * inv_denom
    w2 = (v0x * v2y - v2x * v0y) * inv_denom
    w0 = 1.0 - w1 - w2
    inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
    if not np.any(inside):
        return 0.0, 0

    u_inside = u[inside]
    v_inside = v[inside]
    w0 = w0[inside]
    w1 = w1[inside]
    w2 = w2[inside]

    uv_res_points = w0[:, None] * uv_res[0] + w1[:, None] * uv_res[1] + w2[:, None] * uv_res[2]

    colors_ori = bilinear_sample_batch(tex_ori, u_inside, v_inside)
    colors_res = bilinear_sample_batch(tex_res, uv_res_points[:, 0], uv_res_points[:, 1])

    diff = colors_ori - colors_res
    error = np.sum(diff * diff)
    count = colors_ori.shape[0]
    return float(error), int(count)


def bilinear_sample_batch(texture: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w, c = texture.shape
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    x = u * (w - 1)
    y = (1.0 - v) * (h - 1)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    dx = (x - x0)[:, None]
    dy = (y - y0)[:, None]

    top = (1.0 - dx) * texture[y0, x0] + dx * texture[y0, x1]
    bottom = (1.0 - dx) * texture[y1, x0] + dx * texture[y1, x1]
    return (1.0 - dy) * top + dy * bottom


def compute_texture_psnr(ori_obj_path: Path, res_obj_path: Path) -> Tuple[float, float]:
    ori_data = parse_obj(ori_obj_path)
    res_data = parse_obj(res_obj_path)

    if ori_data.vt.shape[0] == 0 or res_data.vt.shape[0] == 0:
        raise ValueError("OBJ 文件缺少 vt 纹理坐标")

    if len(ori_data.faces) != len(res_data.faces):
        raise ValueError("两个 OBJ 的面数量不一致，无法一一对应")

    face_count = len(ori_data.faces)
    print(f"面片总数: {face_count}")

    total_error = 0.0
    total_samples = 0
    progress_step = max(1, face_count // 100)

    for idx, (face_ori, face_res) in enumerate(zip(ori_data.faces, res_data.faces)):
        vt_idx_ori = face_ori.vt_indices
        vt_idx_res = face_res.vt_indices
        uv_ori = ori_data.vt[list(vt_idx_ori)]
        uv_res = res_data.vt[list(vt_idx_res)]

        tex_ori = ori_data.textures.get(face_ori.material)
        if tex_ori is None:
            tex_ori = ori_data.textures.get(None)
        tex_res = res_data.textures.get(face_res.material)
        if tex_res is None:
            tex_res = res_data.textures.get(None)
        if tex_ori is None or tex_res is None:
            continue

        err, samples = face_error(uv_ori, uv_res, tex_ori, tex_res)
        total_error += err
        total_samples += samples

        if (idx % progress_step == 0) or (idx == face_count - 1):
            percent = (idx + 1) / face_count
            bar_length = 30
            filled = int(bar_length * percent)
            bar = "#" * filled + "-" * (bar_length - filled)
            print(
                f"\r进度: |{bar}| {idx + 1}/{face_count} ({percent * 100:5.1f}%)",
                end="",
                flush=True,
            )

    print()

    if total_samples == 0:
        raise ValueError("未采样到任何像素，可能纹理坐标超出范围或纹理缺失")

    mse = total_error / (total_samples * 3.0)
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10.0 * math.log10((255.0 ** 2) / mse)
    return mse, psnr


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute texture PSNR between two OBJ models with matched topology.")
    parser.add_argument("ori_obj", type=Path, help="Original OBJ file path")
    parser.add_argument("res_obj", type=Path, help="Result OBJ file path")
    args = parser.parse_args()

    if not args.ori_obj.exists():
        parser.error(f"原始 OBJ 文件不存在: {args.ori_obj}")
    if not args.res_obj.exists():
        parser.error(f"结果 OBJ 文件不存在: {args.res_obj}")

    mse, psnr = compute_texture_psnr(args.ori_obj.resolve(), args.res_obj.resolve())
    if math.isinf(psnr):
        print("PSNR: ∞ (纹理完全一致)")
    else:
        print(f"MSE: {mse:.6f}")
        print(f"PSNR: {psnr:.6f} dB")


if __name__ == "__main__":
    main()

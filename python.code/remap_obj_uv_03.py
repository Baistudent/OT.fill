import math
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def parse_m_file(m_path: Path) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Parse .m file to extract XY and UV coordinates."""
    xy_coords: List[Tuple[float, float]] = []
    uv_coords: List[Tuple[float, float]] = []
    vertex_pattern = re.compile(
        r"Vertex\s+\d+\s+([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s+([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s+([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?).*?uv=\(([^\s]+)\s+([^\s\)]+)"
    )

    with m_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("Vertex"):
                continue
            match = vertex_pattern.search(line)
            if not match:
                continue
            x = float(match.group(1))
            y = float(match.group(2))
            u = float(match.group(4))
            v = float(match.group(5))
            xy_coords.append((x, y))
            uv_coords.append((u, v))

    if not xy_coords:
        raise ValueError("在 .m 文件中未找到顶点数据")

    vertex_count = len(xy_coords)
    grid_size = int(round(math.sqrt(vertex_count)))
    if grid_size * grid_size != vertex_count:
        raise ValueError(".m 文件的顶点数量不是完全平方数，无法构建规则网格")

    xy_array = np.array(xy_coords, dtype=np.float64)
    uv_array = np.array(uv_coords, dtype=np.float64)
    return xy_array, uv_array, grid_size, grid_size


def barycentric_coords(point: np.ndarray, tri_xy: np.ndarray) -> np.ndarray:
    """Compute barycentric coordinates of point with respect to triangle tri_xy."""
    a, b, c = tri_xy
    v0 = b - a
    v1 = c - a
    v2 = point - a
    denom = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(denom) < 1e-15:
        return None
    inv_denom = 1.0 / denom
    w1 = (v2[0] * v1[1] - v1[0] * v2[1]) * inv_denom
    w2 = (v0[0] * v2[1] - v2[0] * v0[1]) * inv_denom
    w0 = 1.0 - w1 - w2
    return np.array([w0, w1, w2], dtype=np.float64)


class GridMapper:
    def __init__(self, xy: np.ndarray, uv: np.ndarray, width: int, height: int):
        self.xy = xy
        self.uv = uv
        self.width = width
        self.height = height
        self.x_min = float(np.min(xy[:, 0]))
        self.x_max = float(np.max(xy[:, 0]))
        self.y_min = float(np.min(xy[:, 1]))
        self.y_max = float(np.max(xy[:, 1]))
        self.x_span = self.x_max - self.x_min if self.x_max > self.x_min else 1.0
        self.y_span = self.y_max - self.y_min if self.y_max > self.y_min else 1.0
        self.cols = width - 1
        self.rows = height - 1

    def _triangle_indices(self, row: int, col: int) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        v00 = row * self.width + col
        v10 = row * self.width + (col + 1)
        v01 = (row + 1) * self.width + col
        v11 = (row + 1) * self.width + (col + 1)
        tri1 = (v00, v10, v11)
        tri2 = (v00, v11, v01)
        return tri1, tri2

    def map_xy_to_uv(self, x: float, y: float) -> Tuple[float, float]:
        fx = ((x - self.x_min) / self.x_span) * self.cols
        fy = ((y - self.y_min) / self.y_span) * self.rows
        fx = float(np.clip(fx, 0.0, self.cols - 1e-9))
        fy = float(np.clip(fy, 0.0, self.rows - 1e-9))
        col = int(math.floor(fx))
        row = int(math.floor(fy))
        col = min(col, self.cols - 1)
        row = min(row, self.rows - 1)
        tri_candidates = self._triangle_indices(row, col)
        point = np.array([x, y], dtype=np.float64)
        best_uv = None
        best_min_weight = -1e9
        for tri in tri_candidates:
            tri_xy = self.xy[np.array(tri)]
            weights = barycentric_coords(point, tri_xy)
            if weights is None:
                continue
            min_w = weights.min()
            if min_w < -1e-6:
                if min_w > best_min_weight:
                    best_min_weight = min_w
                    best_uv = self.uv[np.array(tri)].T @ weights
                continue
            tri_uv = self.uv[np.array(tri)]
            uv_point = tri_uv.T @ weights
            return float(uv_point[0]), float(uv_point[1])
        if best_uv is not None:
            return float(best_uv[0]), float(best_uv[1])
        # Fallback: clamp to nearest vertex
        row = min(max(row, 0), self.rows - 1)
        col = min(max(col, 0), self.cols - 1)
        tri = tri_candidates[0]
        uv_avg = self.uv[np.array(tri[0])]
        return float(uv_avg[0]), float(uv_avg[1])


def _split_code_and_comment(line: str) -> Tuple[str, str]:
    """Split a line into code portion and trailing comment, if any."""
    if "#" not in line:
        return line.rstrip("\n"), ""
    code_part, comment = line.split("#", 1)
    return code_part.rstrip("\n"), "#" + comment.rstrip("\n")


def _rewrite_mtl(
    original_mtl: Path,
    new_mtl: Path,
    texture_override: Optional[str],
) -> None:
    """Copy the original MTL while updating map_Kd to the provided texture."""
    texture_replacement: Optional[str] = None
    if texture_override is not None:
        try:
            texture_path = Path(texture_override)
            if not texture_path.is_absolute():
                texture_path = (new_mtl.parent / texture_path).resolve()
            texture_replacement = os.path.relpath(texture_path, new_mtl.parent)
        except Exception:
            texture_replacement = Path(texture_override).name

    try:
        contents = original_mtl.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        contents = original_mtl.read_text(encoding="latin-1")

    lines = contents.splitlines()
    rewritten: List[str] = []
    replaced_texture = False

    for line in lines:
        code_part, comment = _split_code_and_comment(line)
        leading_ws_len = len(code_part) - len(code_part.lstrip(" \t"))
        leading_ws = code_part[:leading_ws_len]
        tokens = code_part.strip().split()

        if (
            tokens
            and tokens[0].lower() == "map_kd"
            and texture_replacement is not None
        ):
            new_tokens = [tokens[0], texture_replacement]
            if len(tokens) > 2:
                new_tokens.extend(tokens[2:])
            new_line = leading_ws + " ".join(new_tokens)
            rewritten.append((new_line + (" " + comment if comment else "")).rstrip() + "\n")
            replaced_texture = True
            continue

        rewritten.append(line.rstrip("\n") + "\n")

    if texture_replacement is not None and not replaced_texture:
        rewritten.append(f"map_Kd {texture_replacement}\n")

    new_mtl.write_text("".join(rewritten), encoding="utf-8")


def remap_obj_uv(
    obj_path: Path,
    m_path: Path,
    output_path: Path,
    texture_override: Optional[str] = None,
) -> None:
    print(f"读取网格: {m_path}...")
    xy_coords, uv_coords, width, height = parse_m_file(m_path)
    mapper = GridMapper(xy_coords, uv_coords, width, height)

    print(f"读取 OBJ: {obj_path}...")
    with obj_path.open("r", encoding="utf-8") as f:
        obj_lines = f.readlines()

    remapped_lines: List[str] = []
    original_mtl_token: Optional[str] = None
    new_mtl_name = output_path.stem + ".mtl"
    total_vt = sum(1 for line in obj_lines if line.lstrip().startswith("vt"))
    processed_vt = 0
    if total_vt:
        print(f"vt 总数: {total_vt}")
    for line in obj_lines:
        stripped = line.strip()
        if stripped.lower().startswith("mtllib"):
            code_part, comment = _split_code_and_comment(line)
            leading_ws_len = len(code_part) - len(code_part.lstrip(" \t"))
            leading_ws = code_part[:leading_ws_len]
            tokens = code_part.strip().split()
            if len(tokens) >= 2 and original_mtl_token is None:
                original_mtl_token = tokens[1]
            new_line = leading_ws + f"mtllib ./{new_mtl_name}"
            if len(tokens) > 2:
                new_line += " " + " ".join(tokens[2:])
            if comment:
                new_line += " " + comment
            new_line += "\n"
            remapped_lines.append(new_line)
            continue

        if not stripped.startswith("vt"):
            remapped_lines.append(line)
            continue
        code_part, comment = _split_code_and_comment(line)
        leading_ws_len = len(code_part) - len(code_part.lstrip(" \t"))
        leading_ws = code_part[:leading_ws_len]
        tokens = code_part.strip().split()
        if len(tokens) < 3:
            remapped_lines.append(line)
            continue
        try:
            x = float(tokens[1])
            y = float(tokens[2])
        except ValueError:
            remapped_lines.append(line)
            continue
        new_u, new_v = mapper.map_xy_to_uv(x, y)
        new_tokens = ["vt", f"{new_u:.6f}", f"{new_v:.6f}"]
        if len(tokens) > 3:
            new_tokens.extend(tokens[3:])
        new_line = leading_ws + " ".join(new_tokens)
        if comment:
            new_line += " " + comment
        new_line += "\n"
        remapped_lines.append(new_line)
        processed_vt += 1
        if total_vt:
            progress_step = max(1, total_vt // 10)
            if processed_vt % progress_step == 0 or processed_vt == total_vt:
                print(f"  已处理：{processed_vt}/{total_vt}")

    with output_path.open("w", encoding="utf-8") as f:
        f.writelines(remapped_lines)
    print(f"已写出新的 OBJ: {output_path}")

    original_mtl_path = (obj_path.parent / Path(original_mtl_token)).resolve()
    new_mtl_path = output_path.with_suffix(".mtl")
    _rewrite_mtl(original_mtl_path, new_mtl_path, texture_override)
    print(f"输出 {output_path}...")



def main() -> None:
    if len(sys.argv) < 5:
        print(
            "用法: python remap_obj_uv.py <obj_file> <mesh_m_file> <output_obj> <texture_name>"
        )
        sys.exit(1)
    obj_path = Path(sys.argv[1])
    m_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])
    texture_name = sys.argv[4]
    remap_obj_uv(obj_path, m_path, output_path, texture_name)


if __name__ == "__main__":
    main()

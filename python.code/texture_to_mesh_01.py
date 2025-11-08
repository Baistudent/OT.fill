"""
从纹理图像和 OBJ 文件生成三角网格
1. 读取 OBJ 文件，获取三角形面的 UV 坐标
2. 使用扫描线法对每个三角形进行光栅化
3. 构建像素网格
4. 基于规则像素网格生成三角形网格
5. 输出 OBJ 文件（坐标在 0-1 范围内）
"""

from PIL import Image
import numpy as np


def parse_obj_faces(obj_path):
    """
    解析 OBJ 文件，获取所有三角形面的 UV 坐标
    
    返回：
        faces_uv: 列表，每个元素是 [(u0,v0), (u1,v1), (u2,v2)]
    """
    textures = []
    faces_uv = []
    
    with open(obj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if not parts:
                continue
            
            if parts[0] == 'vt':
                u = float(parts[1])
                v = float(parts[2]) if len(parts) > 2 else 0.0
                textures.append((u, v))
            elif parts[0] == 'f':
                face_uv = []
                for i in range(1, len(parts)):
                    indices = parts[i].split('/')
                    if len(indices) > 1 and indices[1]:
                        vt_idx = int(indices[1]) - 1
                        if 0 <= vt_idx < len(textures):
                            face_uv.append(textures[vt_idx])
                
                # 只记录三角形面（3 个顶点）
                if len(face_uv) == 3:
                    faces_uv.append(face_uv)
    
    return faces_uv


def rasterize_triangle(triangle_pixels, a, b, c, width, height):
    """
    使用扫描线法光栅化三角形，填充数组
    
    参数：
        triangle_pixels: H×W 的 bool 数组
        a, b, c: 三角形三个顶点的像素坐标 (x, y)
        width, height: 数组尺寸
    """
    # 确保 y 坐标排序
    points = sorted([a, b, c], key=lambda p: p[1])
    p1, p2, p3 = points[0], points[1], points[2]
    
    y1, y2, y3 = int(np.ceil(p1[1])), int(np.floor(p2[1])), int(np.floor(p3[1]))
    
    # 扫描线填充
    def line_x_at_y(p_start, p_end, y):
        """计算直线与扫描线 y 的交点 x 坐标"""
        y_s, y_e = p_start[1], p_end[1]
        if abs(y_e - y_s) < 1e-6:
            return None
        t = (y - y_s) / (y_e - y_s)
        if t < 0 or t > 1:  # 确保交点在线段范围内
            return None
        return p_start[0] + t * (p_end[0] - p_start[0])
    
    # 从 p1 到 p3 的扫描线
    for y in range(max(0, y1), min(height, y3 + 1)):
        y_float = float(y)  # 扫描线在像素中心
        
        # 找到与三条边的交点
        intersections = []
        
        # 边 p1-p2
        if p1[1] <= y_float <= p2[1]:
            x = line_x_at_y(p1, p2, y_float)
            if x is not None:
                intersections.append(x)
        
        # 边 p2-p3
        if p2[1] <= y_float <= p3[1]:
            x = line_x_at_y(p2, p3, y_float)
            if x is not None:
                intersections.append(x)
        
        # 边 p1-p3
        if p1[1] <= y_float <= p3[1]:
            x = line_x_at_y(p1, p3, y_float)
            if x is not None:
                intersections.append(x)
        
        # 排序交点
        if len(intersections) >= 2:
            intersections.sort()
            # 向外大幅扩张：ceil 再减1，floor 再加1
            x_min = max(0, int(np.ceil(intersections[0])) - 1)
            x_max = min(width - 1, int(np.floor(intersections[-1])) + 1)
            
            # 填充像素
            if x_min <= x_max:
                for x in range(x_min, x_max + 1):
                    triangle_pixels[y, x] = True


def load_textured_pixels(obj_path, texture_path, pixel=None):
    """
    读取 OBJ 和纹理图像，使用扫描线法光栅化
    
    参数：
        obj_path: OBJ 文件路径
        texture_path: 纹理图像路径
        pixel: 采样分辨率（如果 None，使用原始图片分辨率）
               比如原图 4096×4096，指定 pixel=256，则视为 256×256 来采样
    
    返回：
    textured_pixels_low: (height, width) 的 bool 数组（用于生成规则网格）
    width, height: 规则网格的采样分辨率
    textured_pixels_high: (orig_height, orig_width) 的 bool 数组（原始纹理分辨率）
    orig_width, orig_height: 原始纹理分辨率
    """
    print(f"解析文件：{obj_path}")
    faces_uv = parse_obj_faces(obj_path)
    print(f"  三角面个数：{len(faces_uv)}")
    
    if len(faces_uv) == 0:
        print("错误：OBJ 文件中没有三角形面或没有 UV 坐标！")
        return None, 0, 0
    
    print(f"纹理图像：{texture_path}")
    img = Image.open(texture_path)
    original_width, original_height = img.size
    
    # 确定采样分辨率
    if pixel is not None:
        width, height = pixel, pixel
        print(f"  图像尺寸：{original_width} x {original_height}")
        print(f"  采样：{width} x {height}")
    else:
        width, height = original_width, original_height
        print(f"  图像尺寸：{width} x {height}")

    if width < 1 or height < 1:
        raise ValueError("采样分辨率必须大于 0")

    textured_pixels_low = np.zeros((height, width), dtype=bool)
    textured_pixels_high = np.zeros((original_height, original_width), dtype=bool)
    
    # 使用扫描线法光栅化每个三角形
    print("光栅化三角形...")
    for face_idx, face_uv in enumerate(faces_uv):
        # if (face_idx + 1) % max(1, len(faces_uv) // 10) == 0:
        #     print(f"  进度：{face_idx + 1}/{len(faces_uv)}")

        (u0, v0), (u1, v1), (u2, v2) = face_uv

        coords_low = (
            (u0 * (width - 1), v0 * (height - 1)),
            (u1 * (width - 1), v1 * (height - 1)),
            (u2 * (width - 1), v2 * (height - 1)),
        )
        coords_high = (
            (u0 * (original_width - 1), v0 * (original_height - 1)),
            (u1 * (original_width - 1), v1 * (original_height - 1)),
            (u2 * (original_width - 1), v2 * (original_height - 1)),
        )

        rasterize_triangle(
            textured_pixels_low,
            coords_low[0],
            coords_low[1],
            coords_low[2],
            width,
            height,
        )
        rasterize_triangle(
            textured_pixels_high,
            coords_high[0],
            coords_high[1],
            coords_high[2],
            original_width,
            original_height,
        )
    
    covered_count = np.sum(textured_pixels_low)

    return textured_pixels_low, width, height, textured_pixels_high, original_width, original_height


def compute_triangle_pixel_stats(
    mask_high: np.ndarray,
    coarse_width: int,
    coarse_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Count covered/all pixels per triangle using original texture resolution."""
    if coarse_width < 2 or coarse_height < 2:
        empty_shape = (max(coarse_height - 1, 0), max(coarse_width - 1, 0), 2)
        return np.zeros(empty_shape, dtype=np.int64), np.zeros(empty_shape, dtype=np.int64)

    orig_height, orig_width = mask_high.shape
    if orig_height == 0 or orig_width == 0:
        empty_shape = (coarse_height - 1, coarse_width - 1, 2)
        return np.zeros(empty_shape, dtype=np.int64), np.zeros(empty_shape, dtype=np.int64)

    triangle_hits = np.zeros((coarse_height - 1, coarse_width - 1, 2), dtype=np.int64)
    triangle_totals = np.zeros_like(triangle_hits)

    x_positions = (np.arange(orig_width, dtype=np.float64) + 0.5) / orig_width
    y_positions = (np.arange(orig_height, dtype=np.float64) + 0.5) / orig_height

    x_scaled = x_positions * (coarse_width - 1)
    y_scaled = y_positions * (coarse_height - 1)

    col_indices = np.clip(np.floor(x_scaled).astype(int), 0, coarse_width - 2)
    row_indices = np.clip(np.floor(y_scaled).astype(int), 0, coarse_height - 2)

    local_x = x_scaled - col_indices
    local_y = y_scaled - row_indices

    for y in range(orig_height):
        row = row_indices[y]
        ly = local_y[y]
        tri_mask = ly > local_x

        cols_tri0 = col_indices[~tri_mask]
        cols_tri1 = col_indices[tri_mask]

        if cols_tri0.size:
            np.add.at(triangle_totals, (row, cols_tri0, 0), 1)
            hits0 = mask_high[y, ~tri_mask].astype(np.int64)
            if hits0.size:
                np.add.at(triangle_hits, (row, cols_tri0, 0), hits0)

        if cols_tri1.size:
            np.add.at(triangle_totals, (row, cols_tri1, 1), 1)
            hits1 = mask_high[y, tri_mask].astype(np.int64)
            if hits1.size:
                np.add.at(triangle_hits, (row, cols_tri1, 1), hits1)

    return triangle_hits, triangle_totals


def build_mesh_from_pixels(
    triangle_hits: np.ndarray,
    triangle_totals: np.ndarray,
    width: int,
    height: int,
):
    """
    从像素统计构建完整的规则三角网格，并计算顶点 area 及 interior 标记。

    参数：
        triangle_hits: (height-1, width-1, 2) 的整数数组，每个三角形中带纹理的像素数量
        triangle_totals: 同形状的整数数组，每个三角形覆盖的总像素数量
        width, height: 规则网格的采样分辨率

    返回：
        vertices: 顶点列表 [(u, v, z, interior, area), ...]
        faces: 面列表 [(v1, v2, v3), ...]
    """
    vertex_hits = np.zeros((height, width), dtype=np.float64)
    vertex_totals = np.zeros((height, width), dtype=np.float64)
    vertex_interior = np.zeros((height, width), dtype=bool)

    for row in range(height - 1):
        for col in range(width - 1):
            tri0_hits = triangle_hits[row, col, 0]
            tri0_total = triangle_totals[row, col, 0]
            if tri0_total > 0:
                coords0 = ((row, col), (row, col + 1), (row + 1, col + 1))
                for r, c in coords0:
                    vertex_totals[r, c] += tri0_total
                    vertex_hits[r, c] += tri0_hits
                if tri0_hits > 0:
                    for r, c in coords0:
                        vertex_interior[r, c] = True

            tri1_hits = triangle_hits[row, col, 1]
            tri1_total = triangle_totals[row, col, 1]
            if tri1_total > 0:
                coords1 = ((row, col), (row + 1, col + 1), (row + 1, col))
                for r, c in coords1:
                    vertex_totals[r, c] += tri1_total
                    vertex_hits[r, c] += tri1_hits
                if tri1_hits > 0:
                    for r, c in coords1:
                        vertex_interior[r, c] = True

    vertex_area = np.zeros((height, width), dtype=np.float64)
    valid_mask = vertex_totals > 0
    vertex_area[valid_mask] = vertex_hits[valid_mask] / vertex_totals[valid_mask]

    vertices = []
    faces = []

    for i in range(height):
        for j in range(width):
            u = j / (width - 1) if width > 1 else 0.0
            v = i / (height - 1) if height > 1 else 0.0

            interior = bool(vertex_interior[i, j])
            area_value = float(vertex_area[i, j])

            vertices.append((u, v, 0.0, interior, area_value))
    
    # 第二步：为每个矩形单位创建两个三角形
    # 对每个矩形 (i, j)，创建左下和右上两个三角形
    for i in range(height - 1):
        for j in range(width - 1):
            v0 = i * width + j              # 左上
            v1 = i * width + j + 1          # 右上
            v2 = (i + 1) * width + j + 1    # 右下
            v3 = (i + 1) * width + j        # 左下
            
            faces.append((v0 + 1, v1 + 1, v2 + 1))
            faces.append((v0 + 1, v2 + 1, v3 + 1))
    
    return vertices, faces


def save_m(filename, vertices, faces, width, height):
    """
    保存为 .m 格式文件
    
    参数：
        filename: 输出文件路径
        vertices: 顶点列表 [(u, v, z, interior), ...]
        faces: 面列表 [(v1, v2, v3), ...]
        width, height: 网格尺寸
    """
    with open(filename, 'w') as f:
        f.write("# Vertices: {}\n".format(len(vertices)))
        f.write("# Faces: {}\n\n".format(len(faces)))
        
        # 写入顶点
        for v_id, (u, v, z, interior, area_value) in enumerate(vertices, 1):
            area_attr = f"area=({area_value:.6f})"
            f.write(
                "Vertex {} {:.6f} {:.6f} {:.6f} {{uv=({:.6f} {:.6f}) {}}}\n".format(
                    v_id,
                    u - 0.5,
                    v - 0.5,
                    z,
                    u - 0.5,
                    v - 0.5,
                    area_attr,
                )
            )
        
        # 写入面
        f.write("\n")
        for face_id, (v1, v2, v3) in enumerate(faces, 1):
            f.write("Face {} {} {} {}\n".format(face_id, v1, v2, v3))


def texture_to_mesh(obj_path, texture_path, output_obj, pixel=None):
    """
    主函数：从纹理图像和 OBJ 文件生成三角网格
    
    参数：
        obj_path: OBJ 文件路径
        texture_path: 纹理文件路径 (PNG/BMP)
        output_obj: 输出 OBJ 文件路径
        pixel: 采样分辨率（可选）
    """
    (
        textured_pixels_low,
        width,
        height,
        textured_pixels_high,
        orig_width,
        orig_height,
    ) = load_textured_pixels(obj_path, texture_path, pixel)
    
    if textured_pixels_low is None:
        return
    
    textured_count = np.sum(textured_pixels_low)
    if textured_count == 0:
        print("警告：纹理图像中没有被 OBJ 面覆盖的像素！")
    
    triangle_hits, triangle_totals = compute_triangle_pixel_stats(
        textured_pixels_high,
        width,
        height,
    )

    print(f"构建规则三角网格...")
    vertices, faces = build_mesh_from_pixels(triangle_hits, triangle_totals, width, height)
    
    print(f"  顶点数：{len(vertices)}, 面数：{len(faces)}")
    
    if len(faces) == 0:
        print("警告：生成的面数为 0")
        return
    
    print(f"保存到：{output_obj}")
    save_m(output_obj, vertices, faces, width, height)
    print(f"完成！")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("使用方法: python texture_to_mesh.py <obj_file> <texture_path> [output_obj] [pixel]")
        print("示例:")
        print("  # 使用原始图片分辨率")
        print("  python texture_to_mesh.py Elephant/model.obj Elephant/texture.png Elephant/texture_mesh.obj")
        print("  # 指定采样分辨率为 256×256")
        print("  python texture_to_mesh.py Elephant/model.obj Elephant/texture.png Elephant/texture_mesh.obj 256")
        sys.exit(1)
    
    obj_path = sys.argv[1]
    texture_path = sys.argv[2]
    output_obj = sys.argv[3] if len(sys.argv) > 3 else obj_path.replace('.obj', '_texture_mesh.obj')
    pixel = int(sys.argv[4]) if len(sys.argv) > 4 else None
    
    texture_to_mesh(obj_path, texture_path, output_obj, pixel)

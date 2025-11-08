"""
通过纹理映射结果，将图片转成映射后的图片
1. 读取 .m 文件，获取顶点映射关系 (x,y) <- (u,v)
2. 预计算三角形常数以优化重心坐标计算
3. 对每个像素，找到对应三角形，用重心插值采样原图
4. 输出映射后的图片
"""

from PIL import Image
import numpy as np


def parse_m_file(m_path):
    """
    解析 .m 文件，提取顶点和面的信息
    
    参数：
        m_path: .m 文件路径
    
    返回：
        vertices: 列表 [(x, y, u, v), ...]，索引从0开始
        faces: 列表 [(v1, v2, v3), ...]，顶点索引从1开始需要转换为0开始
        width, height: 网格尺寸
    """
    vertices = []
    faces = []
    
    with open(m_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if not parts:
                continue
            
            if parts[0] == 'Vertex':
                # 格式: Vertex <id> <x> <y> <z> {uv=(<u> <v>) ...}
                v_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                # z = float(parts[4])  # 不需要z坐标
                
                # 解析 uv 坐标（在大括号中）
                # 将所有剩余部分连接成一个字符串
                remaining = ' '.join(parts[5:])
                
                # 查找 uv=( 和 ) 之间的内容
                uv_start = remaining.find('uv=(')
                if uv_start != -1:
                    uv_start += 4
                    uv_end = remaining.find(')', uv_start)
                    if uv_end != -1:
                        uv_part = remaining[uv_start:uv_end].strip()
                        # 分割 UV 坐标（中间可能有多个空格）
                        uv_values = uv_part.split()
                        if len(uv_values) >= 2:
                            u = float(uv_values[0])
                            v = float(uv_values[1])
                            vertices.append((x, y, u, v))
                            continue
                
                # 如果解析失败，打印警告但继续
                print(f"警告：无法解析第 {len(vertices)+1} 个顶点的 UV 坐标：{remaining}")
            
            elif parts[0] == 'Face':
                # 格式: Face <id> <v1> <v2> <v3>
                v1 = int(parts[2]) - 1  # 转换为0开始的索引
                v2 = int(parts[3]) - 1
                v3 = int(parts[4]) - 1
                faces.append((v1, v2, v3))
    
    # 推断网格尺寸（假设规则的 pixel×pixel 网格）
    # 顶点数 = width * height
    num_vertices = len(vertices)
    width = height = int(np.sqrt(num_vertices))
    
    print(f"  顶点数：{len(vertices)}")
    print(f"  面数：{len(faces)}")
    
    return vertices, faces, width, height


def precompute_triangle_constants_texture_space(vertices, faces):
    """
    预计算三角形的常数（在纹理坐标空间中）
    这里三角形顶点使用纹理坐标 (u, v) 而不是映射坐标
    
    参数：
        vertices: 顶点列表 [(x, y, u, v), ...]
        faces: 面列表 [(v1, v2, v3), ...]
    
    返回：
        tri_data: 列表，每个元素为预计算的三角形数据
    """
    tri_data = []
    
    for face_idx, (v0_idx, v1_idx, v2_idx) in enumerate(faces):
        v0 = vertices[v0_idx]
        v1 = vertices[v1_idx]
        v2 = vertices[v2_idx]
        
        # 纹理坐标（源空间）
        u0, v0_uv = v0[2], v0[3]
        u1, v1_uv = v1[2], v1[3]
        u2, v2_uv = v2[2], v2[3]
        
        # 映射后的坐标（输出空间）
        x0, y0 = v0[0], v0[1]
        x1, y1 = v1[0], v1[1]
        x2, y2 = v2[0], v2[1]
        
        # 计算三角形面积（在纹理坐标空间中）
        area = 0.5 * abs((u1 - u0) * (v2_uv - v0_uv) - (u2 - u0) * (v1_uv - v0_uv))
        
        if area < 1e-10:
            continue
        
        inv_area = 1.0 / area
        
        tri_data.append({
            'texture_coords': ((u0, v0_uv), (u1, v1_uv), (u2, v2_uv)),
            'mapped_coords': ((x0, y0), (x1, y1), (x2, y2)),
            'area': area,
            'inv_area': inv_area,
        })
    
    print(f"预计算完成，{len(tri_data)} 个有效三角形（纹理空间）")
    return tri_data


def barycentric_in_texture_space(uv_point, tri_texture_coords, tri_inv_area):
    """
    计算点在纹理坐标空间中相对于三角形的重心坐标
    
    参数：
        uv_point: 纹理坐标 (u, v)
        tri_texture_coords: 三角形三个顶点的纹理坐标 ((u0,v0), (u1,v1), (u2,v2))
        tri_inv_area: 面积倒数
    
    返回：
        (λ0, λ1, λ2) 或 None
    """
    u0, v0 = tri_texture_coords[0]
    u1, v1 = tri_texture_coords[1]
    u2, v2 = tri_texture_coords[2]
    u, v = uv_point
    
    # 计算三个子三角形的面积（使用叉积）
    area0 = 0.5 * ((u1 - u) * (v2 - v) - (u2 - u) * (v1 - v))
    area1 = 0.5 * ((u2 - u) * (v0 - v) - (u0 - u) * (v2 - v))
    area2 = 0.5 * ((u0 - u) * (v1 - v) - (u1 - u) * (v0 - v))
    
    # 重心坐标
    lam0 = area0 * tri_inv_area
    lam1 = area1 * tri_inv_area
    lam2 = area2 * tri_inv_area
    
    # 检查是否在三角形内
    eps = -1e-6
    if lam0 >= eps and lam1 >= eps and lam2 >= eps:
        return (lam0, lam1, lam2)
    
    return None


def find_triangle_for_texture_coord(uv_point, tri_data):
    """
    为纹理坐标找到其所在的三角形
    
    参数：
        uv_point: 纹理坐标 (u, v)
        tri_data: 预计算的三角形数据列表
    
    返回：
        bary_coords: 重心坐标 (λ0, λ1, λ2) 或 None
        tri_idx: 三角形索引，或 None
    """
    for tri_idx, tri in enumerate(tri_data):
        bary = barycentric_in_texture_space(uv_point, tri['texture_coords'], tri['inv_area'])
        if bary is not None:
            return bary, tri_idx
    
    return None, None


def precompute_affine_transform(uv_tri, xy_tri):
    """
    计算直接的UV→XY仿射变换矩阵
    
    参数：
        uv_tri: ((u0,v0), (u1,v1), (u2,v2))
        xy_tri: ((x0,y0), (x1,y1), (x2,y2))
    
    返回：
        T: 3×3 齐次仿射变换矩阵
           [x]   [a b c] [u]
           [y] = [d e f] [v]
           [1]   [0 0 1] [1]
    """
    u0, v0 = uv_tri[0]
    u1, v1 = uv_tri[1]
    u2, v2 = uv_tri[2]
    
    x0, y0 = xy_tri[0]
    x1, y1 = xy_tri[1]
    x2, y2 = xy_tri[2]
    
    # 构造 UV 矩阵（齐次坐标）
    U = np.array([
        [u0, v0, 1],
        [u1, v1, 1],
        [u2, v2, 1]
    ], dtype=np.float64)
    
    # 检查矩阵是否可逆
    det = np.linalg.det(U)
    if abs(det) < 1e-10:
        return None
    
    # 构造 XY 矩阵（每行一个点）
    xy_hom = np.array([
        [x0, y0],
        [x1, y1],
        [x2, y2]
    ], dtype=np.float64)
    
    # 求逆矩阵
    U_inv = np.linalg.inv(U)  # (3, 3)
    
    # 计算系数矩阵
    # coeffs = U_inv @ xy_hom  (3,3) @ (3,2) = (3,2)
    # coeffs[:, 0] = [a, b, c]  (x的系数)
    # coeffs[:, 1] = [d, e, f]  (y的系数)
    coeffs = U_inv @ xy_hom  # (3, 2)
    
    a_b_c = coeffs[:, 0]  # (3,)
    d_e_f = coeffs[:, 1]  # (3,)
    
    # 构造完整的 3×3 齐次变换矩阵
    T = np.array([
        [a_b_c[0], a_b_c[1], a_b_c[2]],  # [a, b, c]
        [d_e_f[0], d_e_f[1], d_e_f[2]],  # [d, e, f]
        [0.0,      0.0,      1.0]         # [0, 0, 1]
    ], dtype=np.float64)
    
    return T


def point_in_triangle(u, v, uv_tri):
    """
    判断点(u,v)是否在三角形内（使用重心坐标检查）
    
    参数：
        u, v: 点坐标
        uv_tri: 三角形三顶点
    
    返回：
        True/False
    """
    u0, v0 = uv_tri[0]
    u1, v1 = uv_tri[1]
    u2, v2 = uv_tri[2]
    
    # 计算三个子三角形的面积
    area0 = 0.5 * ((u1 - u) * (v2 - v) - (u2 - u) * (v1 - v))
    area1 = 0.5 * ((u2 - u) * (v0 - v) - (u0 - u) * (v2 - v))
    area2 = 0.5 * ((u0 - u) * (v1 - v) - (u1 - u) * (v0 - v))
    
    # 检查是否在三角形内
    eps = -1e-6
    return (area0 >= eps) and (area1 >= eps) and (area2 >= eps)


def apply_affine_transform(u, v, T):
    """
    使用3×3齐次仿射变换矩阵计算(x, y)
    
    参数：
        u, v: 输入坐标
        T: 3×3 齐次仿射变换矩阵
    
    返回：
        (x, y)
    """
    uv_homo = np.array([u, v, 1], dtype=np.float64)
    xy_homo = T @ uv_homo  # (3, 3) @ (3,) = (3,)
    return xy_homo[0], xy_homo[1]


def build_uniform_grid(tri_uv, grid_res):
    """
    为三角形建立均匀网格加速结构
    
    参数：
        tri_uv: (N, 3, 2) 的数组，UV 三角形
        grid_res: 网格划分尺寸（grid_res × grid_res）
    
    返回：
        grid_cells: 长度为 grid_res^2 的列表，每个元素是包含该单元的三角形索引列表
    """
    num_tri = tri_uv.shape[0]
    grid_cells = [[] for _ in range(grid_res * grid_res)]
    eps = 1e-9
    for idx in range(num_tri):
        uv = tri_uv[idx]
        u_min = max(0.0, float(np.min(uv[:, 0]) - eps))
        u_max = min(1.0, float(np.max(uv[:, 0]) + eps))
        v_min = max(0.0, float(np.min(uv[:, 1]) - eps))
        v_max = min(1.0, float(np.max(uv[:, 1]) + eps))
        u_start = int(np.clip(np.floor(u_min * grid_res), 0, grid_res - 1))
        u_end = int(np.clip(np.ceil(u_max * grid_res) - 1, 0, grid_res - 1))
        v_start = int(np.clip(np.floor(v_min * grid_res), 0, grid_res - 1))
        v_end = int(np.clip(np.ceil(v_max * grid_res) - 1, 0, grid_res - 1))
        for cu in range(u_start, u_end + 1):
            for cv in range(v_start, v_end + 1):
                grid_cells[cv * grid_res + cu].append(idx)
    return grid_cells


def bilinear_sample(image, x_norm, y_norm):
    """
    对源图进行双线性采样
    
    参数：
        image: (H, W, C) 数组
        x_norm, y_norm: 归一化坐标，范围 [0, 1]
    
    返回：
        colors: (N, C) 采样结果
    """
    h, w = image.shape[:2]
    x = np.clip(x_norm * (w - 1), 0.0, w - 1.0)
    y = np.clip((1.0 - y_norm) * (h - 1), 0.0, h - 1.0)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    wx = (x - x0)[:, None]
    wy = (y - y0)[:, None]

    top = (1.0 - wx) * image[y0, x0] + wx * image[y0, x1]
    bottom = (1.0 - wx) * image[y1, x0] + wx * image[y1, x1]
    return np.clip((1.0 - wy) * top + wy * bottom, 0, 255).astype(np.uint8)


def get_edge_intersections(v_line, uv_tri):
    """
    计算扫描线v_line与三角形三条边的交点u值
    
    参数：
        v_line: 扫描线的v坐标（0-1之间）
        uv_tri: 三角形三顶点 ((u0,v0), (u1,v1), (u2,v2))
    
    返回：
        交点u值的列表
    """
    u0, v0 = uv_tri[0]
    u1, v1 = uv_tri[1]
    u2, v2 = uv_tri[2]
    
    intersections = []
    
    # 检查三条边
    edges = [(0, 1), (1, 2), (2, 0)]
    edge_points = [uv_tri[i] for i in range(3)]
    
    for start_idx, end_idx in edges:
        u_start, v_start = edge_points[start_idx]
        u_end, v_end = edge_points[end_idx]
        
        # 检查扫描线是否与边相交
        if (v_start <= v_line <= v_end) or (v_end <= v_line <= v_start):
            if abs(v_end - v_start) > 1e-10:
                # 线性插值
                t = (v_line - v_start) / (v_end - v_start)
                u_intersect = u_start + t * (u_end - u_start)
                intersections.append(u_intersect)
    
    return sorted(intersections)


def apply_texture_mapping_scanline(m_file, source_image_path, output_image_path):
    """
    主函数：基于均匀网格的逆向映射（UV -> XY -> 采样）
    
    参数：
        m_file: .m 文件路径
        source_image_path: 源图片路径
        output_image_path: 输出图片路径
    """
    print(f"读取网格文件：{m_file}...")
    vertices, faces, grid_width, grid_height = parse_m_file(m_file)
    
    print(f"读取源图片：{source_image_path}...")
    source_img = Image.open(source_image_path)
    src_width, src_height = source_img.size
    src_pixels = np.array(source_img.convert('RGB'), dtype=np.uint8)
    
    # 目标 UV 图片尺寸：默认使用源图分辨率
    out_width = src_width
    out_height = src_height
    
    # 计算 XY 坐标范围（用于归一化到 [0, 1]）
    x_coords = np.array([v[0] for v in vertices])
    y_coords = np.array([v[1] for v in vertices])
    x_min, x_max = float(np.min(x_coords)), float(np.max(x_coords))
    y_min, y_max = float(np.min(y_coords)), float(np.max(y_coords))
    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0
    
    # 收集三角形数据
    tri_uv_list = []
    tri_T_list = []
    tri_Ainv_list = []
    for face_idx, (v0_idx, v1_idx, v2_idx) in enumerate(faces):
        v0, v1, v2 = vertices[v0_idx], vertices[v1_idx], vertices[v2_idx]
        uv_tri = np.array([
            [v0[2], v0[3]],
            [v1[2], v1[3]],
            [v2[2], v2[3]],
        ], dtype=np.float64)
        xy_tri = np.array([
            [v0[0], v0[1]],
            [v1[0], v1[1]],
            [v2[0], v2[1]],
        ], dtype=np.float64)
        T = precompute_affine_transform(uv_tri, xy_tri)
        if T is None:
            continue
        # 构造 barycentric 求解矩阵
        U = np.array([
            [uv_tri[0, 0], uv_tri[1, 0], uv_tri[2, 0]],
            [uv_tri[0, 1], uv_tri[1, 1], uv_tri[2, 1]],
            [1.0, 1.0, 1.0]
        ], dtype=np.float64)
        try:
            A_inv = np.linalg.inv(U)
        except np.linalg.LinAlgError:
            continue
        tri_uv_list.append(uv_tri)
        tri_T_list.append(T)
        tri_Ainv_list.append(A_inv)
    if not tri_uv_list:
        print("错误：没有有效三角形可用于映射")
        return
    
    triangles_uv = np.stack(tri_uv_list)              # (N, 3, 2)
    triangles_T = np.stack(tri_T_list)                # (N, 3, 3)
    triangles_Ainv = np.stack(tri_Ainv_list)          # (N, 3, 3)
    num_triangles = triangles_uv.shape[0]
    
    # 构建均匀网格加速结构
    print("构建均匀网格...")
    grid_res = int(np.clip(int(np.sqrt(num_triangles)), 32, 512))
    grid_cells = build_uniform_grid(triangles_uv, grid_res)
    
    # 预计算输出 UV 像素的中心坐标
    u_centers = (np.arange(out_width) + 0.5) / out_width
    v_centers = 1.0 - (np.arange(out_height) + 0.5) / out_height  # 翻转 v 轴
    u_map = np.tile(u_centers, (out_height, 1))
    v_map = np.tile(v_centers[:, None], (1, out_width))
    u_flat = u_map.ravel()
    v_flat = v_map.ravel()
    
    # 每个像素所属的网格单元
    cell_u_flat = np.minimum(grid_res - 1, (u_flat * grid_res).astype(np.int32))
    cell_v_flat = np.minimum(grid_res - 1, (v_flat * grid_res).astype(np.int32))
    cell_ids = cell_v_flat * grid_res + cell_u_flat
    counts = np.bincount(cell_ids, minlength=grid_res * grid_res)
    order = np.argsort(cell_ids)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    
    output_pixels = np.full((out_height, out_width, 3), 255, dtype=np.uint8)
    mapped_mask = np.zeros((out_height, out_width), dtype=bool)
    mapped_count = 0
    eps = 1e-8
    
    print("开始纹理映射...")
    for cell_idx in range(grid_res * grid_res):
        pixel_count = counts[cell_idx]
        if pixel_count == 0:
            continue
        tri_indices = grid_cells[cell_idx]
        if not tri_indices:
            continue
        start = offsets[cell_idx]
        end = offsets[cell_idx + 1]
        pixel_flat_indices = order[start:end]
        # UV 点集合 (3, p)
        points = np.vstack((
            u_flat[pixel_flat_indices],
            v_flat[pixel_flat_indices],
            np.ones(pixel_count, dtype=np.float64)
        ))
        assigned = np.full(pixel_count, -1, dtype=np.int32)
        lambda_vals = np.zeros((3, pixel_count), dtype=np.float64)
        for tri_idx in tri_indices:
            lambdas = triangles_Ainv[tri_idx] @ points
            inside = (lambdas >= -eps).all(axis=0)
            new_assign = inside & (assigned == -1)
            if not np.any(new_assign):
                continue
            lambda_vals[:, new_assign] = lambdas[:, new_assign]
            assigned[new_assign] = tri_idx
        valid_mask = assigned >= 0
        if not np.any(valid_mask):
            continue
        selected_pixels = pixel_flat_indices[valid_mask]
        tri_idx_sel = assigned[valid_mask]
        uv_points_sel = np.stack((
            u_flat[selected_pixels],
            v_flat[selected_pixels],
            np.ones(selected_pixels.shape[0], dtype=np.float64)
        ), axis=1)  # (p, 3)
        T_sel = triangles_T[tri_idx_sel]
        xy_homo = np.einsum('nij,nj->ni', T_sel, uv_points_sel)
        x_vals = xy_homo[:, 0]
        y_vals = xy_homo[:, 1]
        x_norm = np.clip((x_vals - x_min) / x_range, 0.0, 1.0)
        y_norm = np.clip((y_vals - y_min) / y_range, 0.0, 1.0)
        colors = bilinear_sample(src_pixels, x_norm, y_norm)
        i_indices = selected_pixels // out_width
        j_indices = selected_pixels % out_width
        output_pixels[i_indices, j_indices] = colors
        mapped_mask[i_indices, j_indices] = True
        mapped_count += selected_pixels.shape[0]
    
    unique_mapped = int(mapped_mask.sum())
    total_pixels = out_width * out_height
    unmapped_pixels = total_pixels - unique_mapped
    print(f"映射了 {mapped_count} 个点（未覆盖 {unmapped_pixels}）")
    
    output_img = Image.fromarray(output_pixels)
    print(f"保存输出图片：{output_image_path}...")
    output_img.save(output_image_path)
    print("完成！")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("使用方法: python texture_mapping.py <m_file> <source_image> [output_image]")
        print("示例:")
        print("  python texture_mapping.py Elephant/texture_mesh_256.m Elephant/texture.png Elephant/output.png")
        sys.exit(1)
    
    m_file = sys.argv[1]
    source_image = sys.argv[2]
    output_image = sys.argv[3] if len(sys.argv) > 3 else m_file.replace('.m', '_mapped.png')
    
    apply_texture_mapping_scanline(m_file, source_image, output_image)

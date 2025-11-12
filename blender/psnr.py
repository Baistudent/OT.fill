"""使用 OpenCV 计算两张图片的 PSNR。"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def compute_psnr(image_path_a: str, image_path_b: str) -> float:
	"""读取两张图片并返回它们的 PSNR（峰值信噪比）。"""

	img_a = cv2.imread(image_path_a, cv2.IMREAD_UNCHANGED)
	img_b = cv2.imread(image_path_b, cv2.IMREAD_UNCHANGED)

	if img_a is None:
		raise FileNotFoundError(f"无法读取图像: {image_path_a}")
	if img_b is None:
		raise FileNotFoundError(f"无法读取图像: {image_path_b}")

	if img_a.shape != img_b.shape:
		raise ValueError("两张图片的尺寸或通道数不一致，无法计算 PSNR。")

	# OpenCV 内置函数会根据像素类型自动处理峰值
	return float(cv2.PSNR(img_a, img_b))


def main() -> None:
	parser = argparse.ArgumentParser(description="使用 OpenCV 计算两张图片的 PSNR")
	parser.add_argument("image_a", type=Path, help="第一张图片路径")
	parser.add_argument("image_b", type=Path, help="第二张图片路径")
	args = parser.parse_args()

	psnr_value = compute_psnr(str(args.image_a), str(args.image_b))
	print(f"两张图片的 PSNR 为: {psnr_value:.4f} dB")


if __name__ == "__main__":
	main()

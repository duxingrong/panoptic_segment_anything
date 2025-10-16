import os
import json
import random
from PIL import Image, ImageDraw
from tqdm import tqdm  # 引入tqdm库用于显示进度条 (pip install tqdm)
import sys
# 从你封装好的模块中导入核心类
from _panoptic_segment import PanopticSegmentAnything

# ------------------- 1. 全局配置 -------------------

# <<<< 请在这里修改你的路径 >>>>
# 输入：包含所有原始图片的文件夹路径
INPUT_DIR = "raw_frame" 
# 输出：保存分割后图片的文件夹路径
OUTPUT_DIR = "output_frame"
# <<<< 修改结束 >>>>

# 定义适用于 Replica Office 数据集的JSON配置
# 这里的 name 和 id 是关键，它们将决定物体的颜色
# has_instances: true 代表"物体(thing)", false 代表"背景(stuff)"
TASK_JSON = """
{
  "categories": [
    { "id": 1, "name": "wall", "has_instances": false },
    { "id": 2, "name": "floor", "has_instances": false },
    { "id": 10, "name": "table", "has_instances": true },
    { "id": 13, "name": "chair", "has_instances": true },
    { "id": 14, "name": "sofa", "has_instances": true },
    { "id": 15, "name": "door", "has_instances": true }
  ]
}
"""

# ------------------- 2. 颜色映射与渲染函数 -------------------

def create_color_map(categories_json):
    """
    根据JSON配置，为每个类别名称创建一个固定颜色的映射表。
    使用ID作为随机种子，确保颜色是确定性的。
    """
    print("正在创建固定的颜色映射表...")
    categories = json.loads(categories_json)["categories"]
    color_map = {}
    for category in categories:
        # 使用 category['id'] 作为随机种子，确保每次运行颜色都一样
        random.seed(category['id'])
        color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        color_map[category['name']] = color
    
    print("颜色映射表创建完成:")
    for name, color in color_map.items():
        print(f"- {name}: {color}")
    return color_map

def render_with_fixed_colors(result_data, color_map, alpha=0.6):
    """
    一个修改版的渲染函数，使用固定的颜色映射表进行渲染。
    """
    image_array, subsection_label_pairs = result_data
    base_image = Image.fromarray(image_array).convert("RGBA")
    overlay_image = Image.new("RGBA", base_image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay_image)

    for mask, label in subsection_label_pairs:
        # 从我们预先生成的color_map中查找颜色
        color = color_map.get(label)
        if color: # 只有在映射表中找到颜色时才绘制
            mask_image = Image.new("L", (mask.shape[1], mask.shape[0]), 0)
            mask_image.putdata(mask.flatten() * 255)
            draw.bitmap((0, 0), mask_image, fill=color)

    final_image = Image.alpha_composite(base_image, Image.blend(base_image, overlay_image, alpha))
    return final_image.convert("RGB")

# ------------------- 3. 主执行逻辑 -------------------

if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 获取所有图片文件列表
    try:
        image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print(f"错误: 在目录 '{INPUT_DIR}' 中没有找到任何图片文件。请检查路径。")
            sys.exit(1)
    except FileNotFoundError:
        print(f"错误: 输入目录 '{INPUT_DIR}' 不存在。请修改脚本中的路径。")
        sys.exit(1)

    # 1. 实例化全景分割模型类
    print("开始实例化模型...")
    # 强制使用CPU，如果你的GPU显存不足以处理大量图片，这是一个安全的选择。
    # 如果你有强大的GPU，可以改为 "cuda" 或 None
    psa = PanopticSegmentAnything(device="cpu") 

    # 2. 创建固定的颜色映射表
    color_map = create_color_map(TASK_JSON)
    
    print(f"\n找到 {len(image_files)} 张图片，开始批处理...")

    # 3. 遍历所有图片并进行处理
    for filename in tqdm(image_files, desc="处理进度"):
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)

        try:
            input_image = Image.open(input_path)

            # 调用核心方法进行分割 (使用JSON配置)
            annotated_data, _, _ = psa.generate_panoptic_mask(
                image=input_image,
                task_attributes_json=TASK_JSON
            )

            # 使用固定颜色进行渲染
            final_image = render_with_fixed_colors(annotated_data, color_map)

            # 保存渲染后的图片
            final_image.save(output_path)

        except Exception as e:
            print(f"\n处理图片 {filename} 时发生错误: {e}")
            # 你可以选择跳过或者停止
            continue
            
    print(f"\n批处理完成！所有分割后的图片已保存到: {OUTPUT_DIR}")
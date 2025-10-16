#! /usr/local/env python3
#-*- coding:utf-8 -*- 

"""
全景分割模块(封装版),以便主程序调用
"""

import sys
import os

import json
import random
import warnings

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy import ndimage
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import hf_hub_download

# 确保 GroundingDINO 在 Python 路径中
if "./GroundingDINO" not in sys.path:
    sys.path.insert(0, "./GroundingDINO")

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.util.inference import predict, annotate

# segment anything
from segment_anything import build_sam, SamPredictor

# CLIPSeg
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

try:
    from segments.utils import bitmap2file
except ImportError:
    print("Warning: `segments` library not found. Bitmap generation will be unavailable.")
    print("Please install with: pip install segments-ai")
    bitmap2file = None


class PanopticSegmentAnything:
    """零样本全景分割图像模型"""

    def __init__(self, device=None):
        """
        初始化模型.
        :param device: 指定设备, 如 'cuda', 'cpu'. 如果为 None, 则自动检测.
        """
        if not os.path.exists("./sam_vit_h_4b8939.pth"):
            print("未找到 SAM 模型文件 'sam_vit_h_4b8939.pth'。")
            print("请从以下网址下载: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
            sys.exit(1) # 关键模型不存在则退出
        else:
            print("成功加载模型文件 sam_vit_h_4b8939.pth")

        ## 全局变量
        self.config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.ckpt_repo_id = "ShilongLiu/GroundingDINO"
        self.ckpt_filename = "groundingdino_swint_ogc.pth"
        self.sam_checkpoint = "./sam_vit_h_4b8939.pth"

        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {self.device}")

        if self.device != "cpu":
            try:
                from GroundingDINO.groundingdino import _C
            except ImportError:
                warnings.warn(
                    "Failed to load custom C++ ops. Running on CPU mode Only in groundingdino!"
                )
        self._init_models()

    def _init_models(self):
        """初始化所有需要的模型"""
        print("正在初始化所有模型")
        # 初始化 GroundingDINO 模型
        self.dino_model = self._load_model_hf(self.config_file, self.ckpt_repo_id, self.ckpt_filename, self.device)

        # 初始化 SAM
        self.sam = build_sam(checkpoint=self.sam_checkpoint)
        self.sam.to(device=self.device)
        self.sam_predictor = SamPredictor(self.sam)

        # 初始化 CLIPSeg
        self.clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.clipseg_model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        )
        self.clipseg_model.to(self.device)
        print("所有模型初始化完成.")

    def _load_model_hf(self, model_config_path, repo_id, filename, device):
        """加载和准备Grounding DINO 模型"""
        args = SLConfig.fromfile(model_config_path)
        model = build_model(args)
        args.device = device

        cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
        checkpoint = torch.load(cache_file, map_location=device)
        log = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(f"GroundingDINO model loaded from {cache_file} \n => {log}")
        model.eval()
        model = model.to(device)
        return model 

    ##########################  核心函数  ################################
    def _load_image_for_dino(self, image):
        """将图片变成DINO输入的格式"""
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333), # 调整尺寸,短边800像素,长边不超过1333像素
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # 根据一组固定的均值和标准差来调整图片每个颜色通道的数值
            ]
        )
        dino_image, _ = transform(image, None)
        return dino_image

    def _dino_detection(self, image, image_array, category_names, category_name_to_id, box_threshold, text_threshold,visualize=False):
        """使用GroudingDINO 零样本根据输入生成boxes"""
        detection_prompt = " . ".join(category_names)
        dino_image = self._load_image_for_dino(image)
        dino_image = dino_image.to(self.device)
        with torch.no_grad():
            boxes,logits,phrases = predict( # 边界框 置信度 在每个框里找到的物体的名字
                model=self.dino_model,
                image=dino_image,
                caption = detection_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=self.device,
            )
        # --- 开始修改 ---
        # 过滤掉无法识别的短语，确保所有列表同步
        valid_indices = []
        for i, phrase in enumerate(phrases):
            if phrase in category_name_to_id:
                valid_indices.append(i)

        boxes = boxes[valid_indices]
        logits = logits[valid_indices]
        # 使用列表推导式创建新的、只包含有效短语的列表
        phrases = [phrases[i] for i in valid_indices]

        # 现在可以安全地创建 category_ids 列表了
        category_ids = [category_name_to_id[phrase] for phrase in phrases]
        # --- 结束修改 ---
        if visualize:
            annotated_frame = annotate(
                image_source=image_array,boxes=boxes,logits=logits,phrases=phrases
            )
            annotated_frame = annotated_frame[...,::-1] # BGR->RGB
            visualization = Image.fromarray(annotated_frame)
            return boxes,category_ids,visualization
        else:
            return boxes,category_ids,phrases


    def _sam_masks_from_dino_boxes(self, image_array, boxes):
        """利用SAM将boxes精确为mask"""
        H, W, _ = image_array.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            boxes_xyxy, image_array.shape[:2]
        ).to(self.device)
        thing_masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        return thing_masks

    def _preds_to_semantic_inds(self, preds, threshold):
        """为图片中的每一个像素做决定，判断到底属于哪个类别"""
        flat_preds = preds.reshape((preds.shape[0], -1))
        flat_preds_with_threshold = torch.full(
            (preds.shape[0] + 1, flat_preds.shape[-1]), threshold, device=self.device
        )
        flat_preds_with_threshold[1:preds.shape[0] + 1, :] = flat_preds
        semantic_inds = torch.topk(flat_preds_with_threshold, 1, dim=0).indices.reshape(
            (preds.shape[-2], preds.shape[-1])
        )
        return semantic_inds

    def _clipseg_segmentation(self, image, category_names, background_threshold):
        """根据指定的背景名称，利用CLIPSeg为每个背景生成一张概率图"""
        inputs = self.clipseg_processor(
            text=category_names,
            images=[image] * len(category_names),
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.clipseg_model(**inputs)
        logits = outputs.logits
        if len(logits.shape) == 2:
            logits = logits.unsqueeze(0)
        upscaled_logits = nn.functional.interpolate(
            logits.unsqueeze(1),
            size=(image.size[1], image.size[0]),
            mode="bilinear",
        )
        preds = torch.sigmoid(upscaled_logits.squeeze(dim=1))
        semantic_inds = self._preds_to_semantic_inds(preds, background_threshold)
        return preds, semantic_inds

    def _semantic_inds_to_shrunken_bool_masks(self, semantic_inds, shrink_kernel_size, num_categories):
        """对模具进行腐蚀，为每个类别制作一张黑白分明的专属图片"""
        shrink_kernel = np.ones((shrink_kernel_size, shrink_kernel_size))
        bool_masks = torch.zeros((num_categories, *semantic_inds.shape), dtype=torch.bool, device=self.device)
        for category in range(num_categories):
            binary_mask = (semantic_inds == category).cpu()
            shrunken_binary_mask_array = (
                ndimage.binary_erosion(binary_mask.numpy(), structure=shrink_kernel)
                if shrink_kernel_size > 0
                else binary_mask.numpy()
            )
            bool_masks[category] = torch.from_numpy(shrunken_binary_mask_array).to(self.device)
        return bool_masks

    def _clip_and_shrink_preds(self, semantic_inds, preds, shrink_kernel_size, num_categories):
        """根据模具，将原始概率图中所有位于核心区域以外的模糊的概率值全部清零"""
        bool_masks = self._semantic_inds_to_shrunken_bool_masks(
            semantic_inds, shrink_kernel_size, num_categories
        ).to(preds.device)
        sizes = [torch.sum(bool_masks[i].int()).item() for i in range(1, bool_masks.size(0))]
        max_size = max(sizes) if sizes else 0
        relative_sizes = [size / max_size for size in sizes] if max_size > 0 else [0] * len(sizes)
        clipped_preds = torch.zeros_like(preds)
        for i in range(1, bool_masks.size(0)):
            float_mask = bool_masks[i].float()
            clipped_preds[i - 1] = preds[i - 1] * float_mask
        return clipped_preds, relative_sizes

    def _sample_points_based_on_preds(self, preds, N):
        """在一张概率图上撒点，点会在概率大的地方集中"""
        if N == 0:
            return []
        height, width = preds.shape
        weights = preds.ravel()
        indices = np.arange(height * width)
        # 确保权重总和不为0
        if weights.sum() == 0:
            return []
        sampled_indices = random.choices(indices, weights=weights, k=N)
        sampled_points = [(index % width, index // width) for index in sampled_indices]
        return sampled_points

    def _upsample_pred(self, pred, image_source):
        """SAM在草稿纸上画出的轮廓，完美放大为图像的大小"""
        pred = pred.unsqueeze(dim=0)
        original_height = image_source.shape[0]
        original_width = image_source.shape[1]
        
        # 使用 SAM 的内部尺寸 (1024) 作为目标尺寸
        target_size = self.sam_predictor.model.image_encoder.img_size
        
        upsampled_tensor = F.interpolate(
            pred, size=(target_size, target_size), mode="bilinear", align_corners=False
        )

        # 裁剪到原始图片的缩放尺寸
        input_size = self.sam_predictor.input_size
        h, w = input_size
        
        upsampled_tensor = upsampled_tensor[..., :h, :w]

        # 再次上采样到最终的原始图像尺寸
        return F.interpolate(upsampled_tensor, size=(original_height, original_width), mode="bilinear", align_corners=False).squeeze(dim=1)

    def _sam_mask_from_points(self, image_array, points):
        """SAM根据点来生成masks"""
        points_array = np.array(points)
        point_labels = np.ones(len(points))
        _, _, logits = self.sam_predictor.predict(
            point_coords=points_array,
            point_labels=point_labels,
        )
        total_pred = torch.max(torch.sigmoid(torch.tensor(logits, device=self.device)), dim=0)[0].unsqueeze(dim=0)
        upsampled_pred = self._upsample_pred(total_pred, image_array)
        return upsampled_pred

    def _inds_to_segments_format(self, panoptic_inds, thing_category_ids, stuff_category_names, category_name_to_id):
        """将这张充满数字ID的内部地图，转换成两种标准的、通用的、可交付的最终成果格式"""
        if not bitmap2file:
            print("Warning: `inds_to_segments_format` requires `segments-ai` package.")
            return None, []

        panoptic_inds_array = panoptic_inds.cpu().numpy().astype(np.uint32)
        bitmap_file = bitmap2file(panoptic_inds_array, is_segmentation_bitmap=True)
        segmentation_bitmap = Image.open(bitmap_file)
        
        stuff_category_ids = [category_name_to_id[name] for name in stuff_category_names if name in category_name_to_id]

        unique_inds = np.unique(panoptic_inds_array)
        stuff_annotations = [
            {"id": int(i), "category_id": stuff_category_ids[i - 1]}
            for i in range(1, len(stuff_category_names) + 1)
            if i in unique_inds and (i - 1) < len(stuff_category_ids)
        ]
        thing_annotations = [
            {"id": int(len(stuff_category_names) + 1 + i), "category_id": thing_category_id}
            for i, thing_category_id in enumerate(thing_category_ids)
        ]
        annotations = stuff_annotations + thing_annotations
        return segmentation_bitmap, annotations

    def generate_panoptic_mask(self, image, thing_category_names_string="", stuff_category_names_string="",
                               dino_box_threshold=0.3, dino_text_threshold=0.25,
                               segmentation_background_threshold=0.1, shrink_kernel_size=20,
                               num_samples_factor=1000, task_attributes_json=""):
        """
        生成全景分割图的主函数。
        可以通过简单的字符串输入或复杂的JSON配置来调用。
        """
        # 判断是常规输入还是高级指令
        if task_attributes_json:
            task_attributes = json.loads(task_attributes_json)
            categories = task_attributes["categories"]
            category_name_to_id = {cat["name"]: cat["id"] for cat in categories}
            stuff_categories = [cat for cat in categories if not cat.get("has_instances", False)]
            thing_categories = [cat for cat in categories if cat.get("has_instances", False)]
            stuff_category_names = [cat["name"] for cat in stuff_categories]
            thing_category_names = [cat["name"] for cat in thing_categories]
        else:
            thing_category_names = [name.strip() for name in thing_category_names_string.split(",") if name.strip()]
            stuff_category_names = [name.strip() for name in stuff_category_names_string.split(",") if name.strip()]
            category_names = thing_category_names + stuff_category_names
            category_name_to_id = {name: i for i, name in enumerate(category_names)}
        
        # 预处理输入图像
        image = image.convert("RGB")
        image_array = np.array(image)

        # 预加载 SAM
        self.sam_predictor.set_image(image_array)

        # 1. 使用Grounding DINO 检测"thing" 的 boxes
        thing_category_ids, thing_masks, detected_thing_category_names = [], [], []
        if thing_category_names:
            thing_boxes, thing_category_ids, detected_thing_category_names = self._dino_detection(
                image, image_array, thing_category_names, category_name_to_id,
                dino_box_threshold, dino_text_threshold
            )
            if thing_boxes.nelement() > 0:
                # 2. 利用SAM根据boxes 得到 thing 的 segmentation masks
                thing_masks = self._sam_masks_from_dino_boxes(image_array, thing_boxes)

        # 处理 "stuff"
        sam_semantic_inds = None
        if stuff_category_names:
            # 3. 利用CLIPSeg 模型粗略得到 "stuff"的 masks
            clipseg_preds, clipseg_semantic_inds = self._clipseg_segmentation(
                image, stuff_category_names, segmentation_background_threshold
            )

            # 4. 从stuff 的masks中,移除物体的区域
            clipseg_semantic_inds_without_things = clipseg_semantic_inds.clone()
            if len(thing_boxes)>0:
                combined_things_mask = torch.any(thing_masks,dim=0)
                clipseg_semantic_inds_without_things[combined_things_mask[0]]=0

            # 5. 清理和收缩概率图
            clipsed_clipped_preds, relative_sizes = self._clip_and_shrink_preds(
                clipseg_semantic_inds_without_things, clipseg_preds,
                shrink_kernel_size, len(stuff_category_names) + 1
            )

            # 6. 使用SAM 为 stuff 获取更加精细的分割掩码
            sam_preds = torch.zeros_like(clipsed_clipped_preds)
            for i in range(clipsed_clipped_preds.shape[0]):
                clipseg_pred = clipsed_clipped_preds[i]
                num_samples = int(relative_sizes[i] * num_samples_factor)
                points = self._sample_points_based_on_preds(clipseg_pred.cpu().numpy(), num_samples)
                if not points: continue
                pred = self._sam_mask_from_points(image_array, points)
                sam_preds[i] = pred
            
            sam_semantic_inds = self._preds_to_semantic_inds(sam_preds, segmentation_background_threshold)

        # 7. 将“物体”索引和“背景”索引合并为全景索引
        panoptic_inds = (
            sam_semantic_inds.clone() if sam_semantic_inds is not None
            else torch.zeros(image_array.shape[0], image_array.shape[1], dtype=torch.long, device=self.device)
        )
        ind = len(stuff_category_names) + 1
        for thing_mask in thing_masks:
            panoptic_inds[thing_mask.squeeze(dim=0)] = ind
            ind += 1

        panoptic_bool_masks = self._semantic_inds_to_shrunken_bool_masks(panoptic_inds, 0, ind).cpu().numpy().astype(bool)
        
        # 确保 panoptic_names 的长度与 panoptic_bool_masks 的第一个维度相匹配
        all_names = ["unlabeled"] + stuff_category_names + detected_thing_category_names
        panoptic_names = all_names[:panoptic_bool_masks.shape[0]]

        subsection_label_pairs = [
            (panoptic_bool_masks[i], panoptic_name)
            for i, panoptic_name in enumerate(panoptic_names) if panoptic_bool_masks[i].any()
        ]

        segmentation_bitmap, annotations = self._inds_to_segments_format(
            panoptic_inds, thing_category_ids, stuff_category_names, category_name_to_id
        )
        annotations_json = json.dumps(annotations)

        return (image_array, subsection_label_pairs), segmentation_bitmap, annotations_json


def render_panoptic_segmentation(result_data, alpha=1):
    """
    一个辅助函数，用于将 generate_panoptic_mask 返回的结果渲染成可视化的图片。
    :param result_data: generate_panoptic_mask 返回的第一个元素 (image_array, subsection_label_pairs)
    :param alpha: 遮罩的透明度
    :return: PIL.Image 对象，包含了渲染好的分割图
    """
    image_array, subsection_label_pairs = result_data
    
    # 将原始图像数组转换为PIL Image对象
    base_image = Image.fromarray(image_array).convert("RGBA")
    
    # 创建一个透明的图层用于绘制遮罩
    overlay_image = Image.new("RGBA", base_image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay_image)

    # 为每个类别生成一个随机颜色
    colors = {}
    for _, label in subsection_label_pairs:
        if label not in colors and label != "unlabeled":
            colors[label] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    print("\n渲染的类别和颜色:")
    for label, color in colors.items():
        print(f"- {label}: {color}")

    # 绘制每个类别的遮罩
    for mask, label in subsection_label_pairs:
        if label in colors:
            color = colors[label]
            # 将布尔遮罩转换为可绘制的格式
            mask_image = Image.new("L", (mask.shape[1], mask.shape[0]), 0)
            mask_image.putdata(mask.flatten() * 255)
            
            # 在透明图层上用指定颜色绘制这个遮罩
            draw.bitmap((0, 0), mask_image, fill=color)

    # 将带有遮罩的图层与原始图片混合
    final_image = Image.alpha_composite(base_image, Image.blend(base_image, overlay_image, alpha))

    return final_image.convert("RGB")


if __name__ == "__main__":
    # ------------------- 使用示例 -------------------
    
    # 1. 实例化全景分割模型类
    print("开始实例化模型...")
    psa = PanopticSegmentAnything(device="cpu")

    # 2. 准备输入数据
    # 定义一个JSON字符串，用于高级调用。
    # 这模拟了从API或配置文件接收到的任务。
    task_json = """
    {
      "categories": [
        { "id": 1, "name": "floor", "has_instances": false },
        { "id": 2, "name": "wall", "has_instances": false },
        { "id": 3, "name": "window", "has_instances": false },
        { "id": 4, "name": "ceiling", "has_instances": false },
        { "id": 5, "name": "door", "has_instances": true },
        { "id": 6, "name": "sofa", "has_instances": true },
        { "id": 7, "name": "table", "has_instances": true },
        { "id": 8, "name": "pillow", "has_instances": true },
        { "id": 9, "name": "curtain", "has_instances": true }
      ]
    }
    """

    # 打开一张本地图片 (请确保你的目录下有这张图片, 或替换成你自己的图片)
    image_path = "example.jpg" 
    if not os.path.exists(image_path):
        print(f"错误: 示例图片 '{image_path}' 不存在，请下载或替换为有效路径。")
    else:
        print(f"\n正在处理图片: {image_path}")
        input_image = Image.open(image_path)

        # 3. 调用核心方法进行分割
        # 我们在这里使用json作为输入，所以thing和stuff的字符串参数留空
        annotated_data, bitmap, annotations = psa.generate_panoptic_mask(
            image=input_image,
            task_attributes_json=task_json
        )

        print("\n分割完成!")
        print("返回的标注信息 (JSON):")
        print(annotations)

        # 4. 渲染结果并展示
        print("\n正在渲染分割图...")
        final_image = render_panoptic_segmentation(annotated_data)
        
        # 展示图片
        final_image.show()

        # 保存图片
        output_path = "example_segmented.png"
        final_image.save(output_path)
        print(f"渲染后的分割图已保存到: {output_path}")

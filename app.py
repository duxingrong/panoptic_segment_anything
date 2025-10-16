#! /usr/local/env python3
# -*- coding:utf-8 -*- 
"""
一种零样本训练的全景分割模型-在线识别
"""

import sys
import os 

# 首先 执行 pip install -v -e GroudingDINO
sys.path.insert(0,"./GroundingDINO")

if not os.path.exists("./sam_vit_h_4b8939.pth"):
    print("目录中未发现必须的模型，请点击网址进行下载:https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
else:
    print("成功加载模型文件sam_vit_h_4b8939.pth")


import argparse
import random
import warnings
import json

import gradio as gr
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy import ndimage
from PIL import Image
from huggingface_hub import hf_hub_download
from segments.utils import bitmap2file

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import (
    clean_state_dict,
)
from GroundingDINO.groundingdino.util.inference import annotate, predict

# segment anything
from segment_anything import build_sam, SamPredictor

# CLIPSeg
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

################## 辅助函数 ####################################
def load_model_hf(model_config_path, repo_id, filename, device):
    """加载和准备Grounding DINO 模型"""
    args = SLConfig.fromfile(model_config_path)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    model = model.to(device)
    return model


############################ 初始化步骤 ###########################
config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filename = "groundingdino_swint_ogc.pth"
sam_checkpoint = "./sam_vit_h_4b8939.pth"

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print("Using device:", device)

if device != "cpu":
    try:
        from GroundingDINO.groundingdino import _C
    except:
        warnings.warn(
            "Failed to load custom C++ ops. Running on CPU mode Only in groundingdino!"
        )

# initialize groundingdino model
dino_model = load_model_hf(config_file, ckpt_repo_id, ckpt_filename, device)

# initialize SAM
sam = build_sam(checkpoint=sam_checkpoint)
sam.to(device=device)
sam_predictor = SamPredictor(sam)

clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
clipseg_model = CLIPSegForImageSegmentation.from_pretrained(
    "CIDAS/clipseg-rd64-refined"
)
clipseg_model.to(device)


###########################  核心函数 ###################################

def load_image_for_dino(image):
    """将图片变成DINO输入的格式"""
    transform = T.Compose(
        [
            T.RandomResize([800],max_size=1333),# 调整尺寸,短边800像素,长边不超过1333像素
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]), # 根据一组固定的均值和标准差来调整图片每个颜色通道的数值
        ]
    )
    dino_image,_ = transform(image,None)
    return dino_image

def dino_detection(
        model,
        image,
        image_array,
        category_names,
        category_name_to_id,
        box_threshold,
        text_threshold,
        device,
        visualize = False,
):
    """使用GroudingDINO 零样本根据输入生成boxes"""
    detection_prompt = " . ".join(category_names)
    dino_image = load_image_for_dino(image)
    dino_image = dino_image.to(device)
    with torch.no_grad():
        boxes,logits,phrases = predict( # 边界框 置信度 在每个框里找到的物体的名字
            model=model,
            image=dino_image,
            caption = detection_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
        )
    category_ids = [category_name_to_id[phrase] for phrase in phrases]
    if visualize:
        annotated_frame = annotate(
            image_source=image_array,boxes=boxes,logits=logits,phrases=phrases
        )
        annotated_frame = annotated_frame[...,::-1] # BGR->RGB
        visualization = Image.fromarray(annotated_frame)
        return boxes,category_ids,visualization
    else:
        return boxes,category_ids,phrases

def sam_masks_from_dino_boxes(predictor,image_array,boxes,device):
    """利用SAM将boxes精确为mask"""
    H,W,_ = image_array.shape
    # 先把格式从"中心点+宽高"翻译为"左上角+右下角",然后再把相对坐标翻译成具体的像素坐标
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes)*torch.Tensor([W,H,W,H])
    transformed_boxes = predictor.transform.apply_boxes_torch(
        boxes_xyxy,image_array.shape[:2]
    ).to(device)
    thing_masks,_,_ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    return thing_masks

def preds_to_semantic_inds(preds,threshold):
    """为图片中的每一个像素做决定，判断到底属于哪个类别"""
    flat_preds = preds.reshape((preds.shape[0],-1)) # 每一行是一个类别的所有像素得分，每一列是一个像素的不同类别的得分

    # 引入一个新的虚拟参赛者"未标记"(unlabeled)
    flat_preds_with_threshold = torch.full(
        (preds.shape[0]+1,flat_preds.shape[-1]),threshold
    )
    flat_preds_with_threshold[1:preds.shape[0]+1,:]=flat_preds

    # 此时每个像素位置中的值将是唯一的stuff 的索引
    semantic_inds = torch.topk(flat_preds_with_threshold,1,dim=0).indices.reshape(
        (preds.shape[-2],preds.shape[-1])
    )

    return semantic_inds

def clipseg_segmentation(
        processor,model,image,category_names,background_threshold,device 
):
    """根据指定的背景名称，利用CLIPSeg为每个背景生成一张概率图，利用最低阈值，对概率图进行整合，得到唯一初步分割报告"""
    inputs = processor(
        text=category_names,
        images=[image]*len(category_names),
        padding="max_length",
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs=model(**inputs)
    logits = outputs.logits # (类别数，高度，宽度)
    if len(logits.shape)==2: # (高度，宽度)
        logits = logits.unsqueeze(0)
    # 重塑 outputs
    upscaled_logits = nn.functional.interpolate(
        logits.unsqueeze(1), # B C H W 
        size = (image.size[1],image.size[0]), # 因为Image的格式一般是(宽度，高度)
        mode = "bilinear", # 双线性插值
    )
    preds = torch.sigmoid(upscaled_logits.squeeze(dim=1)) # 分数转成概率 (B,H,W)
    semantic_inds  = preds_to_semantic_inds(preds,background_threshold)  # (H W) 其中每一个值都是ID
    return preds,semantic_inds

def semantic_inds_to_shrunken_bool_masks(
    semantic_inds, shrink_kernel_size, num_categories # 地盘划分图 收缩多少的指令 多少个类别需要处理
):
    """对模具进行腐蚀，为每个类别制作一张黑白分明的专属图片"""
    shrink_kernel = np.ones((shrink_kernel_size, shrink_kernel_size))

    bool_masks = torch.zeros((num_categories, *semantic_inds.shape), dtype=bool)
    for category in range(num_categories):
        binary_mask = semantic_inds == category
        shrunken_binary_mask_array = ( # 腐蚀操作
            ndimage.binary_erosion(binary_mask.numpy(), structure=shrink_kernel)
            if shrink_kernel_size > 0
            else binary_mask.numpy()
        )
        bool_masks[category] = torch.from_numpy(shrunken_binary_mask_array)

    return bool_masks

def clip_and_shrink_preds(semantic_inds,preds,shrink_kernel_size,num_categories):
    """根据模具，将原始概率图中所有位于核心区域以外的模糊的概率值全部清零"""
    bool_masks = semantic_inds_to_shrunken_bool_masks(
        semantic_inds,shrink_kernel_size,num_categories
    ).to(preds.device)

    sizes = [
        torch.sum(bool_masks[i].int()).item() for i in range(1,bool_masks.size(0))
    ]
    max_size = max(sizes)
    relative_sizes = [size/max_size for size in sizes ] if max_size>0 else sizes

    clipped_preds = torch.zeros_like(preds)
    for i in range(1,bool_masks.size(0)):
        float_mask = bool_masks[i].float()
        clipped_preds[i-1] = preds[i-1]*float_mask

    return clipped_preds , relative_sizes
    

def sample_points_based_on_preds(preds,N):
    """在一张概率图上撒点，点会在概率大的地方集中"""
    height,width = preds.shape
    weights = preds.ravel()
    indices = np.arange(height*width)

    sampled_indices = random.choices(indices,weights=weights,k=N)
    sampled_points = [(index%width,index//width)for index in sampled_indices]

    return sampled_points

def upsample_pred(pred,image_source):
    """SAM在草稿纸上画出的轮廓，完美放大为图像的大小"""
    pred = pred.unsqueeze(dim=0)
    original_height = image_source.shape[0]
    original_width = image_source.shape[1]

    larger_dim = max(original_height, original_width)
    aspect_ratio = original_height / original_width

    # upsample the tensor to the larger dimension
    upsampled_tensor = F.interpolate(
        pred, size=(larger_dim, larger_dim), mode="bilinear", align_corners=False
    )

    # remove the padding (at the end) to get the original image resolution
    if original_height > original_width:
        target_width = int(upsampled_tensor.shape[3] * aspect_ratio)
        upsampled_tensor = upsampled_tensor[:, :, :, :target_width]
    else:
        target_height = int(upsampled_tensor.shape[2] * aspect_ratio)
        upsampled_tensor = upsampled_tensor[:, :, :target_height, :]
    return upsampled_tensor.squeeze(dim=1)

def sam_mask_from_points(predictor,image_array,points):
    """SAM根据点来生成masks"""
    points_array = np.array(points)
    point_labels = np.ones(len(points)) # 我们只需要画一个masks,所以恒等于1
    _,_,logits = predictor.predict(
        point_coords=points_array,
        point_labels=point_labels,
    )
    total_pred = torch.max(torch.sigmoid(torch.tensor(logits)), dim=0)[0].unsqueeze( #取三张中每个像素位置最大值，变成(256,256) 再 升维变成(1,256,256)
        dim=0 
    )
    # logits are 256x256 -> upsample back to image shape
    upsampled_pred = upsample_pred(total_pred, image_array)
    return upsampled_pred

def inds_to_segments_format(
    panoptic_inds, thing_category_ids, stuff_category_names, category_name_to_id
):
    """将这张充满数字ID的内部地图，转换成两种标准的、通用的、可交付的最终成果格式"""
    panoptic_inds_array = panoptic_inds.numpy().astype(np.uint32)
    bitmap_file = bitmap2file(panoptic_inds_array, is_segmentation_bitmap=True)
    segmentation_bitmap = Image.open(bitmap_file)

    stuff_category_ids = [
        category_name_to_id[stuff_category_name]
        for stuff_category_name in stuff_category_names
    ]

    unique_inds = np.unique(panoptic_inds_array)
    stuff_annotations = [
        {"id": i, "category_id": stuff_category_ids[i - 1]}
        for i in range(1, len(stuff_category_names) + 1)
        if i in unique_inds
    ]
    thing_annotations = [
        {"id": len(stuff_category_names) + 1 + i, "category_id": thing_category_id}
        for i, thing_category_id in enumerate(thing_category_ids)
    ]
    annotations = stuff_annotations + thing_annotations

    return segmentation_bitmap, annotations # 加密的ID地图，配套的说明文档

def generate_panoptic_mask(
    image, # 上传的图片
    thing_category_names_string, # 物体名称
    stuff_category_names_string, # 背景名称
    dino_box_threshold=0.3,      # DINO侦探物体置信度阈值
    dino_text_threshold=0.25,    # DINO侦探的文本匹配置信度门槛
    segmentation_background_threshold = 0.1, # 最低分数线:在归类背景元素为哪一类别时用到
    shrink_kernel_size = 20,     # 地图绘制的收缩力度
    num_samples_factor = 1000,   # 撒点机器人的撒点密度
    task_attributes_json = "",   # 高级指令
):
    # 判断是常规输入还是高级指令
    if task_attributes_json != "":
        task_attributes = json.loads(task_attributes_json)
        categories = task_attributes["categories"]
        # 将种类的名字和ID序号映射
        categories_name_to_id = {
            category["name"]:category["id"] for category in categories
        }
        # 将categories 区分为 thing 以及 stuff
        stuff_categories = [
            category 
            for category in categories
            if "has_instances" not in category or not category["has_instances"]
        ]
        thing_categories = [
            category 
            for category in categories
            if "has_instances" in category and category["has_instances"]
        ]
        stuff_category_names = [category["name"] for category in stuff_categories]
        thing_category_names = [category["name"] for category in thing_categories]
        category_names = thing_category_names + stuff_category_names
    else:
        # parse inputs
        thing_category_names = [
            thing_category_name.strip()
            for thing_category_name in thing_category_names_string.split(",")
        ]
        stuff_category_names = [
            stuff_category_name.strip()
            for stuff_category_name in stuff_category_names_string.split(",")
        ]
        category_names = thing_category_names + stuff_category_names
        category_name_to_id = {
            category_name: i for i, category_name in enumerate(category_names)
        }
    
    # 预处理输入图像
    image = image.convert("RGB")
    image_array = np.asarray(image) # 转成numpy 注意: image_array依旧是BGR

    # 预加载 SAM
    sam_predictor.set_image(image_array)

    """ 1. 使用Grouding DINO 检测"thing" 的 boxes """
    thing_category_ids = []
    thing_masks = []
    thing_boxes = []
    detected_thing_category_names = []
    if len(thing_category_names)>0:
        thing_boxes,thing_category_ids,detected_thing_category_names = dino_detection(
            dino_model, # GroudDINO 模型
            image, 
            image_array,
            thing_category_names, # "thing" names
            category_name_to_id , # 名称ID映射字典
            dino_box_threshold,   # DINO侦探的物体置信度门槛
            dino_text_threshold,  # DINO侦探的文本匹配置信度门槛
            device,
        )
        if len(thing_boxes)>0:
            """2. 利用SAM根据boxes 得到 thing 的 segmentation masks"""
            thing_masks = sam_masks_from_dino_boxes(
                sam_predictor,image_array,thing_boxes,device
            )

    if len(stuff_category_names)>0:
        """3. 利用CLIPSeg 模型粗略得到 "stuff"的 masks """
        clipseg_preds,clipseg_semantic_inds = clipseg_segmentation(
            clipseg_processor,
            clipseg_model,
            image,
            stuff_category_names,
            segmentation_background_threshold,
            device
        )

        """4. 从stuff 的masks中,移除物体的区域"""
        clipseg_semantic_inds_without_things = clipseg_semantic_inds.clone()
        if len(thing_boxes)>0:
            combined_things_mask = torch.any(thing_masks,dim=0)
            clipseg_semantic_inds_without_things[combined_things_mask[0]]=0

        """5. 通过一个干净的模具，将preds中所有不干净的地方全部清除掉，为撒点做准备"""
        clipsed_clipped_preds,relative_sizes = clip_and_shrink_preds(
            clipseg_semantic_inds_without_things, # 模具
            clipseg_preds, # 原始概率图
            shrink_kernel_size,
            len(stuff_category_names)+1
        )

        """6. 使用SAM 为 stuff 获取更加精细的分割掩码"""
        sam_preds = torch.zeros_like(clipsed_clipped_preds)
        for i in range(clipsed_clipped_preds.shape[0]):
            clipseg_pred = clipsed_clipped_preds[i]
            # 针对每个类别，根据相对大小撒点
            num_samples = int(relative_sizes[i]*num_samples_factor)
            if num_samples==0:
                continue
            points = sample_points_based_on_preds( # (点数,x,y)
                clipseg_pred.cpu().numpy(),num_samples
            )
            if len(points)==0:
                continue

            pred = sam_mask_from_points(sam_predictor,image_array,points)
            sam_preds[i]=pred
        sam_semantic_inds = preds_to_semantic_inds(
            sam_preds,segmentation_background_threshold
        )


    """7. 将“物体”索引和“背景”索引合并为全景索引(panoptic inds)"""
    panoptic_inds = (
        sam_semantic_inds.clone()
        if len(stuff_category_names)>0
        else torch.zeros(image_array.shape[0],image_array.shape[1],dtype=torch.long)
    )
    ind = len(stuff_category_names)+1 # 跳过背景使用过的ID
    for thing_mask  in thing_masks:
        panoptic_inds[thing_mask.squeeze(dim=0)] = ind
        ind+=1

    panoptic_bool_masks = (
        semantic_inds_to_shrunken_bool_masks(panoptic_inds,0,ind+1)
        .numpy()
        .astype(int)
    )
    panoptic_names = (
        ["unlabeled"] + stuff_category_names + detected_thing_category_names
    )

    subsection_label_pairs = [
        (panoptic_bool_masks[i], panoptic_name)
        for i, panoptic_name in enumerate(panoptic_names)
    ]

    segmentation_bitmap, annotations = inds_to_segments_format(
        panoptic_inds, thing_category_ids, stuff_category_names, category_name_to_id  # 全景地盘图  被成功找到的物体的ID列表 背景类别的名单 类别总字典
    )
    annotations_json = json.dumps(annotations)

    return (image_array, subsection_label_pairs), segmentation_bitmap, annotations_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Panoptic Segment Anything demo", add_help=True)
    parser.add_argument("--debug", action="store_true", help="using debug mode")
    parser.add_argument("--share", action="store_true", help="share the app")
    args = parser.parse_args()

    print(f"args = {args}")

    block = gr.Blocks(title="Panoptic Segment Anything").queue()
    with block:
        with gr.Column():
            title = gr.Markdown(
                "# [Panoptic Segment Anything](https://github.com/segments-ai/panoptic-segment-anything)"
            )
            description = gr.Markdown(
                "Demo for zero-shot panoptic segmentation using Segment Anything, Grounding DINO, and CLIPSeg."
            )
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(sources=["upload"], type="pil")
                    thing_category_names_string = gr.Textbox(
                        label="Thing categories (i.e. categories with instances), comma-separated",
                        placeholder="E.g. car, bus, person",
                    )
                    stuff_category_names_string = gr.Textbox(
                        label="Stuff categories (i.e. categories without instances), comma-separated",
                        placeholder="E.g. sky, road, buildings",
                    )
                    run_button = gr.Button(value="Run")
                    with gr.Accordion("Advanced options", open=False):
                        box_threshold = gr.Slider(
                            label="Grounding DINO box threshold",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.3,
                            step=0.001,
                        )
                        text_threshold = gr.Slider(
                            label="Grounding DINO text threshold",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.25,
                            step=0.001,
                        )
                        segmentation_background_threshold = gr.Slider(
                            label="Segmentation background threshold (under this threshold, a pixel is considered background/unlabeled)",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.1,
                            step=0.001,
                        )
                        shrink_kernel_size = gr.Slider(
                            label="Shrink kernel size (how much to shrink the mask before sampling points)",
                            minimum=0,
                            maximum=100,
                            value=20,
                            step=1,
                        )
                        num_samples_factor = gr.Slider(
                            label="Number of samples factor (how many points to sample in the largest category)",
                            minimum=0,
                            maximum=1000,
                            value=1000,
                            step=1,
                        )
                        task_attributes_json = gr.Textbox(
                            label="Task attributes JSON",
                        )

                with gr.Column():
                    annotated_image = gr.AnnotatedImage()
                    with gr.Accordion("Segmentation bitmap", open=False):
                        segmentation_bitmap_text = gr.Markdown(
                            """
                            分割位图是一张包含分割掩码的32位RGBA格式PNG图片。
                            其 Alpha 通道（透明度）值被设为 255，而其余 RGB 通道中的 24 位数值则对应于标注列表中的对象 ID。
                            未标记区域的值为 0。
                            由于其动态范围很大，该分割位图在图片查看器中会显示为黑色
                            """
                        )
                        segmentation_bitmap = gr.Image(
                            type="pil", label="Segmentation bitmap"
                        )
                        annotations_json = gr.Textbox(
                            label="Annotations JSON",
                        )

            examples = gr.Examples(
                examples=[
                    [
                        "a2d2.png",
                        "car, bus, person",
                        "road, sky, buildings, sidewalk",
                    ],
                    [
                        "bxl.png",
                        "car, tram, motorcycle, person",
                        "road, buildings, sky",
                    ],
                ],
                fn=generate_panoptic_mask,
                inputs=[
                    input_image,
                    thing_category_names_string,
                    stuff_category_names_string,
                ],
                outputs=[annotated_image, segmentation_bitmap, annotations_json],
                cache_examples=True,
            )

        run_button.click(
            fn=generate_panoptic_mask,
            inputs=[
                input_image,
                thing_category_names_string,
                stuff_category_names_string,
                box_threshold,
                text_threshold,
                segmentation_background_threshold,
                shrink_kernel_size,
                num_samples_factor,
                task_attributes_json,
            ],
            outputs=[annotated_image, segmentation_bitmap, annotations_json],
            api_name="segment",
        )

    block.launch(server_name="0.0.0.0", debug=args.debug, share=args.share)
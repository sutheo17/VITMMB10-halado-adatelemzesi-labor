import os
import sys
import json
import logging
import argparse
import math
from pathlib import Path

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import lightning as L
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import wandb

# YOLOv5 könyvtár beállítása a konténeren belül
YOLOV5_DIR = Path('/work/external/yolov5')
if str(YOLOV5_DIR) not in sys.path:
    sys.path.insert(0, str(YOLOV5_DIR))
from models.yolo import Model
from utils.general import non_max_suppression

# ==========================================
# 1. MODELL OSZTÁLYOK ÉS HELPER FÜGGVÉNYEK
# ==========================================

class LitYOLOv5(torch.nn.Module):
    def __init__(self, image_size=640, num_classes=32):
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        yolo_cfg = YOLOV5_DIR / 'models' / 'yolov5s.yaml'
        self.model = Model(yolo_cfg, ch=3, nc=num_classes).float()
        self.model.hyp = {'box': 0.05, 'cls': 0.3, 'obj': 0.7, 'cls_pw': 1.0, 'obj_pw': 1.0, 'fl_gamma': 0.0, 'label_smoothing': 0.0, 'anchor_t': 4.0}


def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if len(boxes1) == 0 or len(boxes2) == 0:
        return torch.zeros((len(boxes1), len(boxes2)), device=boxes1.device)
    x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)

def apply_two_stage_heuristic(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, conflict_iou_threshold=0.75):
    # Stage 1: One per class
    keep_indices = []
    for label in torch.unique(labels):
        class_indices = torch.where(labels == label)[0]
        best_local = torch.argmax(scores[class_indices])
        keep_indices.append(int(class_indices[best_local]))
    keep_indices = sorted(keep_indices)
    
    boxes = boxes[keep_indices]
    scores = scores[keep_indices]
    labels = labels[keep_indices]
    
    # Stage 2: Cross-class NMS
    if len(boxes) <= 1:
        return boxes, scores, labels
        
    order = torch.argsort(scores, descending=True)
    final_keep = []
    
    for idx in order.tolist():
        if not final_keep:
            final_keep.append(idx)
            continue
            
        current_box = boxes[idx].unsqueeze(0)
        kept_boxes = boxes[final_keep]
        ious = box_iou_xyxy(current_box, kept_boxes)[0]
        max_iou = torch.max(ious)
        
        if max_iou <= conflict_iou_threshold:
            final_keep.append(idx)
            
    final_keep = sorted(final_keep)
    return boxes[final_keep], scores[final_keep], labels[final_keep]

def compute_zoom_crop_box(width: int, height: int, zoom_factor=0.90, zoom_center_x=0.50, zoom_center_y=0.666):
    crop_w, crop_h = int(width * zoom_factor), int(height * zoom_factor)
    center_x, center_y = int(width * zoom_center_x), int(height * zoom_center_y)
    x1, y1 = max(0, center_x - crop_w // 2), max(0, center_y - crop_h // 2)
    x2, y2 = min(width, x1 + crop_w), min(height, y1 + crop_h)
    return int(x1), int(y1), int(x2), int(y2)

def prepare_image_for_yolo(orig_img: Image.Image, img_size=640, use_zoom=True):
    orig_w, orig_h = orig_img.size
    
    if use_zoom:
        zx1, zy1, zx2, zy2 = compute_zoom_crop_box(orig_w, orig_h)
        inference_img = orig_img.crop((zx1, zy1, zx2, zy2))
    else:
        # Ha nincs zoom, a teljes képet használjuk, a zoom_box pedig a teljes kép koordinátája (0-tól a szélekig)
        zx1, zy1, zx2, zy2 = 0, 0, orig_w, orig_h
        inference_img = orig_img

    inf_w, inf_h = inference_img.size
    scale = img_size / max(inf_w, inf_h)
    new_w, new_h = int(inf_w * scale), int(inf_h * scale)
    resized = inference_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    pad_w, pad_h = img_size - new_w, img_size - new_h
    padded = TF.pad(resized, (0, 0, pad_w, pad_h), fill=0)
    tensor = TF.to_tensor(padded)
    meta = {'orig_w': orig_w, 'orig_h': orig_h, 'zoom_box': (zx1, zy1, zx2, zy2), 'scale': scale, 'new_w': new_w, 'new_h': new_h}
    return tensor, meta

def yolo_boxes_to_original_xyxy(boxes: torch.Tensor, meta: dict) -> torch.Tensor:
    if len(boxes) == 0: return boxes.clone().float()
    boxes = boxes.clone().float()
    scale = float(meta['scale'])
    zx1, zy1, zx2, zy2 = meta['zoom_box']
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, meta['new_w']) / scale + zx1
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, meta['new_h']) / scale + zy1
    return boxes

# ==========================================
# 2. MAIN FOLYAMAT
# ==========================================

def main(args):
    # Logolás beállítása (Konzol + Fájl)
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger('DentalPipeline')
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(os.path.join(args.log_dir, 'pipeline.log'))
    file_handler.setFormatter(log_formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # WandB beállítása
    if args.use_wandb:
        logger.info("WandB inicializálása...")
        wandb.init(project=args.wandb_project, name="e2e-inference", config=vars(args))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Használt eszköz: {device}")

    # Modellek betöltése
    logger.info("YOLOv5 detekciós modell betöltése...")
    yolo_model = LitYOLOv5(image_size=640, num_classes=32)
    yolo_model.model.load_state_dict(torch.load(args.yolo_weights, map_location='cpu'), strict=False)
    yolo_model.to(device).eval()

    logger.info("EfficientNet-B0 klasszifikációs modell betöltése...")
    classifier_model = torch.load(args.clf_weights, map_location=device, weights_only=False)
    classifier_model.to(device).eval()

    # Kép transzformáció az EfficientNet-hez
    efficientnet_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    logger.info(f"Talált képek száma: {len(image_files)}")

    total_images = len(image_files) # Elmentjük a képek összegzett számát
    total_caries = 0
    total_healthy = 0

    for idx, img_name in enumerate(image_files, start=1):
        logger.info(f"[{idx} / {total_images}] Kép feldolgozása: {img_name}")
        img_path = os.path.join(args.input_dir, img_name)
        
        try:
            orig_img = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Hiba a kép beolvasásakor {img_path}: {e}")
            continue

        # 1. Detekció (YOLO)
        img_tensor, prep_meta = prepare_image_for_yolo(orig_img, use_zoom=args.use_zoom)
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            yolo_out = yolo_model.model(img_tensor)[0]
            # Sima YOLO NMS
            nms_preds = non_max_suppression(yolo_out, conf_thres=0.29, iou_thres=0.45, max_det=200)[0]
        
        if nms_preds is None or len(nms_preds) == 0:
            logger.info(f"{img_name}: Nincsen detektált fog.")
            continue

        # Two-stage heurisztika alkalmazása (Ahogy a notebookban is van)
        boxes = nms_preds[:, :4].cpu()
        scores = nms_preds[:, 4].cpu()
        labels = nms_preds[:, 5].cpu()
        
        filtered_boxes, _, _ = apply_two_stage_heuristic(boxes, scores, labels, conflict_iou_threshold=0.75)
        
        # Visszaszámolás az eredeti képre (A 'yolo_boxes_to_original_xyxy' automatikusan jól fog működni, 
        # mert a prep_meta-ban a zx1, zy1 értékek 0-k lesznek, ha nincs zoom!)
        pred_boxes_original = yolo_boxes_to_original_xyxy(filtered_boxes, prep_meta)
        
        draw = ImageDraw.Draw(orig_img)
        results_json = []

        total_detected_teeth = len(pred_boxes_original) # Előre lekérjük az összes fog számát a képen

        # 2. Klasszifikáció (ResNet)
        for i, box in enumerate(pred_boxes_original):
            logger.info(f"  -> Klasszifikáció: {i+1} / {total_detected_teeth} fog feldolgozása...")
            
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            
            # Védelem a kilógó koordináták ellen
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_img.width, x2), min(orig_img.height, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue

            tooth_crop = orig_img.crop((x1, y1, x2, y2))
            clf_input = efficientnet_transform(tooth_crop).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = classifier_model(clf_input)
                probs = torch.softmax(logits, dim=1)[0]
                healthy_prob = probs[0].item()
                caries_prob = probs[1].item()
            
            is_caries = caries_prob > 0.5
            
            # JSON építés
            results_json.append({
                "box": [x1, y1, x2, y2],
                "status": "Caries" if is_caries else "Healthy",
                "caries_probability": caries_prob
            })

            # 3. Rajzolás
            if is_caries:
                total_caries += 1
                if args.draw_caries:
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                    draw.text((x1, max(0, y1-15)), f"Caries: {caries_prob:.2f}", fill="red")
            else:
                total_healthy += 1
                if args.draw_healthy:
                    draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
                    draw.text((x1, max(0, y1-15)), f"Healthy: {healthy_prob:.2f}", fill="green")

        # 4. Mentés
        base_name = os.path.splitext(img_name)[0]
        out_img_path = os.path.join(args.output_dir, f"{base_name}_annotated.jpg")
        out_json_path = os.path.join(args.output_dir, f"{base_name}_report.json")

        orig_img.save(out_img_path, quality=95)
        with open(out_json_path, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, indent=4)
        
        logger.info(f"{img_name} sikeresen elmentve ({len(results_json)} fog detektálva).")

        if args.use_wandb:
            wandb.log({
                "processed_images": wandb.Image(out_img_path, caption=img_name),
                "image_name": img_name
            })

    logger.info("Folyamat befejeződött!")
    logger.info(f"Összes detektált szuvas (Caries) fog: {total_caries}")
    logger.info(f"Összes detektált egészséges fog: {total_healthy}")

    if args.use_wandb:
        wandb.log({"Total Caries Detected": total_caries, "Total Healthy Detected": total_healthy})
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Dental Caries Pipeline")
    # Alapértelmezett útvonalak átállítva a /work mappára!
    parser.add_argument('--input_dir', type=str, default='/work/input', help='Nyers képek mappája')
    parser.add_argument('--output_dir', type=str, default='/work/output', help='Eredmények mappája')
    parser.add_argument('--log_dir', type=str, default='/work/log', help='Logok mappája')
    parser.add_argument('--yolo_weights', type=str, default='/work/models/detection.pt', help='YOLO súlyok')
    parser.add_argument('--clf_weights', type=str, default='/work/models/classification.pt', help='ResNet súlyok')
    
    # Kép rajzolási beállítások
    parser.add_argument('--draw_healthy', action='store_true', help='Rajzolja be az egészséges fogakat (zöld)')
    parser.add_argument('--draw_caries', action='store_true', help='Rajzolja be a szuvas fogakat (piros)')
    
    parser.add_argument('--use_zoom', action='store_true', help='Alkalmazza a zoom heurisztikát a képeken az inferálás előtt')
    
    # WandB beállítások
    parser.add_argument('--use_wandb', action='store_true', help='WandB logolás bekapcsolása')
    parser.add_argument('--wandb_project', type=str, default='tooth-e2e-pipeline', help='WandB projekt neve')

    args = parser.parse_args()
    
    # Ha egyiket sem kérted kifejezetten, alapból mindkettőt rajzolja:
    if not args.draw_healthy and not args.draw_caries:
        args.draw_healthy = True
        args.draw_caries = True

    main(args)
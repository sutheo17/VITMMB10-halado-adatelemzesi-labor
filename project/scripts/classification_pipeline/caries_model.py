import os
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
import lightning as L
import torchmetrics

from classification_pipeline import (
    load_or_download_classification_dataset,
    ClassificationDownloadConfig,
    build_multiclass_classification_records_from_masks,
    split_grouped_records,
    ToothCropDataset,
    build_classification_image_pipeline,
    build_classification_resize_pipeline
)


class ToothClassificationDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage: str | None = None):
        _, coco_data, image_dirs = load_or_download_classification_dataset(
            ClassificationDownloadConfig(api_key=os.getenv('ROBOFLOW_API_KEY')),
            force_download=self.cfg.force_download
        )
        
        all_records, self.label_map = build_multiclass_classification_records_from_masks(
            coco_data, image_dirs, crop_margin=0.08
        )

        caries_idx = self.label_map.get('Caries', 44)
        
        for rec in all_records:
            rec['label'] = 1 if rec['label'] == caries_idx else 0
        
        self.label_map = {'No caries': 0, 'Caries': 1}
        
        train_rec, val_rec, test_rec = split_grouped_records(
            all_records, train_size=0.7, val_size=0.15, test_size=0.15
        )

        all_labels = [rec['label'] for rec in all_records]
        count_0 = all_labels.count(0)
        count_1 = all_labels.count(1)
        
        total = count_0 + count_1
        weight_for_0 = total / (2 * count_0)
        weight_for_1 = total / (2 * count_1)
        
        self.class_weights = torch.tensor([weight_for_0, weight_for_1], dtype=torch.float)
        print(f"Osztálysúlyok: {self.class_weights}")
        
        # self.train_records_reduced = train_rec[:16] 
        # self.val_records_reduced = val_rec[:16]
        # self.test_records_reduced = test_rec[:16]

        self.train_ds = train_rec
        self.val_ds = val_rec
        self.test_ds = test_rec
        
        aug_pipeline = build_classification_image_pipeline()
        resize_pipeline = build_classification_resize_pipeline(self.cfg.image_size)
        
        self.train_ds = ToothCropDataset(
            self.train_ds,
            image_size=self.cfg.image_size,
            image_transform=aug_pipeline,
            resize_transform=resize_pipeline,
            output_channels=3
        )
        self.val_ds = ToothCropDataset(
            self.val_ds,
            image_size=self.cfg.image_size,
            resize_transform=resize_pipeline,
            output_channels=3
        )
        self.test_ds = ToothCropDataset(
            self.test_ds,
            image_size=self.cfg.image_size,
            resize_transform=resize_pipeline,
            output_channels=3
        )
        
        print(f'Train samples: {len(self.train_ds)}\nVal samples: {len(self.val_ds)}\nTest samples: {len(self.test_ds)}')
        # print(f'Train reduced samples: {len(self.train_records_reduced)}, Val reduced samples: {len(self.val_records_reduced)}, Test reduced samples: {len(self.test_records_reduced)}')
    
    def _loader_kwargs(self) -> dict[str, Any]:
        num_workers = int(self.cfg.num_workers)
        kwargs: dict[str, Any] = {
            'num_workers': num_workers,
            'pin_memory': torch.cuda.is_available(),
        }
        if num_workers > 0:
            kwargs['persistent_workers'] = True
            kwargs['prefetch_factor'] = 2
        return kwargs

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            **self._loader_kwargs()
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            **self._loader_kwargs()
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            **self._loader_kwargs()
        )


class LitToothClassifier(L.LightningModule):
    def __init__(self, cfg, num_classes: int = 2, class_weights=None):
        super().__init__()
        
        self.cfg = cfg
        
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        for param in self.model.parameters():
           param.requires_grad = False

        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
            
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        self.accuracy = torchmetrics.Accuracy(task="binary")
        self.f1 = torchmetrics.F1Score(task="binary")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, weight=self.class_weights)
        
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, weight=self.class_weights)
        
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)
        self.f1(preds, y)
        
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/acc", self.accuracy, prog_bar=True, on_epoch=True)
        self.log("val/f1", self.f1, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        
        logits = self(x)
        
        loss = F.cross_entropy(logits, y, weight=self.class_weights if hasattr(self, 'class_weights') else None)
        
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)
        self.f1(preds, y)
        
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.accuracy, on_step=False, on_epoch=True)
        self.log("test/f1", self.f1, on_step=False, on_epoch=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay
        )
        return optimizer
## Detection pipeline

- `detecton_pipeline/download.py`: downloads the Roboflow `mohamed-uob/denim` dataset and merges all COCO split JSON files instead of loading only the first one.
- `detection_pipeline/records.py`: parses merged COCO annotations into tooth detection records.
- `detection_pipeline/preprocessing.py`: grayscale loading, robust normalization, mild denoising.
- `detection_pipeline/pipelines.py`: train-time detection augmentations.
- `detection_pipeline/datasets.py`: detection datasets and collate function.

## Classification pipeline

- `classification_pipeline/download.py`: downloads the Roboflow `wishis64/se-iwfnq` dataset in COCO segmentation format and merges all split JSON files.
- `classification_pipeline/records.py`: builds positive carious-tooth crop records from the `Caries` segmentation masks by converting each tooth outline into a bounding box and then expanding that box slightly.
- `classification_pipeline/preprocessing.py`: grayscale loading, robust normalization, mild denoising.
- `classification_pipeline/pipelines.py`: bbox-aware classification augmentations applied on the full radiograph before cropping.
- `classification_pipeline/datasets.py`: crop dataset for classification.

## Important behavior

- The downloader merges all subset annotation files under the exported dataset root.
- The classification pipeline uses the `wishis64/se-iwfnq` dataset and derives crop boxes from mask polygons. It falls back to COCO `bbox` only if segmentation coordinates are absent.
- No horizontal or vertical flipping is used.
- Rotations use `cv2.BORDER_REPLICATE`, so rotated crops do not get black side bars.
- Gaussian noise is sampled with `A.GaussNoise(std_range=...)`.

## Note on labels

The segmentation-based classification builder intentionally labels only `Caries` masks as positive tooth crops. It does **not** automatically invent negatives from unrelated classes like implants, fillings, or crowns, because that would contaminate a medical binary classifier. If you want a binary caries-vs-non-caries classifier, add a medically valid negative-tooth source separately.

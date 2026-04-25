## Detection pipeline

- `detection_pipeline/data_preparation/download.py`: downloads the Roboflow `mohamed-uob/denim` dataset and merges all COCO split JSON files instead of loading only the first one.
- `detection_pipeline/data_preparation/records.py`: parses merged COCO annotations into tooth detection records.
- `detection_pipeline/data_preparation/preprocessing.py`: grayscale loading, robust normalization, mild denoising.
- `detection_pipeline/data_preparation/pipelines.py`: train-time detection augmentations.
- `detection_pipeline/data_preparation/datasets.py`: detection datasets and collate function.
- `detection_pipeline/training/`: reserved for training-specific pipeline code.

## Classification pipeline

- `classification_pipeline/data_preparation/download.py`: downloads the Roboflow `wishis64/se-iwfnq` dataset in COCO segmentation format and merges all split JSON files.
- `classification_pipeline/data_preparation/records.py`: builds tooth crop records from segmentation masks for all selected classes by converting each tooth outline into a bounding box and then expanding that box slightly. Supports `max_samples_per_label` to cap examples per class.
- `classification_pipeline/data_preparation/preprocessing.py`: grayscale loading, robust normalization, mild denoising.
- `classification_pipeline/data_preparation/pipelines.py`: bbox-aware classification augmentations applied on the full radiograph before cropping.
- `classification_pipeline/data_preparation/datasets.py`: crop dataset for classification.
- `classification_pipeline/training/`: reserved for training-specific pipeline code.

## Important behavior

- The downloader merges all subset annotation files under the exported dataset root.
- The classification pipeline uses the `wishis64/se-iwfnq` dataset and derives crop boxes from mask polygons. It falls back to COCO `bbox` only if segmentation coordinates are absent.
- No horizontal or vertical flipping is used.
- Rotations use `cv2.BORDER_REPLICATE`, so rotated crops do not get black side bars.
- Gaussian noise is sampled with `A.GaussNoise(std_range=...)`.

## Note on labels

The classification builder now supports multi-class tooth classification by using available segmentation labels from the dataset (including `Caries` and other classes). You can limit class imbalance with a per-label cap.

## Running with `dl-preprocess`

Use the `dl-preprocess` conda environment when running scripts.

- Default:
	- `conda run -n dl-preprocess python scripts/run_example.py`
- Force dataset re-download:
	- `conda run -n dl-preprocess python scripts/run_example.py download`
- Uniform cap per class (e.g. 300):
	- `conda run -n dl-preprocess python scripts/run_example.py max_per_label=300`
- Per-class caps:
	- `conda run -n dl-preprocess python scripts/run_example.py max_per_label=Caries:1000,Filling:500,Implant:500`

The run script also saves a dedicated non-caries visualization panel:

- `output/Classification_train_non-caries.jpg`

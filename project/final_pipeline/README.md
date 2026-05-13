# Caries Detection Pipeline

This repository contains an end-to-end pipeline for detecting dental caries from X-ray images using YOLOv5 and ResNet50.

## Getting Started

### 1. Build the Docker image

Run the following command in the root directory to build the Docker image:

`docker build -t caries-detection .`

### 2. Run the Container

Mount your local `input`, `output`, and `log` folders to the container, provide your Weights & Biases API key (if using it), and run the image. The standard output and errors will be redirected to a `run.log` file.

`docker run --rm --gpus all -v "[path/to/input]:/work/input" -v "[path/to/output]:/work/output" -v "[path/to/log]:/work/log" -e WANDB_API_KEY="[wandb_api_key]" caries-detection > "[path/to/log]\run.log" 2>&1`

*Note: Ensure you replace `[path/to/input]`, `[path/to/output]`, `[path/to/log]`, and `[wandb_api_key]` with your actual local paths and credentials.*

### 3. Check the Logs

Wait for the process to finish. Inside your designated `log` folder, you will find:
* **`run.log`**: Contains the complete log of the Docker execution, from package imports to the end of the pipeline.
* **`pipeline.log`**: Contains only the specific execution logs of the Python pipeline (e.g., model loading, image processing steps).

---

## Run Parameters (Command-Line Arguments)

You can customize the pipeline's behavior using the following command-line flags. If you are using the built-in Docker environment, you generally won't need to change the default paths as they point to the internal `/work` directory.

### Directories (Paths)
* **`--input_dir`**: The input directory containing the raw X-ray images to be processed.
    *(Default: `/work/input`)*
* **`--output_dir`**: The directory where the processing results (annotated `.jpg` images and detailed `.json` reports) will be saved.
    *(Default: `/work/output`)*
* **`--log_dir`**: The directory where text-based log files (e.g., `pipeline.log`) generated during the process will be saved.
    *(Default: `/work/log`)*

### Model Weights
* **`--yolo_weights`**: The exact path to the weights file for the YOLO tooth detection model (bounding boxes).
    *(Default: `/work/models/detection.pt`)*
* **`--clf_weights`**: The exact path to the weights file for the ResNet classification model (caries detection).
    *(Default: `/work/models/classification.ckpt`)*

### Visualization Settings (Image Output)
These flags control what gets drawn on the final output images.
* **`--draw_healthy`**: If enabled, the program will draw a **green box** around teeth classified as healthy.
* **`--draw_caries`**: If enabled, the program will draw a **red box** around teeth classified as having caries (decay).

> **Tip:** For convenience, if you don't specify either of these visualization flags when running the script, the program will automatically draw **both categories (green and red boxes)** on the images. If you only want to see one specific type, explicitly provide only that flag!

### Logging and Monitoring (Weights & Biases)
* **`--use_wandb`**: Enables cloud-based logging via Weights & Biases (WandB), which uploads the processed images and basic statistics. To use this, you must provide your `WANDB_API_KEY` as an environment variable when running the Docker container!
* **`--wandb_project`**: Specifies the WandB project name where the run results will be synchronized.
    *(Default: `tooth-e2e-pipeline`)*

---

## 💡 Example Usage

Here is a practical example of how to run the pipeline if you **only want to draw red boxes on caries**, enable **WandB logging**, and assign the run to a custom project named `my-dental-project`. 

*(Notice how custom arguments are added to the very end of the command, right after `caries-detection` but before the log redirection `>`)*:

`docker run --rm --gpus all -v "D:\projects\dental\input:/work/input" -v "D:\projects\dental\output:/work/output" -v "D:\projects\dental\log:/work/log" -e WANDB_API_KEY="abc123yourkeyhere" caries-detection --draw_caries --use_wandb --wandb_project "my-dental-project" > "D:\projects\dental\log\run.log" 2>&1`
# Seeking AI Segment Anything (SA2)
## SA2: Vision-Oriented MultiModal AI


SA2 integrates SOTA (state-of-the-art) models and provides a vision-oriented multi-modal framework. It's not an LLM (large-language model), but comprises multiple large-scale models, some of which are built on top of cutting-edge foundation models.

## Purposes

## Installation

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

### SA2 Unified MultiModel Framework (UMMF)

```bash
git clone git@github.com:LHBuilder/SA-Segment-Anything.git
```

### Meta SAM
Install Segment Anything:

Please follow the instructions [here](https://github.com/LHBuilder/SA-Segment-Anything/blob/main/SAM/README.md) to install Meta SAM.

Or

```bash
pip install segment_anything
```


The following optional dependencies are necessary for mask post-processing, saving masks in COCO format, the example notebooks, and exporting the model in ONNX format. `jupyter` is also required to run the example notebooks.

```
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

### YOLO-NAS
Please follow the instructions [here](https://github.com/LHBuilder/SA-Segment-Anything/blob/main/YOLO/README.md) to install YOLO-NAS.

Or 

```bash
pip install super-gradients
```

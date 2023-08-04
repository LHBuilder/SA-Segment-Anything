![](./public/SA2_logo.png)


## SA2: Vision-Oriented MultiModal AI


SA2 integrates SOTA (state-of-the-art) models and provides a vision-oriented multi-modal framework. It's not an LLM (large-language model), but comprises multiple large-scale models, some of which are built on top of cutting-edge foundation models.

## Purposes
The surging momentum of generative AI (GAI) heralds the dawn of a new era in Artificial General Intelligence (AGI). LLMs and CV multi-modal large-scale models are two dominant trends in the GAI age. ChatGPT and GPT-4 set a ceiling bar for LLMs, but CV multi-modal large-scale models are still emerging.

Seeking AI is an AI company focusing on AI for Industry. We have built a solid foundation for AI innovation and standardized data development over the past 5 years. We roll out SA2 to help the community of CV multi-modal large-scale models. This SA2 project has the following purposes:

1. Provide a unified multi-modal framework for different applications based on multi-modal foundation models.
2. Integrate the SOTA vision models to build up a complete multi-modal platform by leveraging the real SOTA parts of these models.
3. Focus on vision-oriented AI to accelerate CV development compared with the status quo of LLMs.


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

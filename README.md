# Robust Vision-Language Models via Manifold-Adversarial Adapters [ICML 2026]

Official code release for Robust Vision-Language Models via Manifold-Adversarial Adapters, accepted to ICML 2026.

## Links

- Model checkpoint: [acnul/maa/maa.pth](https://huggingface.co/acnul/maa/blob/main/maa.pth)
- Training data: [acnul/maa-datasets](https://huggingface.co/datasets/acnul/maa-datasets)
- Base model: [liuhaotian/llava-v1.6-mistral-7b](https://huggingface.co/liuhaotian/llava-v1.6-mistral-7b)

## Installation

```bash
conda create -n maa python=3.10 -y
conda activate maa
pip install --upgrade pip
pip install -e .
```

The release has been tested with `torch==2.9.1`, `torchvision==0.24.1`, and `transformers==4.46.3`. If your system requires a specific CUDA build, install the matching PyTorch wheel first, then run `pip install -e .`.

For VLMEvalKit evaluation:

```bash
pip install -e eval/VLMEvalKit
```

Optional LLaVA utilities:

```bash
pip install -e ".[quantization]"
pip install -e ".[serve]"
pip install -e ".[llava-train]"
```

## Checkpoint

Download the released MAA adapter:

```bash
pip install -U huggingface_hub
hf download acnul/maa maa.pth --repo-type model --local-dir checkpoints/maa
```

The checkpoint contains only adapter weights. The LLaVA base model is loaded separately from `liuhaotian/llava-v1.6-mistral-7b`.

## Data

Download and extract the paired training data. The dataset is released as a multi-volume zip archive: `maa-datasets.z01`, `maa-datasets.z02`, ..., and `maa-datasets.zip`. All parts must be in the same directory before extraction.

```bash
mkdir -p data/maa-datasets-archive
hf download acnul/maa-datasets \
  --repo-type dataset \
  --include "maa-datasets.z*" \
  --include "maa-datasets.zip" \
  --local-dir data/maa-datasets-archive

# Install 7-Zip if it is not available.
apt-get update && apt-get install -y p7zip-full

7z x data/maa-datasets-archive/maa-datasets.zip -odata
```

Expected layout:

```text
data/maa-datasets/
  CC12M/
    ref/
    dis/
    metadata.jsonl
  COCO/
    ref/
    dis/
    metadata.jsonl
  DIV2K/
  Flickr2K/
  LAION/
  TextVQA/
```

Files in `dis` are matched to files in `ref` by stem. Suffixes such as `_v2` denote additional degraded versions of the same reference image.

## Training

```bash
python -m maa.train_maa \
  --model_name_or_path liuhaotian/llava-v1.6-mistral-7b \
  --dataset_path data/maa-datasets \
  --output_dir checkpoints/maa
```

Training defaults follow the paper. Run `python -m maa.train_maa --help` for all options.

## Evaluation

```bash
export MAA_BASE_MODEL=liuhaotian/llava-v1.6-mistral-7b
export MAA_ADAPTER_PATH=checkpoints/maa/maa.pth
python eval/VLMEvalKit/run.py --model maa_llava --data <DATASET_NAME>
```

The released checkpoint targets `liuhaotian/llava-v1.6-mistral-7b`. Other LLaVA variants are best-effort.

## Citation

The paper has been accepted to ICML 2026. The official BibTeX entry will be added after the camera-ready version appears in the ICML proceedings.

## Acknowledgements

This repository builds on [LLaVA](https://github.com/haotian-liu/LLaVA) and [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). The included upstream components retain their original Apache-2.0 license notices. Users are responsible for complying with the licenses of base models, datasets, and released checkpoints.

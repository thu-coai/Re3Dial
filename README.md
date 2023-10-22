# Re$^3$Dial: Retrieve, Reorganize and Rescale Conversations for Long-Turn Open-Domain Dialogue Pre-training

This repository contains data, code, and pre-trained retriever models for our EMNLP 2023 paper
> [Re$^3$Dial: Retrieve, Reorganize and Rescale Conversations for Long-Turn Open-Domain Dialogue Pre-training](https://arxiv.org/abs/2305.02606)

In this work, we propose Re$^3$Dial (Retriever, Reorganize, and Rescale), a framework to automatically construct billion-scale long-turn dialogues by reorganizing existing short-turn ones.

![](figures/framework.png "Re$^3$Dial Framework")
![](figures/example.png "Example of the constructed long-turn dialogue by Re$^3$Dial")

### 1. Install

```
conda create -n redial python=3.8
pip install -r requirements.txt
```

### 2. Prepare Original Dialogue Corpus

- each line is a multi-turn dialogue seperated by `\t`
- for example: 'u1\tu2\tu3\tu4'

### 3. Construct Long-turn Dialogue Corpus

Before running the code, please change some arguments (e.g., the path of pre-trained models, data path, save path) in the script according to your own path.

#### 3.1 Training and Inference of UDSR

- train

```bash
cd src
bash scripts/train.sh 8 # 8 means data-parallel training over 8 GPUs
```

- inference

```bash
cd src
bash scripts/predict.sh 8 # 8 means data-parallel inference over 8 GPUs
```

#### 3.2 Retrieval

- retrieval on a single GPU

```bash
cd src
bash scripts/retrieve.sh
```

- data-parallel retrieval over multi-GPUs

```bash
cd src
python scripts/parallel_retrieve.py
```

#### 3.3 Build Long-turn Dialogue Corpus

```bash
cd src
bash scripts/build_corpus.sh
```

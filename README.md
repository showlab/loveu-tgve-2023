<p align="center">
  <a href="https://sites.google.com/view/loveucvpr23/track4"><img width=800px src="https://user-images.githubusercontent.com/55792387/236656719-b10b5fb0-b2bf-4de0-89cf-146360b90276.png" alt="logo"/></a>
</p>

**Welcome to the [Text-Guided Video Editing (TGVE)](https://sites.google.com/view/loveucvpr23/track4) 
competition of [LOVEU Workshop @ CVPR 2023](https://sites.google.com/view/loveucvpr23/home)!** 

This repository contains the data, baseline code and submission guideline for the LOVEU-TGVE competition. 
If you have any questions, please feel free to reach out to us at loveu-tgve@googlegroups.com.

## Introduction

Leveraging AI for video editing has the potential to unleash creativity for artists across all skill levels. 
The rapidly-advancing field of Text-Guided Video Editing (TGVE) is here to address this challenge.
Recent works in this field include [Tune-A-Video](https://tuneavideo.github.io/), [Gen-2](https://research.runwayml.com/gen2), 
and [Dreamix](https://dreamix-video-editing.github.io/). 

In this competition track, we provide a standard set of videos and prompts. As a researcher, you will develop a model 
that takes a video and a prompt for how to edit it, and your model will produce an edited video. For instance, 
you might be given a video of “people playing basketball in a gym,” and your model will edit the video to 
“dinosaurs playing basketball on the moon.”

With this competition, we aim to offer a place where researchers can rigorously compare video editing methods. 
After the competition ends, we hope the LOVEU-TGVE-2023 dataset can provide a standardized way of comparing AI 
video editing algorithms.

## Dates
- May 1, 2023: The competition data and baseline code become available.
- May 8, 2023: The leaderboard and submission instructions become available.
- June 5, 2023: Deadline for submitting your generated videos.
- June 18, 2023: LOVEU 2023 Workshop. Presentations by winner and runner-up.

## Data

We conducted a survey of text guided video editing papers, and we found the following patterns in how they evaluate their work:
- Input: 10 to 100 videos, with ~3 editing prompts per video
- Human evaluation to compare the generated videos to a baseline

We follow a similar protocol in our [LOVEU-TGVE-2023 dataset](https://drive.google.com/file/d/1D7ZVm66IwlKhS6UINoDgFiFJp_mLIQ0W/view?usp=sharing). Our dataset consists of 76 videos. Each video has 4 editing prompts. 
All videos are creative commons licensed. Each video consists of either 32 or 128 frames, with a resolution of 480x480.

## Submission

Please ensure that you complete all the necessary details and upload your edited videos and report on the 
[LOVEU-TGVE Registration & Submission Form](https://forms.gle/or2Yop5CAHm6pSNZ9) prior to **June 1, 2023**.

**NOTE**: 
- Each team should register only once. Registering multiple times using different accounts is not permitted. 
- The Google form can be edited multiple times. Each submission will overwrite the previous one.
- Only the **latest** submission will be sent to human evaluation.

### Edited Videos

Kindly upload a zip file named `YOUR-TEAM-NAME_videos.zip` to Edited Videos portal in the Google form. 
The uploaded zip file should include **ALL** edited prompts in the LOVEU-TGVE-2023 dataset. 
 
Please adhere to the following format and folder structure when saving your edited videos.
```bash
YOUR-TEAM-NAME_videos.zip
├── DAVIS_480p
│   ├── stunt
│   │   ├── style
│   │   │   ├── 00000.jpg
│   │   │   ├── 00001.jpg
│   │   │   ├── ...
│   │   ├── object
│   │   │   ├── 00000.jpg
│   │   │   ├── 00001.jpg
│   │   │   ├── ...
│   │   ├── background
│   │   │   ├── 00000.jpg
│   │   │   ├── 00001.jpg
│   │   │   ├── ...
│   │   ├── multiple
│   │   │   ├── 00000.jpg
│   │   │   ├── 00001.jpg
│   │   │   ├── ...
│   ├── gold-fish
│   ├── drift-turn
│   ├── ...
├── youtube_480p
├── videvo_480p
```

### Report
Use CVPR style (double column) in the form of 3-6 pages or NeurIPS style (single column) in the form of 6-10 pages 
inclusive of any references. Please explain clearly...
- Your data, supervision, and any pre-trained models
- Pertinent hyperparameters such as classifier-free guidance scale
- If you used prompt engineering, please describe your approach

Please name your report as `YOUR-TEAM-NAME_report.pdf`, and submit it to Report portal in the Google form.

## Evaluation

After submission, your edited videos will undergo automatic evaluation on our server based on [CLIP score](https://arxiv.org/abs/2104.08718) 
and [PickScore](https://arxiv.org/abs/2305.01569). Our system will calculate these scores and present them on the competition website's leaderboard. 
The leaderboard will be refreshed every 24 hours to showcase the most current scores. In addition, you may compute these automatic metrics locally 
utilizing the evaluation code provided in this repository.

After all submissions are uploaded, we will run a human-evaluation of all submitted videos. Specifically, we will have human labelers compare all submitted videos to the baseline videos that were edited with the Tune-A-Video model. Labelers will evaluate videos on the following criteria:
- Text alignment: How well does the generated video match the caption?
- Structure: How well does the generated video preserve the structure of the original video?
- Quality: Aesthetically, how good is this video?

We will choose a winner and a runner-up based on the human evaluation results.

## Baseline code

**[Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation](https://arxiv.org/abs/2212.11565)**
<br/>
[Jay Zhangjie Wu](https://zhangjiewu.github.io/), 
[Yixiao Ge](https://geyixiao.com/), 
[Xintao Wang](https://xinntao.github.io/), 
[Stan Weixian Lei](), 
[Yuchao Gu](https://ycgu.site/), 
[Yufei Shi](),
[Wynne Hsu](https://www.comp.nus.edu.sg/~whsu/), 
[Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en), 
[Xiaohu Qie](https://scholar.google.com/citations?user=mk-F69UAAAAJ&hl=en), 
[Mike Zheng Shou](https://sites.google.com/view/showlab)
<br/>

[![Project Website](https://img.shields.io/badge/Project-Website-orange)](https://tuneavideo.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2212.11565-b31b1b.svg)](https://arxiv.org/abs/2212.11565)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/showlab/Tune-A-Video/blob/main/notebooks/Tune-A-Video.ipynb)
[![GitHub](https://img.shields.io/github/stars/showlab/Tune-A-Video?style=social)](https://github.com/showlab/Tune-A-Video)

### Setup

#### Install requirements

```bash
git clone https://github.com/showlab/loveu-tgve-2023.git
cd loveu-tgve-2023
pip install -r requirements.txt
```

Installing [xformers](https://github.com/facebookresearch/xformers) is highly recommended for more efficiency and speed on GPUs. 
To enable xformers, set `enable_xformers_memory_efficient_attention=True` (default).

#### Download pretrained weights

[Stable Diffusion](https://arxiv.org/abs/2112.10752) is a latent text-to-image diffusion model capable of generating 
photo-realistic images given any text input. The pre-trained Stable Diffusion models can be downloaded from HuggingFace 
(e.g., [Stable Diffusion v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4), [v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
[v2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1)). 

```bash
git lfs install
git clone https://huggingface.co/CompVis/stable-diffusion-v1-4 checkpoints/stable-diffusion-v1-4
```

#### Prepare the data

Download the [loveu-tgve-2023.zip](https://drive.google.com/file/d/1D7ZVm66IwlKhS6UINoDgFiFJp_mLIQ0W/view?usp=sharing), and unpack it to the `./data` folder. 

```bash
unzip loveu-tgve-2023.zip -d ./data
```

#### Create config files

Modify the required paths in `scripts/create_configs.py` and run:
```bash
python scripts/create_configs.py
```

### Training

To fine-tune the text-to-image diffusion models for text-to-video generation, run this command:

```bash
accelerate launch train_tuneavideo.py --config="configs/loveu-tgve-2023/DAVIS_480p/gold-fish.yaml"
```

<details><summary>To run training for all</summary>

```bash
CONFIG_PATH=./configs/loveu-tgve-2023
for config_file in $(find $CONFIG_PATH -name "*.yaml"); do
  accelerate launch train_tuneavideo.py --config=$config_file
done
```

</details>


**Tips**: 

- Fine-tuning a 32-frame video (480x480) requires approximately 300 to 500 steps, taking around 10 to 15 minutes when utilizing one A100 GPU (40GB). 
- Fine-tuning a 128-frame video (480x480) necessitates more VRAM (more than 40GB) and can be executed using one A100 GPU (80GB). In case your VRAM is restricted, you may split the video into 32-frame clips for fine-tuning.

### Inference

Once the training is done, run inference:

```bash
python test_tuneavideo.py --config="configs/loveu-tgve-2023/DAVIS_480p/gold-fish.yaml" --fp16
```

<details><summary>Convert GIF to JPG</summary>

#### ffmpeg
```bash
ffmpeg -r 1 -i input.gif %05d.jpg
```

#### PIL
```python
from PIL import Image

gif = Image.open("input.gif")
frame_index = 0
while True:
    try: gif.seek(frame_index)
    except EOFError: break
    image = gif.convert('RGB')
    image.save("{:05d}.jpg".format(frame_index))
    frame_index += 1
```

</details>


### Automatic metrics

In addition to human evaluation, we employ [CLIP score](https://arxiv.org/abs/2104.08718) 
and [PickScore](https://arxiv.org/abs/2305.01569) and as automatic metrics to measure the quality of generated videos.

- **CLIP score for frame consistency**: we compute CLIP image embeddings on all frames of output video and report the average cosine similarity between all pairs of video frames.
- **CLIP score for textual alignment**: we compute average CLIP score between all frames of output video and corresponding edited prompt.
- **PickScore for human preference**: we compute average PickScore between all frames of output video and corresponding edited prompt.

The evaluation code is provided in `scripts/run_eval.py`. See [Submission Section](#submission)
for the format and structure of your submission folder.

```bash
python scripts/run_eval.py --submission_path="PATH_TO_YOUR_SUBMISSION_FOLDER" --metric="clip_score_text"
```

### Results

The full results of Tune-A-Video on our LOVEU-TGVE-2023 dataset can be downloaded from [here]().

https://user-images.githubusercontent.com/55792387/236660703-f0073cd6-55bb-4fb5-8b61-8a57cff658f2.mp4


### Citation
```bibtex
@article{wu2022tuneavideo,
    title={Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation},
    author={Wu, Jay Zhangjie and Ge, Yixiao and Wang, Xintao and Lei, Stan Weixian and Gu, Yuchao and Hsu, Wynne and Shan, Ying and Qie, Xiaohu and Shou, Mike Zheng},
    journal={arXiv preprint arXiv:2212.11565},
    year={2022}
}
```

import os
import pandas as pd
from glob import glob
from omegaconf import OmegaConf

DATA_PATH = "./data/loveu-tgve-2023"
CONFIG_PATH = "./configs/loveu-tgve-2023"
OUTPUT_PATH = "./outputs/loveu-tgve-2023"
PRETRAINED_MODEL_PATH = "./checkpoints/stable-diffusion-v1-4"

df = pd.read_csv(f"{DATA_PATH}/LOVEU-TGVE-2023_Dataset.csv")
sub_dfs = {
    'DAVIS_480p': df[1:17],
    'youtube_480p': df[19:42],
    'videvo_480p': df[44:82],
}

for sub_name, sub_df in sub_dfs.items():
    for index, row in sub_df.iterrows():
        config = OmegaConf.load("./configs/template.yaml")
        video_name = row['Video name']
        train_prompt = row['Our GT caption']
        edited_prompts = {x.split(" ")[0].lower(): str(row[x]).strip() for x in [
            "Style Change Caption",
            "Object Change Caption",
            "Background Change Caption",
            "Multiple Changes Caption"
        ]}

        config.pretrained_model_path = PRETRAINED_MODEL_PATH
        config.train_data.video_path = f"{DATA_PATH}/{sub_name}/480p_frames/{video_name}"
        if not os.path.exists(config.train_data.video_path):
            raise FileNotFoundError(config.train_data.video_path)
        config.train_data.num_frames = len(glob(f"{config.train_data.video_path}/*.jpg"))
        config.train_data.frame_rate = 1
        config.checkpointing_steps = 100
        config.validation_steps = 100

        config.train_data.prompt = train_prompt
        config.validation_data.prompts = edited_prompts
        config.output_dir = f"{OUTPUT_PATH}/{sub_name}/{video_name}"

        save_config_path = f"{CONFIG_PATH}/{sub_name}/{video_name}.yaml"
        os.makedirs(os.path.dirname(save_config_path), exist_ok=True)
        OmegaConf.save(config, save_config_path)


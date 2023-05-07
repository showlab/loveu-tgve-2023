import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from glob import glob


class TuneAVideoDataset(Dataset):
    def __init__(
            self,
            video_path: str,
            prompt: str,
            height: int = 480,
            width: int = 480,
            num_frames: int = 32,
            frame_rate: int = 1,
            ext: str = "jpg",
    ):
        self.video_path = video_path
        self.frames = [Image.open(x) for x in sorted(glob(f"{video_path}/*.{ext}"))]
        self.prompt = prompt
        self.prompt_ids = None
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.frame_rate = frame_rate

        self.image_transforms = transforms.Compose([
            transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return 1

    def __getitem__(self, index):
        start_index = 0
        if self.num_frames < len(self.frames):
            start_index = random.randint(0, len(self.frames) - self.num_frames)
        frames = self.frames[start_index:start_index+self.num_frames]
        frames = [self.image_transforms(x) for x in frames]
        video = torch.stack(frames)

        example = {
            "pixel_values": (video * 2.0 - 1.0),
            "prompt_ids": self.prompt_ids
        }

        return example

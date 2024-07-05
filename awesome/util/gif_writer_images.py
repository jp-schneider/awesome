
from typing import Any, Dict, List
import uuid
import os
import matplotlib.pyplot as plt
import imageio
from awesome.util.path_tools import numerated_file_name
from tqdm.auto import tqdm

class GifWriterImages():

    def __init__(self, name: str, images: List[str], output_dir="output/gif_writer") -> None:
        self.images = images
        self.name = name
        self.gif_path = None
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def __call__(self, append_times: int = 12) -> Any:
        images = self.images
        duration = (append_times) * (1000 / 24)
        self.gif_path = numerated_file_name(os.path.join(self.output_dir, self.name))
        with imageio.get_writer(self.gif_path, mode='I', loop=0, duration=duration) as writer:
            it = tqdm(images, delay=5, desc="Writing gif")
            for filename in it:
                image = imageio.imread(filename)
                writer.append_data(image)


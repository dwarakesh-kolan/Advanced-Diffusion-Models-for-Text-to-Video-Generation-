pip install diffusers
pip install transformers
pip install accelerate
pip install torch

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b",torch_dtype=torch.float16,
variant="fp16")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe.enable
model_cpu_offload()

import cv2
import numpy as np

def generate_video(prompt):
  video_frames = pipe(prompt, num_inference_steps=25).frames
  print(len(video_frames)) 
  video_path = export_to_video(video_frames)
  print('Video saved to:', video_path)

def export_to_video(video_frames, output_video_path='output.mp4', fps=24):
 video_frames = np.array(video_frames)
 print("Shape of video_frames:", video_frames.shape)

if len(video_frames.shape) == 4:
      num_frames, h, w, c = video_frames.shape
elif len(video_frames.shape) == 5:
      batch_size, num_frames, h, w, c = video_frames.shape
      video_frames = video_frames[0]
      num_frames = video_frames.shape[0]
else:
    raise ValueError("Unexpected shape of video_frames: {}".format(video_frames.shape))

if c == 1:
    video_frames = np.repeat(video_frames, 3, axis=-1)
    c = 3 

if video_frames.dtype != np.uint8:
      video_frames=(video_frames * 255).astype(np.uint8)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))

for frame in video_frames:
      if frame.shape[-1] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
       video_writer.write(frame)
       video_writer.release()
       return output_video_path

# generate_video("A dog playing in the park")
# generate_video("A girl eating IceCream")
# generate_video("A fish swimming in an aquarium")
# generate_video("A panda eating bamboo on the rock")
# generate_video("Spiderman fighting villains in an alley")
# generate_video("A colorful hot air balloon flying over a mountain range")
# generate_video("A person walking along the beach")
# generate_video("A cup of coffee on a table near a window")

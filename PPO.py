
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import mani_skill.envs
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.sb3 import ManiSkillSB3VectorEnv
import torch
from PIL import Image
import os
import shutil


num_envs = 16
max_episode_steps = 100

#Initialize a tensor to store the info values when the test succeeds
success = torch.zeros(num_envs, dtype=torch.bool).cuda()

#Loac the model that was previously trained, or another model
model = PPO.load('ppo_pickcube')

#Visualize the trained agent
eval_vec_env = gym.make("PickCube-v1", num_envs=num_envs, control_mode='pd_joint_delta_pos', render_mode="rgb_array", sim_backend='gpu')
eval_vec_env = RecordEpisode(eval_vec_env, output_dir="Videos", save_video=True, save_trajectory=False, max_steps_per_video=max_episode_steps)
obs, _ = eval_vec_env.reset()

for i in range(max_episode_steps):
    action, _states = model.predict(obs.cpu().numpy(), deterministic=True)
    obs, rewards, dones, _, info = eval_vec_env.step(action)
    #print(info['success'])

    # Collect success info
    step_success = info['success']  # This is a tensor
    #This performs a logical or statement so if either the value in success or the
    #value in step_success is True, it will set the value of success to true
    success = torch.logical_or(success, step_success)  


# Count the number of successful environments
success_count = torch.sum(success).item()
success_rate = success_count / num_envs

# Save video with a better name than 0
video_dir = "Videos"
video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

# Get the most recent video file and rename it
latest_video = max(video_files, key=lambda f: os.path.getctime(os.path.join(video_dir, f)))
# pick the new video name
new_video_name = f"{'pickcube'}.mp4"

# Rename the video
shutil.move(os.path.join(video_dir, latest_video), os.path.join(video_dir, new_video_name))
print(f"Video saved and renamed to: {new_video_name}")


#Save image
image_tensor = eval_vec_env.render()[0] # torch tensor image as 256 x 256 x 3
# Convert torch tensor to PIL Image
image = Image.fromarray(image_tensor.cpu().numpy().astype('uint8'))
image.save("pickcube.jpg")
# Print success data
print(f"Number of successful episodes: {success_count}")
print(f"PickCube Success rate: {success_rate:.2%}")
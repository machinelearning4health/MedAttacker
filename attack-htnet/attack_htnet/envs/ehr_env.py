import gym
from gym import error, spaces, utils
from gym.utils import seeding
import torch

class EhrEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, ehr_input, gt, original_score, attack_model, threshold, pos_mat, input_time, input_visit_length):
      self.ehr_input = ehr_input.detach().clone()
      self.ehr_time = input_time.clone().detach()
      self.ehr_visit_length = input_visit_length.copy()

      self.attack_model = attack_model
      self.threshold = threshold
      self.gt = gt
      self.original_score = original_score
      self.pos_mat = pos_mat
      self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  def step(self, new_observation):
      ob = new_observation
      new_score = self.attack_model(new_observation, self.ehr_time, self.ehr_visit_length)

      if(self.gt == 1):
          reward = self.original_score - new_score
      else:
          reward = new_score - self.original_score
      
      attack_label = (new_score > self.threshold).int().to(self.device)
      
      if(attack_label != self.gt):
          episode_over = True
      else:
          episode_over = False

      return ob, reward.item(), episode_over, {}


  def reset(self):
      self._just_reset = True
      ob = self.ehr_input
      self._just_reset = False
      return ob
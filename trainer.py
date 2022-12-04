from transformer import *
import numpy as np
import torch
from torch.optim import Adam
import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import inspect
from sklearn import metrics
import pickle
import gym
from torch.distributions import Categorical
import os 



class MedAttacker:
    def __init__(self, target_model, train_dataloader, validate_dataloader, test_dataloader, with_cuda=0,lr=0.001,output_dir=None):
        super().__init__()
        self.device = torch.device("cuda:0" if with_cuda==1 else "cpu")
        self.target_model = target_model.to(self.device)
        self.train_dataloader = train_dataloader
        self.validate_dataloader = validate_dataloader
        self.test_dataloader = test_dataloader
        self.optim = Adam(self.target_model.parameters(), lr=lr)
        self.criterion1 = nn.NLLLoss(ignore_index=0)
        self.output_dir = output_dir
        self.BCELoss = nn.BCELoss()
        self.code_catg_pair = pickle.load(open("code_catg_dict.pkl","rb"))
        self.catg_codes = pickle.load(open("catg_dict.pkl","rb"))


    def attack(self,epoch):
        step = 0
        
        labels = np.array([])
        preds = np.array([])
         
        total_attack_success = 0
        first_patient_index = 0
        for data in self.test_dataloader:
            step += 1
            print("step:{}".format(step))

            padding_input, input_labels, visit_time, visit_length, x = data
            padding_input, input_labels, visit_time = Variable(padding_input), Variable(input_labels), Variable(visit_time)
            padding_input, input_labels, visit_time = padding_input.to(self.device), input_labels.to(self.device), visit_time.to(self.device) 
            predict_output = self.target_model(padding_input, visit_time,visit_length).squeeze(1)

            labels = input_labels.cpu().numpy()
            preds_temp = predict_output.detach().cpu().numpy()

            t = Variable(torch.Tensor([0.45])).to(self.device)
            
            predict_label = (predict_output > t).float().to(self.device)
            predict_label_int = predict_label.type(input_labels.dtype)

            ground_truth = torch.from_numpy(labels).float().to(self.device).type(input_labels.dtype)
            result = (ground_truth == predict_label_int)
            index_correct = (result == True).nonzero()
            index_correct_np = index_correct.view(-1).cpu().numpy()
            attack_success = 0

            for index in index_correct_np:

                patient_index = first_patient_index+index
                print("attacking exampe:{}".format(index))
                orig_example = padding_input[index].unsqueeze(0)
                orig_time = visit_time[index].unsqueeze(0)
                orig_visit_length = [visit_length[index]]
                orig_example_output = self.target_model(orig_example, orig_time, orig_visit_length).squeeze(1)
                ground_truth_label = input_labels[index] 


                copy_test = padding_input[index].unsqueeze(0).detach().clone()
                copy_time = visit_time[index].unsqueeze(0).detach().clone()
                copy_visit_length = orig_visit_length.copy()

                max_visit_num = 20
                record_element = 0
                for count_index, count_result in enumerate(x[index]):
                    if count_index < max_visit_num:
                        record_element = record_element+len(count_result)

                count = 0    
                record = x[index]
                position_mat = np.zeros((record_element, 2), dtype = np.int32)
                saliency_set = torch.zeros(record_element).cuda().detach()

                visit_all_pos = []
                for visit_index, visit_result in enumerate(x[index]):
                    if visit_index >= max_visit_num:
                        break

                    visit_each_pos = []
                    for visit_result_index, diagnosis_code  in enumerate(visit_result):
                        if visit_index < max_visit_num:

                            position_mat[count][0] = visit_index
                            position_mat[count][1] = visit_result_index

                            orig_code = record[visit_index][visit_result_index]
                            empty_replace =  orig_example.clone().detach()
                            empty_time = orig_time.clone().detach()
                            empty_visit_length = orig_visit_length.copy()
                            zero_pos = diagnosis_code
                            empty_replace[0][visit_index][zero_pos] = 0
                            empty_replace_output = self.target_model(empty_replace, empty_time, empty_visit_length).squeeze(1)

                            code_saliency = orig_example_output - empty_replace_output
                            if(predict_label[index] == 0):
                                code_saliency = -code_saliency
                            visit_each_pos.append(code_saliency.item())
                            saliency_set[count] = code_saliency
                            count = count+1
                    visit_all_pos.append(visit_each_pos)

                visit_element = len(x[index])
                if visit_element < max_visit_num:
                    visit_contri = torch.zeros(visit_element).cuda().detach()
                else:
                    visit_contri = torch.zeros(max_visit_num).cuda().detach()

                for visit_index, visit_result in enumerate(x[index]):
                    if visit_index >= max_visit_num :
                        break

                    empty_replace = orig_example.clone().detach()
                    empty_time = orig_time.clone().detach()
                    empty_visit_length = orig_visit_length.copy()

                    empty_replace[0][visit_index+1:] = 0
                    empty_time[0][visit_index+1:] = empty_time[0][visit_index]
                    empty_visit_length = [visit_index+1]
                    empty_replace_o1 = self.target_model(empty_replace, empty_time, empty_visit_length).squeeze(1)

                    empty_replace[0][visit_index] = 0
                    if visit_index > 0:
                        empty_time[0][visit_index:] = empty_time[0][visit_index - 1]
                    empty_visit_length = [visit_index]
                    empty_replace_o2 = self.target_model(empty_replace, empty_time, empty_visit_length).squeeze(1)
                    visit_contri[visit_index] = empty_replace_o1 - empty_replace_o2

                if(predict_label[index] == 0):
                    visit_contri = -visit_contri

                env = gym.make('attack_htnet:htnet-v0', ehr_input = copy_test, gt = predict_label[index], original_score = orig_example_output, attack_model = self.target_model, threshold = 0.45, pos_mat = position_mat, input_time = copy_time, input_visit_length = copy_visit_length)

                env.seed(242) 
                ob = env.reset()

                class Pos_Policy(nn.Module):
                    def __init__(self, visit_dist):
                        super(Pos_Policy, self).__init__()
                        prior_dist = ((visit_dist-visit_dist.min())/(visit_dist.max()-visit_dist.min()+torch.finfo(torch.float32).eps)).detach()
                        self.visit_dist_param = torch.nn.Parameter(prior_dist)
                        self.saved_log_probs_pos = []
                        self.pos_rewards = []

                    def forward(self):
                        return self.visit_dist_param


                class Sub_Policy(nn.Module):
                    def __init__(self, prior_dist):
                        super(Sub_Policy, self).__init__()
                        code_dist = torch.tensor(prior_dist)
                        if len(prior_dist) > 1:
                            prior_dist = ((code_dist-code_dist.min())/(code_dist.max()-code_dist.min()+torch.finfo(torch.float32).eps)).detach()
                        else:
                            prior_dist = torch.ones(1)
                        self.code_dist_param = torch.nn.Parameter(prior_dist)
                        self.saved_log_probs = []
                        self.rewards = []

                    def forward(self):
                        return self.code_dist_param

                pos_policy = Pos_Policy(visit_contri).to(self.device)
                pos_optimizer = Adam(pos_policy.parameters(), lr=1e-3)
                eps = np.finfo(np.float32).eps.item() 

                for i in range(len(visit_all_pos)):
                    setattr(self, "sub_policy_%d"%i, Sub_Policy(visit_all_pos[i]))
                    setattr(self, "optimizer%d"%i, Adam(getattr(self,"sub_policy_%d"%i).parameters(), lr = 1e-3))



                def select_action(record_element):
                    code_dist_probs = pos_policy()
                    
                    code_dist_probs = torch.nn.functional.softmax(input = code_dist_probs, dim = -1)

                    code_dist_m = Categorical(code_dist_probs)
   
                    action1 = torch.zeros(5)
   
                    for i in range(5):
                        action1_tem = code_dist_m.sample()
                        action1[i] = action1_tem
                        code_log_prob = code_dist_m.log_prob(action1_tem)
                        pos_policy.saved_log_probs_pos.append(code_log_prob)
                       
                    return action1


                def pos_finish_episode():
                    gamma = 0.99
                    R = 0
                    pos_policy_loss = []
                    pos_returns = []
                    for r in pos_policy.pos_rewards[::-1]:
                        R = r + gamma * R
                        pos_returns.insert(0, R)
                    pos_returns = torch.tensor(pos_returns)
                    pos_returns = (pos_returns - pos_returns.mean()) / (pos_returns.std() + eps)
                    for log_prob, R in zip(pos_policy.saved_log_probs_pos, pos_returns):
                        pos_policy_loss.append(-log_prob * R)
                    pos_optimizer.zero_grad()

                    for i in range(len(pos_policy_loss)):
                        if i == 0:
                            pos_policy_loss_sum = pos_policy_loss[0]
                        else:
                            pos_policy_loss_sum = pos_policy_loss_sum + pos_policy_loss[i]

                    pos_policy_loss_sum.backward()
                    pos_optimizer.step()
                    del pos_policy.pos_rewards[:]
                    del pos_policy.saved_log_probs_pos[:]


                def select_rep_action(pos):
                    subpolicy = getattr(self, "sub_policy_%d"%pos)
                    rep_dist_probs = subpolicy()
                    rep_dist_probs = torch.nn.functional.softmax(input = rep_dist_probs, dim = -1)
                    rep_dist_m = Categorical(rep_dist_probs)
                    rep_action = rep_dist_m.sample()
                    rep_log_prob = rep_dist_m.log_prob(rep_action)
                    subpolicy.saved_log_probs.append(rep_log_prob)
                    return rep_action


                def rep_update(pos):
                    gamma = 0.99
                    R = 0
                    rep_returns = []
                    
                    rep_policy = getattr(self, "sub_policy_%d"%pos) 
                     
                    R = torch.tensor(rep_policy.rewards[0])

                    log_prob = rep_policy.saved_log_probs[0]
                    rep_policy_loss = -log_prob*R
                    rep_optimizer = getattr(self, "optimizer%d"%pos) 

                    rep_optimizer.zero_grad()

                    rep_policy_loss.backward()
                    rep_optimizer.step()
                    del rep_policy.rewards[:]
                    del rep_policy.saved_log_probs[:]


                running_reward = 1
                for i_episode in range(500):
                    state, ep_reward = env.reset().detach().clone(), 0
                    code_pos = select_action(visit_element)
                    done = False
                    for t in range(5):  
                        pos_action_t = code_pos[t].int().item()
                        row_pos = pos_action_t
                        col_pos = select_rep_action(pos_action_t).item()

                        orig_code = record[row_pos][col_pos]
                        replace_set = self.catg_codes[self.code_catg_pair[orig_code]]
                        delta_p = torch.zeros(10).cuda()
                        
                        state[0][row_pos][orig_code] = 0 
                        replace_index = 0
                        for replace_code in replace_set:
                            if(replace_index < 10):

                                state[0][row_pos][replace_code] = 1
                                modified_output = self.target_model(state,copy_time,copy_visit_length).squeeze(1)
                                delta_p[replace_index] = orig_example_output - modified_output
                                if(predict_label[index] == 0):
                                    delta_p[replace_index] =  -delta_p[replace_index]  
                                state[0][row_pos][replace_code] = 0
                                replace_index = replace_index + 1
                                

                        max_element_pos = torch.argmax(delta_p[:len(replace_set)])
                        best_substition_word = replace_set[max_element_pos]
                        
                        state[0][row_pos][best_substition_word] = 1

                        state, reward, done, _ = env.step(state)
                        pos_policy.pos_rewards.append(reward)
                        ep_reward += reward
                        getattr(self, "sub_policy_%d"%pos_action_t).rewards.append(reward)
                        rep_update(pos_action_t)

                        if done:
                            state_output = self.target_model(state, copy_time, copy_visit_length).squeeze(1)
                            state_label = (state_output > 0.45).float().to(self.device)
                            if(int(state_label.item()) != int(ground_truth[index].item())):
                                print("the label is changed")

                            attack_success = attack_success + 1 
                            print("attack patient index:{} successfully".format(patient_index))
                            str1 = "attack patient index:{} successfully".format(patient_index)

                            f = open(self.output_dir,'a+')
                            f.write(str1+'\n')
                            f.close()

                            break

                    if done:
                        break
                    running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
                    pos_finish_episode()

                    if i_episode % 100 == 0:
                        print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(i_episode, ep_reward, running_reward))

            total_attack_success = attack_success + total_attack_success
            first_patient_index = first_patient_index + len(labels)

        print("total attack success:{}".format(total_attack_success))
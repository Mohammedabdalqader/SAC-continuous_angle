import os
import time
import random
import copy
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict, namedtuple, deque
from torch.autograd import Variable
from utils import CrossEntropyLoss2d
from models import  reinforcement_net
from scipy import ndimage
import matplotlib.pyplot as plt
from sac import Actor,Critic




gamma = 0.5               # discount factor
TAU = 0.01                # for soft update of target parameters
LR_ACTOR = 1e-4
LR_CRITIC = 3e-4
WEIGHT_DECAY = 2e-5       # L2 weight decay
seed = 2020
device = "cuda"
action_size = 1
action_high = 1.0
action_low = -1.0

class Trainer(object):
    def __init__(self, method, push_rewards, future_reward_discount,
                 is_testing, load_snapshot, snapshot_folder, force_cpu):
        self.explore_prob = 0.5
        self.method = method
        self.TAU = TAU
        self.is_testing = is_testing
        self.push_update = 0
        self.grasp_update = 0


        ## SAC parameters
        self.target_entropy = -action_size  # -dim(A)

        self.alpha = 0.2
        '''
        self.push_log_alpha = torch.tensor([0.0], requires_grad=True)
        self.push_alpha_optimizer = optim.SGD(params=[self.push_log_alpha], lr=LR_ACTOR) 

        self.grasp_log_alpha = torch.tensor([0.0], requires_grad=True)
        self.grasp_alpha_optimizer = optim.SGD(params=[self.grasp_log_alpha], lr=LR_ACTOR) 

        self._action_prior = "uniform"
        '''


        # Check if CUDA can be used
        if torch.cuda.is_available() and not force_cpu:
            print("CUDA detected. Running with GPU acceleration.")
            self.use_cuda = True
        elif force_cpu:
            print("CUDA detected, but overriding with option '--cpu'. Running with only CPU.")
            self.use_cuda = False
        else:
            print("CUDA is *NOT* detected. Running with only CPU.")
            self.use_cuda = False

      

        # Fully convolutional Q network for deep reinforcement learning
        if self.method == 'reinforcement': 
            self.action_size = action_size  



            self.model = reinforcement_net(self.use_cuda)
            self.push_rewards = push_rewards
            self.future_reward_discount = future_reward_discount


            # Actor Network (PUSH)
            self.push_actor = Actor().cuda()


            # push actor optimizer
            self.push_actor_optimizer = optim.SGD(self.push_actor.parameters(), lr=LR_ACTOR,momentum=0.9, weight_decay=0.0)     


            # push critic ntworks
            self.push_critic1_target = Critic().cuda()
            self.push_critic1_target.load_state_dict(self.model.push_critic1.state_dict())

            
            self.push_critic2_target = Critic().cuda()
            self.push_critic2_target.load_state_dict(self.model.push_critic2.state_dict())


            #######################################################################################################################

            # Actor Network (GRASP)
            self.grasp_actor = Actor().cuda()
 
            
            # grasp actor optimizer
            self.grasp_actor_optimizer = optim.SGD(self.grasp_actor.parameters(), lr=LR_ACTOR,momentum=0.9,weight_decay=0.0)

            # grasp critic ntworks
            self.grasp_critic1_target = Critic().to(device)
            self.grasp_critic1_target.load_state_dict(self.model.grasp_critic1.state_dict())


            self.grasp_critic2_target = Critic().to(device)
            self.grasp_critic2_target.load_state_dict(self.model.grasp_critic2.state_dict())

            

            # Initialize Huber loss
            self.criterion = torch.nn.SmoothL1Loss(reduction='none')  # Huber loss
            if self.use_cuda:
                self.criterion = self.criterion.cuda()
            
  
        
        # load pretrained models for testing
      
        if load_snapshot:
      


            self.model.load_state_dict(torch.load(os.path.join(snapshot_folder,"snapshot-10-obj.reinforcement.pth")))

            
            
            # load actor-critic models (PUSH)
            self.push_actor.load_state_dict(torch.load(os.path.join(snapshot_folder,"snapshot-10-obj.push_actor.pth")))
            self.push_critic1_target.load_state_dict(torch.load(os.path.join(snapshot_folder,"snapshot-10-obj.push_critic1_target.pth")))
            self.push_critic2_target.load_state_dict(torch.load(os.path.join(snapshot_folder,"snapshot-10-obj.push_critic2_target.pth")))

            
            # load actor-critic models (GRASP)
            self.grasp_actor.load_state_dict(torch.load(os.path.join(snapshot_folder,"snapshot-10-obj.grasp_actor.pth")))
            self.grasp_critic1_target.load_state_dict(torch.load(os.path.join(snapshot_folder,"snapshot-10-obj.grasp_critic1_target.pth")))
            self.grasp_critic2_target.load_state_dict(torch.load(os.path.join(snapshot_folder,"snapshot-10-obj.grasp_critic2_target.pth")))
            
            
         
            

        # Convert model from CPU to GPU
        if self.use_cuda:
            self.model = self.model.cuda()


        # Set model to training mode
        self.model.train()

        # Initialize optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=LR_CRITIC,momentum=0.9,weight_decay=WEIGHT_DECAY)
        self.iteration = 0

 
        # Initialize lists to save execution info and RL variables
        self.executed_action_log = []
        self.best_pixel_location_log = []
        self.label_value_log = []
        self.reward_value_log = []
        self.q_target_log = []
        self.predicted_value_log = []
        self.average_score_log = []
        self.use_heuristic_log = []
        self.is_exploit_log = []
        self.clearance_log = []
        self.grasp_alpha_log = []
        self.push_alpha_log = []
        self.q_functions_log = []





    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    # Pre-load execution info and RL variables
    def preload(self, transitions_directory):
        self.executed_action_log = np.loadtxt(os.path.join(transitions_directory, 'executed-action.log.txt'), delimiter=' ')
        self.iteration = self.executed_action_log.shape[0] - 2
        self.executed_action_log = self.executed_action_log[0:self.iteration,:]
        self.executed_action_log = self.executed_action_log.tolist()
        self.label_value_log = np.loadtxt(os.path.join(transitions_directory, 'label-value.log.txt'), delimiter=' ')
        self.label_value_log = self.label_value_log[0:self.iteration]
        self.label_value_log.shape = (self.iteration,1)
        self.label_value_log = self.label_value_log.tolist()
        self.predicted_value_log = np.loadtxt(os.path.join(transitions_directory, 'predicted-value.log.txt'), delimiter=' ')
        self.predicted_value_log = self.predicted_value_log[0:self.iteration]
        self.predicted_value_log.shape = (self.iteration,1)
        self.predicted_value_log = self.predicted_value_log.tolist()
        self.reward_value_log = np.loadtxt(os.path.join(transitions_directory, 'reward-value.log.txt'), delimiter=' ')
        self.reward_value_log = self.reward_value_log[0:self.iteration]
        self.reward_value_log.shape = (self.iteration,1)
        self.reward_value_log = self.reward_value_log.tolist()
        self.use_heuristic_log = np.loadtxt(os.path.join(transitions_directory, 'use-heuristic.log.txt'), delimiter=' ')
        self.use_heuristic_log = self.use_heuristic_log[0:self.iteration]
        self.use_heuristic_log.shape = (self.iteration,1)
        self.use_heuristic_log = self.use_heuristic_log.tolist()
        self.is_exploit_log = np.loadtxt(os.path.join(transitions_directory, 'is-exploit.log.txt'), delimiter=' ')
        self.is_exploit_log = self.is_exploit_log[0:self.iteration]
        self.is_exploit_log.shape = (self.iteration,1)
        self.is_exploit_log = self.is_exploit_log.tolist()
        self.clearance_log = np.loadtxt(os.path.join(transitions_directory, 'clearance.log.txt'), delimiter=' ')
        self.clearance_log.shape = (self.clearance_log.shape[0],1)
        self.clearance_log = self.clearance_log.tolist()


    # Compute forward pass through model to compute affordances/Q
    def forward(self, color_heightmap, depth_heightmap,push_angles=None,grasp_angles=None, is_volatile=False,use_target_model=False, specific_rotation=-1):


        # Apply 2x scale to input heightmaps
        color_heightmap_2x = ndimage.zoom(color_heightmap, zoom=[2,2,1], order=0)
        depth_heightmap_2x = ndimage.zoom(depth_heightmap, zoom=[2,2], order=0)
        assert(color_heightmap_2x.shape[0:2] == depth_heightmap_2x.shape[0:2])

        # Add extra padding (to handle rotations inside network)
        diag_length = float(color_heightmap_2x.shape[0]) * np.sqrt(2)
        diag_length = np.ceil(diag_length/32)*32
        padding_width = int((diag_length - color_heightmap_2x.shape[0])/2)
        color_heightmap_2x_r = np.pad(color_heightmap_2x[:,:,0], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_r.shape = (color_heightmap_2x_r.shape[0], color_heightmap_2x_r.shape[1], 1)
        color_heightmap_2x_g = np.pad(color_heightmap_2x[:,:,1], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_g.shape = (color_heightmap_2x_g.shape[0], color_heightmap_2x_g.shape[1], 1)
        color_heightmap_2x_b = np.pad(color_heightmap_2x[:,:,2], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_b.shape = (color_heightmap_2x_b.shape[0], color_heightmap_2x_b.shape[1], 1)
        color_heightmap_2x = np.concatenate((color_heightmap_2x_r, color_heightmap_2x_g, color_heightmap_2x_b), axis=2)
        depth_heightmap_2x = np.pad(depth_heightmap_2x, padding_width, 'constant', constant_values=0)


        # Pre-process color image (scale and normalize)
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        input_color_image = color_heightmap_2x.astype(float)/255
        for c in range(3):
            input_color_image[:, :, c] = (input_color_image[:, :, c] - image_mean[c])/image_std[c]

        # Pre-process depth image (normalize)
        image_mean = 0.01
        image_std = 0.03
        depth_heightmap_2x.shape = (depth_heightmap_2x.shape[0], depth_heightmap_2x.shape[1], 1)
        input_depth_image = (depth_heightmap_2x - image_mean) / image_std



        # Construct minibatch of size 1 (b,c,h,w)
        input_color_image.shape = (input_color_image.shape[0], input_color_image.shape[1], input_color_image.shape[2], 1)
        input_depth_image.shape = (input_depth_image.shape[0], input_depth_image.shape[1], input_depth_image.shape[2], 1)

        input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(3, 2, 0, 1)
        input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(3, 2, 0, 1)


        output_prob = self.model.forward(input_color_data, input_depth_data, self.push_critic1_target, self.push_critic2_target, self.push_actor, self.grasp_critic1_target, self.grasp_critic2_target, self.grasp_actor, push_angles, grasp_angles, is_volatile, use_target_model, specific_rotation)

 
 
      

        if self.method == 'reinforcement':


            # Return Q values (and remove extra padding)
            for rotate_idx in range(len(output_prob)):
                if rotate_idx == 0:
                    push_predictions_Q1 = output_prob[rotate_idx][0].cpu().detach().numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                    push_predictions_Q2 = output_prob[rotate_idx][1].cpu().detach().numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                    push_angles_full = output_prob[rotate_idx][2].cpu().detach().numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                    log_pis_next_push = output_prob[rotate_idx][4].cpu().detach().numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]


                    grasp_predictions_Q1 = output_prob[rotate_idx][5].cpu().detach().numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                    grasp_predictions_Q2 = output_prob[rotate_idx][6].cpu().detach().numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                    grasp_angles_full = output_prob[rotate_idx][7].cpu().detach().numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                    log_pis_next_grasp = output_prob[rotate_idx][9].cpu().detach().numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]


                    if is_volatile:
                        push_angles = output_prob[rotate_idx][3].cpu().data.numpy()
                        grasp_angles = output_prob[rotate_idx][8].cpu().data.numpy()
                    else:
                        push_angles = output_prob[rotate_idx][3]
                        grasp_angles = output_prob[rotate_idx][8]


                else:
                    push_predictions_Q1 = np.concatenate((push_predictions_Q1, output_prob[rotate_idx][0].cpu().detach().numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
                    push_predictions_Q2 = np.concatenate((push_predictions_Q2, output_prob[rotate_idx][1].cpu().detach().numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
                    push_angles_full = np.concatenate((push_angles_full, output_prob[rotate_idx][2].cpu().detach().numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
                    log_pis_next_push = np.concatenate((log_pis_next_push, output_prob[rotate_idx][4].cpu().detach().numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)

                    grasp_predictions_Q1 = np.concatenate((grasp_predictions_Q1, output_prob[rotate_idx][5].cpu().detach().numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
                    grasp_predictions_Q2 = np.concatenate((grasp_predictions_Q2, output_prob[rotate_idx][6].cpu().detach().numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
                    grasp_angles_full = np.concatenate((grasp_angles_full, output_prob[rotate_idx][7].cpu().detach().numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
                    log_pis_next_grasp = np.concatenate((log_pis_next_grasp, output_prob[rotate_idx][9].cpu().detach().numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)


                    if is_volatile:
                        push_angles = np.concatenate((push_angles, output_prob[rotate_idx][3].cpu().detach().numpy()), axis=0)
                        grasp_angles = np.concatenate((grasp_angles, output_prob[rotate_idx][8].cpu().detach().numpy()), axis=0)
                    else:
                        push_angles = np.concatenate((push_angles, output_prob[rotate_idx][3]), axis=0)
                        grasp_angles = np.concatenate((grasp_angles, output_prob[rotate_idx][8]), axis=0)

              

        return push_predictions_Q1, push_predictions_Q2 , push_angles_full,push_angles, log_pis_next_push, grasp_predictions_Q1,grasp_predictions_Q2, grasp_angles_full, grasp_angles,log_pis_next_grasp




    def get_label_value(self, primitive_action, push_success, grasp_success, change_detected, next_color_heightmap, next_depth_heightmap):

        if self.method == 'reinforcement':

            # Compute current reward
            current_reward = 0.0
            grasp_failed =False
            push_faild = False
            if primitive_action == 'push':
                if change_detected:
                    current_reward = 0.5
                else:
                    push_faild = True
            elif primitive_action == 'grasp':
                if grasp_success:
                    current_reward = 1.0
                else:
                    grasp_failed = True

            # Compute future reward
            if not change_detected and not grasp_success:
                future_reward = 0
            else:
                next_push_predictions_Q1, next_push_predictions_Q2 , push_angles_full,push_angles, log_pis_next_push, next_grasp_predictions_Q1,next_grasp_predictions_Q2, grasp_angles_full, grasp_angles,log_pis_next_grasp = self.forward(next_color_heightmap, next_depth_heightmap, is_volatile=True,use_target_model=True)

        



                Q1_target_push = torch.from_numpy(next_push_predictions_Q1)- self.alpha * torch.from_numpy(log_pis_next_push)
                Q2_target_push = torch.from_numpy(next_push_predictions_Q2) - self.alpha * torch.from_numpy(log_pis_next_push)
                future_reward_push = torch.min(torch.max(Q1_target_push),torch.max(Q2_target_push))

                Q1_target_grasp = torch.from_numpy(next_grasp_predictions_Q1) - self.alpha * torch.from_numpy(log_pis_next_grasp)
                Q2_target_grasp = torch.from_numpy(next_grasp_predictions_Q2) - self.alpha * torch.from_numpy(log_pis_next_grasp)
                future_reward_grasp = torch.min(torch.max(Q1_target_grasp),torch.max(Q2_target_grasp))


                future_reward = torch.max(future_reward_push,future_reward_grasp).detach()


                # # Experiment: use Q differences
                # push_predictions_difference = next_push_predictions - prev_push_predictions
                # grasp_predictions_difference = next_grasp_predictions - prev_grasp_predictions
                # future_reward = max(np.max(push_predictions_difference), np.max(grasp_predictions_difference))

            print('Current reward: %f' % (current_reward))
            print('Future reward: %f' % (future_reward))
            if primitive_action == 'push' and not self.push_rewards:
                expected_reward = self.future_reward_discount * future_reward
                print('Expected reward: %f + %f x %f = %f' % (0.0, self.future_reward_discount, future_reward, expected_reward))
            else:
                expected_reward = current_reward + (gamma * future_reward)
                print('Expected reward: %f + %f x %f = %f' % (current_reward, gamma, future_reward, expected_reward))
            return expected_reward, current_reward


    # Compute labels and backpropagate
    def backprop(self, color_heightmap, depth_heightmap, primitive_action, best_pix_ind, label_value,push_angles,grasp_angles,critic_type):


        if self.method == 'reinforcement':

            # Compute labels
            label = np.zeros((1,320,320))
            action_area = np.zeros((224,224))
            action_area[best_pix_ind[1]][best_pix_ind[2]] = 1
            # blur_kernel = np.ones((5,5),np.float32)/25
            # action_area = cv2.filter2D(action_area, -1, blur_kernel)
            tmp_label = np.zeros((224,224))
            tmp_label[action_area > 0] = label_value
            label[0,48:(320-48),48:(320-48)] = tmp_label

            # Compute label mask
            label_weights = np.zeros(label.shape)
            tmp_label_weights = np.zeros((224,224))
            tmp_label_weights[action_area > 0] = 1
            label_weights[0,48:(320-48),48:(320-48)] = tmp_label_weights

            # Compute loss and backward pass





            loss_v = 0
            loss_value = 0
            self.optimizer.zero_grad()
            actor_loss_value = []
            if primitive_action == 'push':

                # Do forward pass with specified rotation (to save gradients)
                push_predictions_Q1, push_predictions_Q2 , push_angles_full,push_angles, log_pis_next_push, grasp_predictions_Q1,grasp_predictions_Q2, grasp_angles_full, grasp_angles,log_pis_next_grasp = self.forward(color_heightmap, depth_heightmap,push_angles[best_pix_ind[0]],grasp_angles[best_pix_ind[0]], is_volatile=False, specific_rotation=best_pix_ind[0])


                if self.use_cuda:

                    if critic_type == 0:
                        loss = self.criterion(self.model.output_prob[0][0].view(1,320,320), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
                    elif critic_type == 1:
                        loss = self.criterion(self.model.output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)

                    loss= loss.sum()




                loss.backward(retain_graph=True)

                self.optimizer.step()


                loss_value = loss.cpu().data.numpy()

            

                # Compute alpha loss

                angles_pred, log_pis = self.push_actor.evaluate(self.model.output_prob[0][10].detach().cuda())


                push_actor_loss = 0

                if np.max(push_predictions_Q1) > np.max(push_predictions_Q2):
                    Q_push = self.model.push_critic1(self.model.output_prob[0][10].cuda(), angles_pred.cuda())
                    #Q_push = nn.Upsample(scale_factor=16, mode='bilinear').forward(Q_push)
                    actor_loss = (self.alpha * log_pis - Q_push) 
                elif np.max(push_predictions_Q1) < np.max(push_predictions_Q2):
                    Q_push = self.model.push_critic2(self.model.output_prob[0][10].cuda(), angles_pred.cuda())
                    #Q_push = nn.Upsample(scale_factor=16, mode='bilinear').forward(Q_push)

                    actor_loss = (self.alpha * log_pis - Q_push) 

                    
                push_actor_loss = actor_loss.mean()


                    
                # Minimize the loss
                self.push_actor_optimizer.zero_grad()
                push_actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.push_actor.parameters(), 1) # clip gradient to max 1
                self.push_actor_optimizer.step()

                actor_loss_value.append(push_actor_loss.cpu().data.numpy())
                self.push_update +=1
                if self.push_update % 2 == 0:
                    # ----------------------- update target networks ----------------------- #
                    self.soft_update(self.model.push_critic1, self.push_critic1_target, TAU)
                    self.soft_update(self.model.push_critic2, self.push_critic2_target, TAU)





            if primitive_action == 'grasp':

                # Do forward pass with specified rotation (to save gradients)
                push_predictions_Q1, push_predictions_Q2 , push_angles_full,push_angles, log_pis_next_push, grasp_predictions_Q1,grasp_predictions_Q2, grasp_angles_full, grasp_angles,log_pis_next_grasp = self.forward(color_heightmap, depth_heightmap,push_angles[best_pix_ind[0]],grasp_angles[best_pix_ind[0]], is_volatile=False, specific_rotation=best_pix_ind[0])


                if self.use_cuda:

                    if critic_type == 0:
                        loss = self.criterion(self.model.output_prob[0][5].view(1,320,320), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
                    elif critic_type == 1:
                        loss = self.criterion(self.model.output_prob[0][6].view(1,320,320), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)

                    loss= loss.sum()




                loss.backward(retain_graph=True)
                self.optimizer.step()


                loss_value = loss.cpu().data.numpy()

            

                # Compute alpha loss

                angles_pred, log_pis = self.grasp_actor.evaluate(self.model.output_prob[0][11].detach().cuda())


                grasp_actor_loss = 0

                if np.max(grasp_predictions_Q1) > np.max(grasp_predictions_Q2):
                    Q_grasp = self.model.grasp_critic1(self.model.output_prob[0][11].cuda(), angles_pred.cuda())
                    #Q_push = nn.Upsample(scale_factor=16, mode='bilinear').forward(Q_push)
                    actor_loss = (self.alpha * log_pis - Q_grasp) 
                elif np.max(grasp_predictions_Q1) < np.max(grasp_predictions_Q2):
                    Q_grasp = self.model.grasp_critic2(self.model.output_prob[0][11].cuda(), angles_pred.cuda())
                    #Q_push = nn.Upsample(scale_factor=16, mode='bilinear').forward(Q_push)

                    actor_loss = (self.alpha * log_pis - Q_grasp) 

                    
                grasp_actor_loss = actor_loss.mean()


                    
                # Minimize the loss
                self.grasp_actor_optimizer.zero_grad()
                grasp_actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.grasp_actor.parameters(), 1) # clip gradient to max 1
                self.grasp_actor_optimizer.step()

                actor_loss_value.append(grasp_actor_loss.cpu().data.numpy())
                self.grasp_update +=1
                if self.grasp_update % 2 == 0:
                    # ----------------------- update target networks ----------------------- #
                    self.soft_update(self.model.grasp_critic1, self.grasp_critic1_target, TAU)
                    self.soft_update(self.model.grasp_critic2, self.grasp_critic2_target, TAU)
                
                
            print('loss_value: %f' % (loss_value))
            print('actor_loss_value: ', (actor_loss_value))







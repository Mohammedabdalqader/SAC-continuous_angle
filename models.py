#!/usr/bin/env python

from collections import OrderedDict
import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
import time
from sac import Actor,Critic
from networks import FeatureTunk

torch.set_printoptions(profile="full")



class reinforcement_net(nn.Module):

    def __init__(self, use_cuda):
        super(reinforcement_net, self).__init__()
        self.use_cuda = use_cuda

        self.feature_tunk_push = FeatureTunk()
        self.feature_tunk_grasp = FeatureTunk()



        self.num_rotations = 4

        # Construct network branches for pushing and grasping


        # Critic Network (PSUH) 
        self.push_critic1 = Critic().cuda()
        self.push_critic2 = Critic().cuda()

        ###################################

        # Critic Network (GRASP) 
        self.grasp_critic1 = Critic().cuda()
        self.grasp_critic2 = Critic().cuda()






        # Initialize network weights
        for m in self.named_modules():
            if 'push-' in m[0] or 'grasp-' in m[0] or 'actor-' in m[0] or 'critic-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()

        # Initialize output variable (for backprop)
        self.interm_feat = []
        self.output_prob = []


    def forward(self, input_color_data, input_depth_data,push_critic1_target,push_critic2_target,push_actor,grasp_critic1_target,grasp_critic2_target,grasp_actor,push_angles,grasp_angles,
is_volatile=False,use_target_model=False, specific_rotation=-1):

        if is_volatile:
            with torch.no_grad():
                output_prob = []
                interm_feat = []

                log_pis_next_push = torch.zeros(1,1,20,20).cuda()
                log_pis_next_grasp = torch.zeros(1,1,20,20).cuda()

                # Apply rotations to images
                for rotate_idx in range(self.num_rotations):
                    rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))

                    # Compute sample grid for rotation BEFORE neural network
                    affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                    affine_mat_before.shape = (2,3,1)
                    affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
                    if self.use_cuda:
                        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_color_data.size())
                    else:
                        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False), input_color_data.size())

                    # Rotate images clockwise
                    if self.use_cuda:
                        rotate_color = F.grid_sample(Variable(input_color_data, volatile=True).cuda(), flow_grid_before, mode='nearest')
                        rotate_depth = F.grid_sample(Variable(input_depth_data, volatile=True).cuda(), flow_grid_before, mode='nearest')
                    else:
                        rotate_color = F.grid_sample(Variable(input_color_data, volatile=True), flow_grid_before, mode='nearest')
                        rotate_depth = F.grid_sample(Variable(input_depth_data, volatile=True), flow_grid_before, mode='nearest')

                    # Compute intermediate features
                    push_interm_feat = self.feature_tunk_push(rotate_color, rotate_depth)
                    grasp_interm_feat= self.feature_tunk_grasp(rotate_color, rotate_depth)

                    # Compute sample grid for rotation AFTER branches
                    affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],[-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                    affine_mat_after.shape = (2,3,1)
                    affine_mat_after = torch.from_numpy(affine_mat_after).permute(2,0,1).float()
                    if self.use_cuda:
                        flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), push_interm_feat.data.size())
                    else:
                        flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False), push_interm_feat.data.size())



                    if use_target_model:
                        # Get predicted next-state actions and Q values from target models
                        push_angles, log_pis_next_push,_ = push_actor.evaluate(push_interm_feat.detach())

                        

                        Q_1_push = push_critic1_target(push_interm_feat, push_angles.cuda())
                        Q_2_push = push_critic2_target(push_interm_feat, push_angles.cuda())

                
                        # grasping Target Network

                        # Get predicted next-state actions and Q values from target models
                        grasp_angles, log_pis_next_grasp,_ = grasp_actor.evaluate(grasp_interm_feat.detach())


                        Q_1_grasp = grasp_critic1_target(grasp_interm_feat, grasp_angles.cuda())
                        Q_2_grasp = grasp_critic2_target(grasp_interm_feat, grasp_angles.cuda())





                
                    else:
                        # Psuhing
                        push_angles = push_actor.get_angle(push_interm_feat.detach())

                        Q_1_push = self.push_critic1(push_interm_feat, push_angles.cuda())
                        Q_2_push = self.push_critic2(push_interm_feat, push_angles.cuda())

                        # grasping

                        grasp_angles= grasp_actor.get_angle(grasp_interm_feat.detach())


                        Q_1_grasp = self.grasp_critic1(grasp_interm_feat, grasp_angles.cuda())
                        Q_2_grasp = self.grasp_critic2(grasp_interm_feat, grasp_angles.cuda())

                    # Forward pass through branches, undo rotation on output predictions, upsample results
                    output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(Q_1_push, flow_grid_after, mode='nearest')),
                                        nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(Q_2_push, flow_grid_after, mode='nearest')),
                                        nn.Upsample(scale_factor=16, mode='nearest').forward(F.grid_sample(push_angles.cuda(), flow_grid_after, mode='nearest')),
                                        push_angles,
                                        nn.Upsample(scale_factor=16, mode='nearest').forward(F.grid_sample(log_pis_next_push.cuda(), flow_grid_after, mode='nearest')),
                                        nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(Q_1_grasp, flow_grid_after, mode='nearest')),
                                        nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(Q_2_grasp, flow_grid_after, mode='nearest')),
                                        nn.Upsample(scale_factor=16, mode='nearest').forward(F.grid_sample(grasp_angles.cuda(), flow_grid_after, mode='nearest')),
                                        grasp_angles,
                                        nn.Upsample(scale_factor=16, mode='nearest').forward(F.grid_sample(log_pis_next_grasp.cuda(), flow_grid_after, mode='nearest')),
                                        push_interm_feat,
                                        grasp_interm_feat])

            return output_prob

        else:
            self.output_prob = []
            self.interm_feat = []
            log_pis_next_push = torch.zeros(1,1,20,20).cuda()
            log_pis_next_grasp = torch.zeros(1,1,20,20).cuda()

            # Apply rotations to intermediate features
            # for rotate_idx in range(self.num_rotations):
            rotate_idx = specific_rotation
            rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))

            # Compute sample grid for rotation BEFORE branches
            affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
            affine_mat_before.shape = (2,3,1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
            if self.use_cuda:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_color_data.size())
            else:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False), input_color_data.size())

            # Rotate images clockwise
            if self.use_cuda:
                rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False).cuda(), flow_grid_before, mode='nearest')
                rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False).cuda(), flow_grid_before, mode='nearest')
            else:
                rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False), flow_grid_before, mode='nearest')
                rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False), flow_grid_before, mode='nearest')

            # Compute intermediate features
            push_interm_feat = self.feature_tunk_push(rotate_color, rotate_depth)
            grasp_interm_feat= self.feature_tunk_grasp(rotate_color, rotate_depth)

            # Compute sample grid for rotation AFTER branches
            affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],[-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat_after.shape = (2,3,1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2,0,1).float()
            if self.use_cuda:
                self.flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), push_interm_feat.data.size())
            else:
                self.flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False), push_interm_feat.data.size())



            # Psuhing
            push_angles = torch.from_numpy(push_angles).unsqueeze(0).cuda()
            Q_1_push = self.push_critic1(push_interm_feat, push_angles)
            Q_2_push = self.push_critic2(push_interm_feat, push_angles)
                
            # grasping
            grasp_angles = torch.from_numpy(grasp_angles).unsqueeze(0).cuda()
            Q_1_grasp = self.grasp_critic1(grasp_interm_feat, grasp_angles)
            Q_2_grasp = self.grasp_critic2(grasp_interm_feat, grasp_angles)

            # Forward pass through branches, undo rotation on output predictions, upsample results
            self.output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(Q_1_push, self.flow_grid_after, mode='nearest')),
                                     nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(Q_2_push, self.flow_grid_after, mode='nearest')),
                                     nn.Upsample(scale_factor=16, mode='nearest').forward(F.grid_sample(push_angles, self.flow_grid_after, mode='nearest')),
                                     push_angles,
                                     nn.Upsample(scale_factor=16, mode='nearest').forward(F.grid_sample(log_pis_next_push, self.flow_grid_after, mode='nearest')),
                                     nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(Q_1_grasp, self.flow_grid_after, mode='nearest')),
                                     nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(Q_2_grasp, self.flow_grid_after, mode='nearest')),
                                     nn.Upsample(scale_factor=16, mode='nearest').forward(F.grid_sample(grasp_angles, self.flow_grid_after, mode='nearest')),
                                     grasp_angles,
                                     nn.Upsample(scale_factor=16, mode='nearest').forward(F.grid_sample(log_pis_next_grasp, self.flow_grid_after, mode='nearest')),
                                     push_interm_feat,
                                     grasp_interm_feat])

            return self.output_prob


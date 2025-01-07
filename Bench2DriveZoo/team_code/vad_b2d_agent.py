import os
import json
import datetime
import pathlib
import time
import cv2
import carla
from collections import deque
import math
from collections import OrderedDict
from scipy.optimize import fsolve
import torch
import carla
import numpy as np
from PIL import Image
from torchvision import transforms as T
from Bench2DriveZoo.team_code.pid_controller import PIDController
from Bench2DriveZoo.team_code.planner import RoutePlanner
from leaderboard.autoagents import autonomous_agent
from mmcv import Config
from mmcv.models import build_model
from mmcv.utils import (get_dist_info, init_dist, load_checkpoint,
                        wrap_fp16_model)
from mmcv.datasets.pipelines import Compose
from mmcv.parallel.collate import collate as  mm_collate_to_batch_form
from mmcv.core.bbox import get_box_type
from pyquaternion import Quaternion
from Bench2DriveZoo.team_code.replay_buffer import LazyPrioritizedMultiStepMemory
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

SAVE_PATH = os.environ.get('SAVE_PATH', None)
IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)


def get_entry_point():
    return 'VadAgent'

def update_params(optim, loss, retain_graph=False):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    optim.step()
    
class VadAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self.steer_step = 0
        self.last_moving_status = 0
        self.last_moving_step = -1
        self.last_steer = 0
        self.pidcontroller = PIDController() 
        self.config_path = path_to_conf_file.split('+')[0]
        self.ckpt_path = path_to_conf_file.split('+')[1]
        if IS_BENCH2DRIVE:
            self.save_name = path_to_conf_file.split('+')[-1]
        else:
            self.save_name = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        cfg = Config.fromfile(self.config_path)
        if hasattr(cfg, 'plugin'):
            if cfg.plugin:
                import importlib
                if hasattr(cfg, 'plugin_dir'):
                    plugin_dir = cfg.plugin_dir
                    plugin_dir = os.path.join("Bench2DriveZoo", plugin_dir)
                    _module_dir = os.path.dirname(plugin_dir)
                    _module_dir = _module_dir.split('/')
                    _module_path = _module_dir[0]
                    for m in _module_dir[1:]:
                        _module_path = _module_path + '.' + m
                    print(_module_path)
                    plg_lib = importlib.import_module(_module_path)  
  
        self.model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

        checkpoint = load_checkpoint(self.model, self.ckpt_path, map_location='cpu', strict=True)
        
        self.model.cuda()
        self.model.eval()
        
        # self.init_rl_parameter(cfg)
        # self.init_rl_model()
        
        self.inference_only_pipeline = []
        for inference_only_pipeline in cfg.inference_only_pipeline:
            if inference_only_pipeline["type"] not in ['LoadMultiViewImageFromFilesInCeph','LoadMultiViewImageFromFiles']:
                self.inference_only_pipeline.append(inference_only_pipeline)
        self.inference_only_pipeline = Compose(self.inference_only_pipeline)

        self.takeover = False
        self.stop_time = 0
        self.takeover_time = 0
        self.save_path = None
        self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
        self.lat_ref, self.lon_ref = 42.0, 2.0

        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0	
        self.prev_control = control
        self.prev_control_cache = []
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ['ROUTES']).stem + '_'
            string += self.save_name
            self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
            self.save_path.mkdir(parents=True, exist_ok=False)
            (self.save_path / 'rgb_front').mkdir()
            (self.save_path / 'rgb_front_right').mkdir()
            (self.save_path / 'rgb_front_left').mkdir()
            (self.save_path / 'rgb_back').mkdir()
            (self.save_path / 'rgb_back_right').mkdir()
            (self.save_path / 'rgb_back_left').mkdir()
            (self.save_path / 'meta').mkdir()
            (self.save_path / 'bev').mkdir()
   
        self.lidar2img = {
        'CAM_FRONT':np.array([[ 1.14251841e+03,  8.00000000e+02,  0.00000000e+00, -9.52000000e+02],
                              [ 0.00000000e+00,  4.50000000e+02, -1.14251841e+03, -8.09704417e+02],
                              [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -1.19000000e+00],
                              [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_FRONT_LEFT':np.array([[ 6.03961325e-14,  1.39475744e+03,  0.00000000e+00, -9.20539908e+02],
                                   [-3.68618420e+02,  2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
                                   [-8.19152044e-01,  5.73576436e-01,  0.00000000e+00, -8.29094072e-01],
                                   [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_FRONT_RIGHT':np.array([[ 1.31064327e+03, -4.77035138e+02,  0.00000000e+00,-4.06010608e+02],
                                    [ 3.68618420e+02,  2.58109396e+02, -1.14251841e+03,-6.47296750e+02],
                                    [ 8.19152044e-01,  5.73576436e-01,  0.00000000e+00,-8.29094072e-01],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]),
        'CAM_BACK':np.array([[-5.60166031e+02, -8.00000000e+02,  0.00000000e+00, -1.28800000e+03],
                     [ 5.51091060e-14, -4.50000000e+02, -5.60166031e+02, -8.58939847e+02],
                     [ 1.22464680e-16, -1.00000000e+00,  0.00000000e+00, -1.61000000e+00],
                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_BACK_LEFT':np.array([[-1.14251841e+03,  8.00000000e+02,  0.00000000e+00, -6.84385123e+02],
                                  [-4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                  [-9.39692621e-01, -3.42020143e-01,  0.00000000e+00, -4.92889531e-01],
                                  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
  
        'CAM_BACK_RIGHT': np.array([[ 3.60989788e+02, -1.34723223e+03,  0.00000000e+00, -1.04238127e+02],
                                    [ 4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                    [ 9.39692621e-01, -3.42020143e-01,  0.00000000e+00, -4.92889531e-01],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        }
        self.lidar2cam = {
        'CAM_FRONT':np.array([[ 1.  ,  0.  ,  0.  ,  0.  ],
                              [ 0.  ,  0.  , -1.  , -0.24],
                              [ 0.  ,  1.  ,  0.  , -1.19],
                              [ 0.  ,  0.  ,  0.  ,  1.  ]]),
        'CAM_FRONT_LEFT':np.array([[ 0.57357644,  0.81915204,  0.  , -0.22517331],
                                   [ 0.        ,  0.        , -1.  , -0.24      ],
                                   [-0.81915204,  0.57357644,  0.  , -0.82909407],
                                   [ 0.        ,  0.        ,  0.  ,  1.        ]]),
        'CAM_FRONT_RIGHT':np.array([[ 0.57357644, -0.81915204, 0.  ,  0.22517331],
                                   [ 0.        ,  0.        , -1.  , -0.24      ],
                                   [ 0.81915204,  0.57357644,  0.  , -0.82909407],
                                   [ 0.        ,  0.        ,  0.  ,  1.        ]]),
        'CAM_BACK':np.array([[-1. ,  0.,  0.,  0.  ],
                             [ 0. ,  0., -1., -0.24],
                             [ 0. , -1.,  0., -1.61],
                             [ 0. ,  0.,  0.,  1.  ]]),
     
        'CAM_BACK_LEFT':np.array([[-0.34202014,  0.93969262,  0.  , -0.25388956],
                                  [ 0.        ,  0.        , -1.  , -0.24      ],
                                  [-0.93969262, -0.34202014,  0.  , -0.49288953],
                                  [ 0.        ,  0.        ,  0.  ,  1.        ]]),
  
        'CAM_BACK_RIGHT':np.array([[-0.34202014, -0.93969262,  0.  ,  0.25388956],
                                  [ 0.        ,  0.         , -1.  , -0.24      ],
                                  [ 0.93969262, -0.34202014 ,  0.  , -0.49288953],
                                  [ 0.        ,  0.         ,  0.  ,  1.        ]])
        }
        self.lidar2ego = np.array([[ 0. ,  1. ,  0. , -0.39],
                                   [-1. ,  0. ,  0. ,  0.  ],
                                   [ 0. ,  0. ,  1. ,  1.84],
                                   [ 0. ,  0. ,  0. ,  1.  ]])
        
        topdown_extrinsics =  np.array([[0.0, -0.0, -1.0, 50.0], [0.0, 1.0, -0.0, 0.0], [1.0, -0.0, 0.0, -0.0], [0.0, 0.0, 0.0, 1.0]])
        unreal2cam = np.array([[0,1,0,0], [0,0,-1,0], [1,0,0,0], [0,0,0,1]])
        self.coor2topdown = unreal2cam @ topdown_extrinsics
        topdown_intrinsics = np.array([[548.993771650447, 0.0, 256.0, 0], [0.0, 548.993771650447, 256.0, 0], [0.0, 0.0, 1.0, 0], [0, 0, 0, 1.0]])
        self.coor2topdown = topdown_intrinsics @ self.coor2topdown

    def init_rl_parameter(self,cfg):

        self.qf1_optimizer = None
        self.qf2_optimizer = None
        self.vf_optimizer = None
        self.policy_optimizer = None
        self.alpha_optimizer = None
        self.alpha = None
        self.memory_size = cfg.memory_size
        self.plan_anchors = np.load(cfg.plan_anchors_path, allow_pickle=True)
        self.device = torch.device("cuda")
        
        self.tau = cfg.tau
        self.alpha = cfg.alpha
        self.auto_alpha = cfg.auto_alpha
        self.target_update_interval = cfg.target_update_interval
        self.learning_steps = 0
        self.log_interval = 1
        self._squash = cfg.squash
        
        if cfg.auto_alpha:
            self.log_alpha = torch.tensor(np.log(cfg.alpha), requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.target_alpha = -np.log(1.0 / cfg.action_size) * cfg.target_entropy_ratio
            self.alpha_optimizer = AdamW([self.log_alpha], lr=lr)

        if cfg.use_per:
            beta_steps = (cfg.num_steps - cfg.start_steps) / cfg.update_interval
            self.memory = LazyPrioritizedMultiStepMemory(
                capacity=cfg.memory_size,
                device=self.device, 
                gamma=cfg.gamma, 
                multi_step=cfg.multi_step,
                beta_steps=beta_steps)
        else:
            self.memory = LazyMultiStepMemory(
                capacity=cfg.memory_size,
                device=self.device, 
                gamma=cfg.gamma, 
                multi_step=cfg.multi_step)
            
        self.log_dir = cfg.log_dir
        self.model_dir = os.path.join(cfg.log_dir, 'model')

        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        timestamp_dir = 'summary_' + current_time
        self.summary_dir = os.path.join(cfg.log_dir, 'summary', timestamp_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        
    def init_rl_model(self):
        self.qf1 = build_model(cfg.qf_model,train_cfg=cfg.get('train_cfg'),test_cfg=cfg.get('test_cfg'))
        self.qf2 = build_model(cfg.qf_model,train_cfg=cfg.get('train_cfg'),test_cfg=cfg.get('test_cfg'))
        self.vf = build_model(cfg.vf_model,train_cfg=cfg.get('train_cfg'),test_cfg=cfg.get('test_cfg'))
        self.vf_target = build_model(cfg.vf_model,train_cfg=cfg.get('train_cfg'),test_cfg=cfg.get('test_cfg'))
        self.policy = build_model(cfg.policy_model,train_cfg=cfg.get('train_cfg'),test_cfg=cfg.get('test_cfg'))
        
        self.qf1.init_weights()
        self.qf2.init_weights()
        self.vf.init_weights()
        self.vf_target.init_weights()
        self.policy.init_weights()
        
        state_dict_w = self.model.state_dict()
        state_dict_qf1 = self.qf1.state_dict()
        state_dict_qf2 = self.qf2.state_dict()
        state_dict_vf = self.vf.state_dict()
        state_dict_vf_target = self.vf_target.state_dict()
        state_dict_policy = self.policy.state_dict()
        
        for name, param in state_dict_w.items():
            if name in state_dict_qf1:
                state_dict_qf1[name].copy_(param)
            if name in state_dict_qf2:
                state_dict_qf2[name].copy_(param)
            if name in state_dict_vf:
                state_dict_vf[name].copy_(param)
            if name in state_dict_vf_target:
                state_dict_vf_target[name].copy_(param)
            if name in state_dict_policy:
                state_dict_policy[name].copy_(param)
                    
        self.qf1.load_state_dict(state_dict_qf1)
        self.qf2.load_state_dict(state_dict_qf2)
        self.vf.load_state_dict(state_dict_vf)
        self.vf_target.load_state_dict(state_dict_vf_target)
        self.policy.load_state_dict(state_dict_policy)
        
        for name, param in self.qf1.named_parameters():
            if name in state_dict_w:
                param.requires_grad = False
        for name, param in self.qf2.named_parameters():
            if name in state_dict_w:
                param.requires_grad = False
        for name, param in self.vf.named_parameters():
            if name in state_dict_w:
                param.requires_grad = False
        for name, param in self.policy.named_parameters():
            if name in state_dict_w:
                param.requires_grad = False
                
        self.vf_target.load_state_dict(self.vf.state_dict())
        
        self.qf1_optimizer = AdamW(self.qf1.parameters(), lr=lr)
        self.qf2_optimizer = AdamW(self.qf2.parameters(), lr=lr)
        self.vf_optimizer = AdamW(self.vf.parameters(), lr=lr)
        self.policy_optimizer = AdamW(self.policy.parameters(), lr=lr)
        
        self.qf1.cuda()
        self.qf2.cuda()
        self.vf.cuda()
        self.vf_target.cuda()
        self.policy.cuda()
        
        self.qf1.eval()
        self.qf2.eval()
        self.vf.eval()
        self.vf_target.eval()
        self.policy.eval()
        
    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1 = self.qf1(states, actions)
        curr_q2 = self.qf2(states, actions)

        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_values = self.vf_target(next_states)
            target_q = rewards + (1 - dones) * self.alpha * next_values
        return target_q
    
    def calc_critic_loss(self, batch, weights):
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)

        target_q_obs = target_q.detach().mean().item()

        errors = torch.abs(curr_q1.detach() - target_q)
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()
        
        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)
        
        return q1_loss, q2_loss, errors, mean_q1, mean_q2
        
    def calc_critic_q_loss(self, batch, weights):
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)
        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)
        return q1_loss, q2_loss
    
    def calc_critic_v_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch
        with torch.no_grad():
            outs = self.policy.exploit(states)
            outputs_ego_trajs, index, action_probs, entropy, log_pis = self.process_action(outs)
            
            actions_new = outputs_ego_trajs[index]
            q1_new = self.qf1(states, actions_new)
            q2_new = self.qf2(states, actions_new)
            min_q = torch.min(q1_new, q2_new)
            target_v = min_q - self.alpha * log_pis
        v = self.vf(states)
        v_loss = torch.mean((v - target_v).pow(2) * weights)
        td_error = (v - target_v).detach().item()
        return v_loss, td_error
    
    def calc_policy_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch
        output_data_batch, prev_frame_info = self.policy.exploit(states)
        
        outputs_ego_trajs, index, action_probs, entropy, log_pis = self.process_action(output_data_batch)
        
        actions_new = outputs_ego_trajs[index]
        q1_new = self.qf1(states, actions_new)
        q2_new = self.qf2(states, actions_new)
        min_q = torch.min(q1_new, q2_new)
        actor_loss = (weights *(self.alpha * log_pis - min_q)).mean()

        return actor_loss, log_pis
    
    def calc_entropy_loss(self, log_pis, weights):
        assert not log_pis.requires_grad
        if self.auto_alpha:
            alpha_loss = -tf.reduce_mean((self.alpha * tf.stop_gradient(log_pis + self.target_alpha))* weights)
        return alpha_loss
         
    def process_action(self, outs, test=False):
        outputs_ego_cls_expert = outs[0]['pts_bbox']['ego_cls_expert_preds']
        outputs_ego_trajs = outs[0]['pts_bbox']['ego_fut_preds']
        
        action_probs = F.softmax(outputs_ego_cls_expert, dim=1)
        action_dist = Categorical(action_probs)
        
        if test:
            raw_actions = torch.argmax(outputs_ego_cls_expert, dim=1, keepdim=True)
        else:
            raw_actions = action_dist.sample()
        entropy = action_dist.entropy()
        log_pis = action_dist.log_prob(raw_actions)
        
        if self._squash:
            actions = tf.tanh(raw_actions)
            diff = tf.reduce_sum(tf.math.log(1 - actions ** 2 + self.EPS), axis=1)
            log_pis -= diff
        else:
            actions = raw_actions
        return outputs_ego_trajs, actions, action_probs, entropy, log_pis
    
    def learn(self):
        assert hasattr(self, 'qf1_optimizer,') and hasattr(self, 'qf2_optimizer') and\
            hasattr(self, 'policy_optimizer') and hasattr(self, 'alpha_optimizer') and\
            hasattr(self, 'vf_optimizer')

        self.learning_steps += 1

        if self.use_per:
            batch, weights = self.memory.sample(self.batch_size)
        else:
            batch = self.memory.sample(self.batch_size)
            # Set priority weights to 1 when we don't use PER.
            weights = 1.
            
        q1_loss, q2_loss = self.calc_critic_q_loss(batch, weights)
        v_loss,td_error = self.calc_critic_v_loss(batch, weights)
        actor_loss, log_pis = self.calc_policy_loss(batch, weights)
        alpha_loss = self.calc_entropy_loss(log_pis, weights)
        
        update_params(self.qf1_optimizer, q1_loss)
        update_params(self.qf2_optimizer, q2_loss)
        update_params(self.vf_optimizer, v_loss)
        update_params(self.policy_optimizer, actor_loss)
        update_params(self.alpha_optimizer, alpha_loss)

        if self.use_per:
            self.memory.update_priority(td_error)

        if self.learning_steps % self.log_interval == 0:

            self.writer.add_scalar(
                'loss/Q1', q1_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/Q2', q2_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/V', v_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/policy', actor_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/alpha', alpha_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'stats/alpha', self.alpha.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'stats/entropy', log_pis.detach().mean().item(),
                self.learning_steps)
            
    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        torch.save(self.policy.state_dict(), os.path.join(save_dir, current_time + 'policy.pth'))
        torch.save(self.qf1.state_dict(), os.path.join(save_dir, current_time + 'qf1.pth'))
        torch.save(self.qf2.state_dict(), os.path.join(save_dir, current_time + 'qf2.pth'))
        torch.save(self.vf.state_dict(), os.path.join(save_dir, current_time + 'vf.pth'))
        
    def update_target(self):
        with torch.no_grad():
            for target_param, param in zip(self.vf_target.parameters(), self.vf.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
                
    def _init(self):
        try:
            locx, locy = self._global_plan_world_coord[0][0].location.x, self._global_plan_world_coord[0][0].location.y
            lon, lat = self._global_plan[0][0]['lon'], self._global_plan[0][0]['lat']
            EARTH_RADIUS_EQUA = 6378137.0
            def equations(vars):
                x, y = vars
                eq1 = lon * math.cos(x * math.pi / 180) - (locx * x * 180) / (math.pi * EARTH_RADIUS_EQUA) - math.cos(x * math.pi / 180) * y
                eq2 = math.log(math.tan((lat + 90) * math.pi / 360)) * EARTH_RADIUS_EQUA * math.cos(x * math.pi / 180) + locy - math.cos(x * math.pi / 180) * EARTH_RADIUS_EQUA * math.log(math.tan((90 + x) * math.pi / 360))
                return [eq1, eq2]
            initial_guess = [0, 0]
            solution = fsolve(equations, initial_guess)
            self.lat_ref, self.lon_ref = solution[0], solution[1]
        except Exception as e:
            print(e, flush=True)
            self.lat_ref, self.lon_ref = 0, 0      
        self._route_planner = RoutePlanner(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True
        self.metric_info = {}
  

    def sensors(self):
        sensors =[
                # camera rgb
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.80, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_LEFT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_RIGHT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -2.0, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                    'width': 1600, 'height': 900, 'fov': 110,
                    'id': 'CAM_BACK'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_LEFT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_RIGHT'
                },
                # imu
                {
                    'type': 'sensor.other.imu',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'IMU'
                },
                # gps
                {
                    'type': 'sensor.other.gnss',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'GPS'
                },
                # speed
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'SPEED'
                },
                # collision sensor
                {
                    'type': 'sensor.other.collision',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.1,
                    'id': 'COLLISION_SENSOR'
                },
                # lane invasion sensors
                {
                    'type': 'sensor.other.lane_invasion',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.1,
                    'id': 'LANE_INVASION_SENSOR'
                },
            ]
        if IS_BENCH2DRIVE:
            sensors += [
                    {	
                        'type': 'sensor.camera.rgb',
                        'x': 0.0, 'y': 0.0, 'z': 50.0,
                        'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                        'width': 512, 'height': 512, 'fov': 5 * 10.0,
                        'id': 'bev'
                    }]
        return sensors

    def tick(self, input_data):
        self.step += 1
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
        imgs = {}
        for cam in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']:
            img = cv2.cvtColor(input_data[cam][1][:, :, :3], cv2.COLOR_BGR2RGB)
            _, img = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            imgs[cam] = img
        bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data['GPS'][1][:2]
        speed = input_data['SPEED'][1]['speed']
        compass = input_data['IMU'][1][-1]
        acceleration = input_data['IMU'][1][:3]
        angular_velocity = input_data['IMU'][1][3:6]
  
        pos = self.gps_to_location(gps)
        near_node, near_command = self._route_planner.run_step(pos)
  
        if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
            compass = 0.0
            acceleration = np.zeros(3)
            angular_velocity = np.zeros(3)

        result = {
                'imgs': imgs,
                'gps': gps,
                'pos':pos,
                'speed': speed,
                'compass': compass,
                'bev': bev,
                'acceleration':acceleration,
                'angular_velocity':angular_velocity,
                'command_near':near_command,
                'command_near_xy':near_node
                }
        
        return result
    
    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        tick_data = self.tick(input_data)
        results = {}
        results['lidar2img'] = []
        results['lidar2cam'] = []
        results['img'] = []
        results['folder'] = ' '
        results['scene_token'] = ' '  
        results['frame_idx'] = 0
        results['timestamp'] = self.step / 20
        results['box_type_3d'], _ = get_box_type('LiDAR')
  
        for cam in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']:
            results['lidar2img'].append(self.lidar2img[cam])
            results['lidar2cam'].append(self.lidar2cam[cam])
            results['img'].append(tick_data['imgs'][cam])
        results['lidar2img'] = np.stack(results['lidar2img'],axis=0)
        results['lidar2cam'] = np.stack(results['lidar2cam'],axis=0)
        raw_theta = tick_data['compass']   if not np.isnan(tick_data['compass']) else 0
        ego_theta = -raw_theta + np.pi/2
        rotation = list(Quaternion(axis=[0, 0, 1], radians=ego_theta))
        can_bus = np.zeros(18)
        can_bus[0] = tick_data['pos'][0]
        can_bus[1] = -tick_data['pos'][1]
        can_bus[3:7] = rotation
        can_bus[7] = tick_data['speed']
        can_bus[10:13] = tick_data['acceleration']
        can_bus[11] *= -1
        can_bus[13:16] = -tick_data['angular_velocity']
        can_bus[16] = ego_theta
        can_bus[17] = ego_theta / np.pi * 180 
        results['can_bus'] = can_bus
        ego_lcf_feat = np.zeros(9)
        ego_lcf_feat[0:2] = can_bus[0:2].copy()
        ego_lcf_feat[2:4] = can_bus[10:12].copy()
        ego_lcf_feat[4] = rotation[-1]
        ego_lcf_feat[5] = 4.89238167
        ego_lcf_feat[6] = 1.83671331
        ego_lcf_feat[7] = np.sqrt(can_bus[0]**2+can_bus[1]**2)

        if len(self.prev_control_cache)<10:
            ego_lcf_feat[8] = 0
        else:
            ego_lcf_feat[8] = self.prev_control_cache[0].steer

        command = tick_data['command_near']
        if command < 0:
            command = 4
        command -= 1
        results['command'] = command
        command_onehot = np.zeros(6)
        command_onehot[command] = 1
        results['ego_fut_cmd'] = command_onehot
        theta_to_lidar = raw_theta
        command_near_xy = np.array([tick_data['command_near_xy'][0]-can_bus[0],-tick_data['command_near_xy'][1]-can_bus[1]])
        rotation_matrix = np.array([[np.cos(theta_to_lidar),-np.sin(theta_to_lidar)],[np.sin(theta_to_lidar),np.cos(theta_to_lidar)]])
        local_command_xy = rotation_matrix @ command_near_xy
  
        ego2world = np.eye(4)
        ego2world[0:3,0:3] = Quaternion(axis=[0, 0, 1], radians=ego_theta).rotation_matrix
        ego2world[0:2,3] = can_bus[0:2]
        lidar2global = ego2world @ self.lidar2ego
        results['l2g_r_mat'] = lidar2global[0:3,0:3]
        results['l2g_t'] = lidar2global[0:3,3]
        stacked_imgs = np.stack(results['img'],axis=-1)
        results['img_shape'] = stacked_imgs.shape
        results['ori_shape'] = stacked_imgs.shape
        results['pad_shape'] = stacked_imgs.shape
        results = self.inference_only_pipeline(results)
        self.device="cuda"
        input_data_batch = mm_collate_to_batch_form([results], samples_per_gpu=1)
        for key, data in input_data_batch.items():
            if key != 'img_metas':
                if torch.is_tensor(data[0]):
                    data[0] = data[0].to(self.device)
        output_data_batch, prev_frame_info = self.model(input_data_batch, return_loss=False, rescale=True)
        all_out_truck_d1 = output_data_batch[0]['pts_bbox']['ego_fut_preds'].cpu().numpy()
        all_out_truck =  np.cumsum(all_out_truck_d1,axis=1)
        out_truck = all_out_truck[command]
        steer_traj, throttle_traj, brake_traj, metadata_traj = self.pidcontroller.control_pid(out_truck, tick_data['speed'], local_command_xy)
        if brake_traj < 0.05: brake_traj = 0.0
        if throttle_traj > brake_traj: brake_traj = 0.0

        control = carla.VehicleControl()
        self.pid_metadata = metadata_traj
        self.pid_metadata['agent'] = 'only_traj'
        control.steer = np.clip(float(steer_traj), -1, 1)
        control.throttle = np.clip(float(throttle_traj), 0, 0.75)
        control.brake = np.clip(float(brake_traj), 0, 1)     
        self.pid_metadata['steer'] = control.steer
        self.pid_metadata['throttle'] = control.throttle
        self.pid_metadata['brake'] = control.brake
        self.pid_metadata['steer_traj'] = float(steer_traj)
        self.pid_metadata['throttle_traj'] = float(throttle_traj)
        self.pid_metadata['brake_traj'] = float(brake_traj)
        self.pid_metadata['plan'] = out_truck.tolist()
        self.pid_metadata['command'] = command
        self.pid_metadata['all_plan'] = all_out_truck.tolist()

        metric_info = self.get_metric_info()
        self.metric_info[self.step] = metric_info
        if SAVE_PATH is not None and self.step % 1 == 0:
            self.save(tick_data)
        self.prev_control = control
        
        if len(self.prev_control_cache)==10:
            self.prev_control_cache.pop(0)
        self.prev_control_cache.append(control)
        return control, {"prev_frame_info":prev_frame_info, "input":input_data_batch, "output":output_data_batch}

    @torch.no_grad()
    def rl_run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        tick_data = self.tick(input_data)
        results = {}
        results['lidar2img'] = []
        results['lidar2cam'] = []
        results['img'] = []
        results['folder'] = ' '
        results['scene_token'] = ' '  
        results['frame_idx'] = 0
        results['timestamp'] = self.step / 20
        results['box_type_3d'], _ = get_box_type('LiDAR')
  
        for cam in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']:
            results['lidar2img'].append(self.lidar2img[cam])
            results['lidar2cam'].append(self.lidar2cam[cam])
            results['img'].append(tick_data['imgs'][cam])
        results['lidar2img'] = np.stack(results['lidar2img'],axis=0)
        results['lidar2cam'] = np.stack(results['lidar2cam'],axis=0)
        raw_theta = tick_data['compass']   if not np.isnan(tick_data['compass']) else 0
        ego_theta = -raw_theta + np.pi/2
        rotation = list(Quaternion(axis=[0, 0, 1], radians=ego_theta))
        can_bus = np.zeros(18)
        can_bus[0] = tick_data['pos'][0]
        can_bus[1] = -tick_data['pos'][1]
        can_bus[3:7] = rotation
        can_bus[7] = tick_data['speed']
        can_bus[10:13] = tick_data['acceleration']
        can_bus[11] *= -1
        can_bus[13:16] = -tick_data['angular_velocity']
        can_bus[16] = ego_theta
        can_bus[17] = ego_theta / np.pi * 180 
        results['can_bus'] = can_bus
        ego_lcf_feat = np.zeros(9)
        ego_lcf_feat[0:2] = can_bus[0:2].copy()
        ego_lcf_feat[2:4] = can_bus[10:12].copy()
        ego_lcf_feat[4] = rotation[-1]
        ego_lcf_feat[5] = 4.89238167
        ego_lcf_feat[6] = 1.83671331
        ego_lcf_feat[7] = np.sqrt(can_bus[0]**2+can_bus[1]**2)

        if len(self.prev_control_cache)<10:
            ego_lcf_feat[8] = 0
        else:
            ego_lcf_feat[8] = self.prev_control_cache[0].steer

        command = tick_data['command_near']
        if command < 0:
            command = 4
        command -= 1
        results['command'] = command
        command_onehot = np.zeros(6)
        command_onehot[command] = 1
        results['ego_fut_cmd'] = command_onehot
        theta_to_lidar = raw_theta
        command_near_xy = np.array([tick_data['command_near_xy'][0]-can_bus[0],-tick_data['command_near_xy'][1]-can_bus[1]])
        rotation_matrix = np.array([[np.cos(theta_to_lidar),-np.sin(theta_to_lidar)],[np.sin(theta_to_lidar),np.cos(theta_to_lidar)]])
        local_command_xy = rotation_matrix @ command_near_xy
  
        ego2world = np.eye(4)
        ego2world[0:3,0:3] = Quaternion(axis=[0, 0, 1], radians=ego_theta).rotation_matrix
        ego2world[0:2,3] = can_bus[0:2]
        lidar2global = ego2world @ self.lidar2ego
        results['l2g_r_mat'] = lidar2global[0:3,0:3]
        results['l2g_t'] = lidar2global[0:3,3]
        stacked_imgs = np.stack(results['img'],axis=-1)
        results['img_shape'] = stacked_imgs.shape
        results['ori_shape'] = stacked_imgs.shape
        results['pad_shape'] = stacked_imgs.shape
        results = self.inference_only_pipeline(results)
        self.device="cuda"
        input_data_batch = mm_collate_to_batch_form([results], samples_per_gpu=1)
        for key, data in input_data_batch.items():
            if key != 'img_metas':
                if torch.is_tensor(data[0]):
                    data[0] = data[0].to(self.device)
                    
        output_data_batch, prev_frame_info = self.policy.explore(input_data_batch, rescale=True)
        
        outputs_ego_cls_expert = output_data_batch[0]['pts_bbox']['ego_cls_expert_preds'].cpu().numpy()
        outputs_ego_trajs = output_data_batch[0]['pts_bbox']['ego_fut_preds'].cpu().numpy()
        action_probs = F.softmax(outputs_ego_cls_expert, dim=1)
        action_dist = Categorical(action_probs)
        raw_actions = action_dist.sample()
        out_truck = outputs_ego_trajs[raw_actions]
        out_truck =  np.cumsum(out_truck,axis=1)

        steer_traj, throttle_traj, brake_traj, metadata_traj = self.pidcontroller.control_pid(out_truck, tick_data['speed'], local_command_xy)
        if brake_traj < 0.05: brake_traj = 0.0
        if throttle_traj > brake_traj: brake_traj = 0.0

        control = carla.VehicleControl()
        self.pid_metadata = metadata_traj
        self.pid_metadata['agent'] = 'only_traj'
        control.steer = np.clip(float(steer_traj), -1, 1)
        control.throttle = np.clip(float(throttle_traj), 0, 0.75)
        control.brake = np.clip(float(brake_traj), 0, 1)     
        self.pid_metadata['steer'] = control.steer
        self.pid_metadata['throttle'] = control.throttle
        self.pid_metadata['brake'] = control.brake
        self.pid_metadata['steer_traj'] = float(steer_traj)
        self.pid_metadata['throttle_traj'] = float(throttle_traj)
        self.pid_metadata['brake_traj'] = float(brake_traj)
        self.pid_metadata['plan'] = out_truck.tolist()
        self.pid_metadata['command'] = command
        self.pid_metadata['all_plan'] = all_out_truck.tolist()

        metric_info = self.get_metric_info()
        self.metric_info[self.step] = metric_info
        if SAVE_PATH is not None and self.step % 1 == 0:
            self.save(tick_data)
        self.prev_control = control
        
        if len(self.prev_control_cache)==10:
            self.prev_control_cache.pop(0)
        self.prev_control_cache.append(control)
        return control, {"prev_frame_info":prev_frame_info, "input":input_data_batch, "output":output_data_batch}
    
    def save(self, tick_data):
        frame = self.step // 10

        Image.fromarray(tick_data['imgs']['CAM_FRONT']).save(self.save_path / 'rgb_front' / ('%04d.png' % frame))
        Image.fromarray(tick_data['imgs']['CAM_FRONT_LEFT']).save(self.save_path / 'rgb_front_left' / ('%04d.png' % frame))
        Image.fromarray(tick_data['imgs']['CAM_FRONT_RIGHT']).save(self.save_path / 'rgb_front_right' / ('%04d.png' % frame))
        Image.fromarray(tick_data['imgs']['CAM_BACK']).save(self.save_path / 'rgb_back' / ('%04d.png' % frame))
        Image.fromarray(tick_data['imgs']['CAM_BACK_LEFT']).save(self.save_path / 'rgb_back_left' / ('%04d.png' % frame))
        Image.fromarray(tick_data['imgs']['CAM_BACK_RIGHT']).save(self.save_path / 'rgb_back_right' / ('%04d.png' % frame))
        Image.fromarray(tick_data['bev']).save(self.save_path / 'bev' / ('%04d.png' % frame))

        outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
        json.dump(self.pid_metadata, outfile, indent=4)
        outfile.close()

        # metric info
        outfile = open(self.save_path / 'metric_info.json', 'w')
        json.dump(self.metric_info, outfile, indent=4)
        outfile.close()

    def destroy(self):
        del self.model
        torch.cuda.empty_cache()

    def gps_to_location(self, gps):
        EARTH_RADIUS_EQUA = 6378137.0
        # gps content: numpy array: [lat, lon, alt]
        lat, lon = gps
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat+90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
        mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
        y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
        x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
        return np.array([x, y])

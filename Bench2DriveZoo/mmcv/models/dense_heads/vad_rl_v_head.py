import copy
from math import pi, cos, sin

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from mmcv.models.builder import HEADS, build_loss 
from mmcv.models.dense_heads import DETRHead
from mmcv.utils import force_fp32, auto_fp16
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.core.bbox.builder import build_assigner, build_sampler
from mmcv.core.bbox.coder import build_bbox_coder
from mmcv.models.utils.transformer import inverse_sigmoid
from mmcv.core.bbox.transforms import bbox_xyxy_to_cxcywh
from mmcv.models.bricks import Linear
from mmcv.models.utils import bias_init_with_prob, xavier_init
from mmcv.core.utils import (multi_apply, multi_apply, reduce_mean)
from mmcv.models.bricks.transformer import build_transformer_layer_sequence
from .anchor_free_head import AnchorFreeHead
from mmcv.core.bbox.util import normalize_bbox
from mmcv.models.vad_utils.traj_lr_warmup import get_traj_warmup_loss_weight
from mmcv.models.vad_utils.map_utils import (
    normalize_2d_pts, normalize_2d_bbox, denormalize_2d_pts, denormalize_2d_bbox
)
from mmcv.models.bricks.transformer import FFN, build_positional_encoding
from mmcv.models.utils import build_transformer
from mmcv.models.utils.functional import pos2posemb2d
from mmcv.models import DETECTORS, build_backbone, build_head, build_neck

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_unit, verbose=False):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.mlp(x)
        return x

class LaneNet(nn.Module):
    def __init__(self, in_channels, hidden_unit, num_subgraph_layers):
        super(LaneNet, self).__init__()
        self.num_subgraph_layers = num_subgraph_layers
        self.layer_seq = nn.Sequential()
        for i in range(num_subgraph_layers):
            self.layer_seq.add_module(
                f'lmlp_{i}', MLP(in_channels, hidden_unit))
            in_channels = hidden_unit*2

    def forward(self, pts_lane_feats):
        '''
            Extract lane_feature from vectorized lane representation

        Args:
            pts_lane_feats: [batch size, max_pnum, pts, D]

        Returns:
            inst_lane_feats: [batch size, max_pnum, D]
        '''
        x = pts_lane_feats
        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, MLP):
                # x [bs,max_lane_num,9,dim]
                x = layer(x)
                x_max = torch.max(x, -2)[0]
                x_max = x_max.unsqueeze(2).repeat(1, 1, x.shape[2], 1)
                x = torch.cat([x, x_max], dim=-1)
        x_max = torch.max(x, -2)[0]
        return x_max


@HEADS.register_module()
class VADRLVHead(DETRHead):
    """Head of VAD model.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """
    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 bev_h=30,
                 bev_w=30,
                 fut_ts=6,
                 fut_mode=6,
                 tot_epoch=None,
                 use_pe=False,
                 map_thresh=0.5,
                 dis_thresh=0.2,
                 pe_normalization=True,
                 query_thresh=None,
                 query_use_fix_pad=None,
                 valid_fut_ts=6,
                 ego_bev_decoder=None,
                 interaction_pe_type = None,
                 ego_lcf_feat_idx=None,
                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.tot_epoch = tot_epoch
        self.use_pe = use_pe
        self.map_thresh = map_thresh
        self.dis_thresh = dis_thresh
        self.pe_normalization = pe_normalization
        self.query_thresh = query_thresh
        self.query_use_fix_pad = query_use_fix_pad
        self.valid_fut_ts = valid_fut_ts
        self.ego_bev_decoder = ego_bev_decoder
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        self.interaction_pe_type = interaction_pe_type
        self.ego_lcf_feat_idx = ego_lcf_feat_idx
        
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1

        super(VADRLVHead, self).__init__(*args,transformer=transformer,**kwargs)

        # plan_anchors_path = '/home/whut613/zjy/zjy/vad_rl/Bench2DriveZoo/farthest_point_sampling_256.npy'
        # self.plan_anchors = np.load(plan_anchors_path)
        # self.plan_anchors = torch.from_numpy(self.plan_anchors).to(torch.float32).cuda()
        # self.plan_fut_mode = 256

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)
            
        plan_cls_expert_branch = []
        plan_cls_expert_branch.append(Linear(self.embed_dims, self.embed_dims))
        plan_cls_expert_branch.append(nn.LayerNorm(self.embed_dims))
        plan_cls_expert_branch.append(nn.ReLU(inplace=True))
        plan_cls_expert_branch.append(Linear(self.embed_dims, 1))
        self.plan_cls_expert_branch = nn.Sequential(*plan_cls_expert_branch)

        ego_query_pre_branch = []
        ego_query_pre_branch.append(Linear(self.embed_dims * self.fut_ts, self.embed_dims))
        ego_query_pre_branch.append(nn.ReLU())
        ego_query_pre_branch.append(Linear(self.embed_dims, self.embed_dims))
        self.ego_query_pre_branch = nn.Sequential(*ego_query_pre_branch)

        self.ego_feat_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.embed_dims, self.embed_dims, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_dims, self.embed_dims, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_dims, self.embed_dims, bias=True),
                ), # for traffic light
                nn.Sequential(
                    nn.Linear(len(self.ego_lcf_feat_idx), self.embed_dims, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_dims, self.embed_dims, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_dims, self.embed_dims, bias=True),
                ), # for lcf feat
                nn.Sequential(
                    nn.Linear(self.bev_h*self.bev_w*self.embed_dims, self.embed_dims, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_dims, self.embed_dims, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_dims, self.embed_dims, bias=True),
                ), # for bev_feat
                nn.Sequential(
                    nn.Linear(2, self.embed_dims, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_dims, self.embed_dims, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_dims, self.embed_dims, bias=True),
                ), # for target point
                nn.Sequential(
                    nn.Linear(140 * 140, self.embed_dims, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_dims, self.embed_dims, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_dims, self.embed_dims, bias=True),
                ), # for target point rasterized
            ]
        )


        tl_feats_branch = []
        tl_feats_branch.append(Linear(23040, self.embed_dims))
        tl_feats_branch.append(nn.LayerNorm(self.embed_dims))
        tl_feats_branch.append(nn.ReLU(inplace=True))
        tl_feats_branch.append(Linear(self.embed_dims, self.embed_dims))
        tl_feats_branch.append(nn.LayerNorm(self.embed_dims))
        tl_feats_branch.append(nn.ReLU(inplace=True))
        self.tl_feats_branch = nn.Sequential(*tl_feats_branch)

        if self.interaction_pe_type == 'sine_mlp':
            pe_embed_mlps = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims*2),
                nn.ReLU(),
                nn.Linear(self.embed_dims*2, self.embed_dims),
            )
        elif self.interaction_pe_type == 'mlp':
            pe_embed_mlps = nn.Linear(2, self.embed_dims)
        else:
            raise NotImplementedError('Not implement yet')
        
        self.pe_embed_mlps = pe_embed_mlps

        if self.ego_bev_decoder is not None:
            self.ego_bev_decoder = build_transformer_layer_sequence(self.ego_bev_decoder)
            MAXNUM_BEV_TOKEN = 200*200
            self.bev_pos_embedding = nn.Embedding(
                MAXNUM_BEV_TOKEN, self.embed_dims)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.ego_bev_decoder is not None:
            for p in self.ego_bev_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        if self.interaction_pe_type is not None:
            xavier_init(self.pe_embed_mlps, distribution='uniform', bias=0.)

    # @auto_fp16(apply_to=('mlvl_feats'))
    @force_fp32(apply_to=('mlvl_feats', 'prev_bev'))
    def forward(self,
                mlvl_feats,
                img_metas,
                prev_bev=None,
                only_bev=False,
                ego_lcf_feat=None,
                target_point=None
            ):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        bev_queries = self.bev_embedding.weight.to(dtype)
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
            
        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            bev_embed = self.transformer(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev
        )
       
        if isinstance(target_point, torch.Tensor):
            _tmp_target_point = target_point.unsqueeze(1).to(dtype)
        else:
            _tmp_target_point = torch.tensor(target_point[None,None], device=ego_cf_feat.device)
        # range (-70m, +70m) grid_size 1m
        _tmp_rasterized_feat = torch.zeros((bs, 140, 140), dtype=torch.float32, device=ego_cf_feat.device)

        # TODO no need / 2.
        _idx = torch.floor((_tmp_target_point.clip(min=-69., max=69.) - (-70.)) / 2.).long()
        for i in range(bs):
            _tmp_rasterized_feat[i, _idx[i,0,0], _idx[i,0,1]] = 1.
        _tmp_rasterized_feat = _tmp_rasterized_feat.reshape(bs, 1, 140 * 140)
        ego_wp_feat = self.ego_feat_projs[3](_tmp_target_point)
        ego_wp_feat += 1. * self.ego_feat_projs[4](_tmp_rasterized_feat)
        
        # cf_img_h, cf_img_w = cf_img.shape[2:]
        # crop_h = int(cf_img_h/2)
        # crop_w1, crop_w2 = int(cf_img_w/4), int(cf_img_w*3/4)
        # front_view_img = cf_img[:, :, :crop_h, crop_w1:crop_w2]
        # cf_img_feats = self.cf_backbone(cf_img)
        # if isinstance(cf_img_feats, dict):
        #     cf_img_feats = list(cf_img_feats.values())
        # cf_img_feats = torch.cat((cf_img_feats[-1].flatten(1, 3),  mlvl_feats[-1].flatten(1, 4)), dim=-1)
        cf_img_feats = mlvl_feats[-1].flatten(1, 4)
        cf_img_feats = self.tl_feats_branch(cf_img_feats)
        cf_img_feats = cf_img_feats.unsqueeze(1)

        ego_cf_feat = self.ego_feat_projs[0](cf_img_feats)
        ego_status = ego_lcf_feat.squeeze(1)[..., self.ego_lcf_feat_idx].to(dtype)
        ego_status_feat = self.ego_feat_projs[1](ego_status)
        
        bev_feat = bev_embed.reshape(bev_embed.shape[0],-1)
        ego_bev_feat = self.ego_feat_projs[2](bev_feat)
        ego_feats = ego_status_feat + ego_bev_feat + ego_cf_feat + ego_wp_feat
        v_value = self.plan_cls_expert_branch(ego_feats)

        return v_value

    def map_transform_box(self, pts, y_first=False):
        """
        Converting the points set into bounding box.

        Args:
            pts: the input points sets (fields), each points
                set (fields) is represented as 2n scalar.
            y_first: if y_fisrt=True, the point set is represented as
                [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
                represented as [x1, y1, x2, y2 ... xn, yn].
        Returns:
            The bbox [cx, cy, w, h] transformed from points.
        """
        pts_reshape = pts.view(pts.shape[0], self.map_num_vec,
                                self.map_num_pts_per_vec,2)
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        if self.map_transform_method == 'minmax':
            # import pdb;pdb.set_trace()

            xmin = pts_x.min(dim=2, keepdim=True)[0]
            xmax = pts_x.max(dim=2, keepdim=True)[0]
            ymin = pts_y.min(dim=2, keepdim=True)[0]
            ymax = pts_y.max(dim=2, keepdim=True)[0]
            bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
            bbox = bbox_xyxy_to_cxcywh(bbox)
        else:
            raise NotImplementedError
        return bbox, pts_reshape

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_attr_labels,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 10].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 9) in [x,y,z,w,l,h,yaw,vx,vy] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler

        gt_bbox_c = gt_bboxes.shape[-1]

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_bbox_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        return (
            labels, label_weights, bbox_targets, bbox_weights,
            pos_inds, neg_inds
        )

    def _map_get_target_single(self,
                           cls_score,
                           bbox_pred,
                           pts_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_shifts_pts,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]
        assign_result, order_index = self.map_assigner.assign(bbox_pred, cls_score, pts_pred,
                                             gt_bboxes, gt_labels, gt_shifts_pts,
                                             gt_bboxes_ignore)

        sampling_result = self.map_sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.map_num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)
        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        # pts targets
        if order_index is None:
            assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
        else:
            assigned_shift = order_index[sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds]
        pts_targets = pts_pred.new_zeros((pts_pred.size(0),
                        pts_pred.size(1), pts_pred.size(2)))
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0
        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        pts_targets[pos_inds] = gt_shifts_pts[sampling_result.pos_assigned_gt_inds,assigned_shift,:,:]
        return (labels, label_weights, bbox_targets, bbox_weights,
                pts_targets, pts_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_attr_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list,pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_attr_labels_list, gt_bboxes_ignore_list
         )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
                 num_total_pos, num_total_neg)

    def map_get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    pts_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pts_targets_list, pts_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
            self._map_get_target_single, cls_scores_list, bbox_preds_list,pts_preds_list,
            gt_labels_list, gt_bboxes_list, gt_shifts_pts_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, pts_targets_list, pts_weights_list,
                num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_attr_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_attr_labels_list, gt_bboxes_ignore_list)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)



        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)


        return loss_cls, loss_bbox

    def get_best_fut_preds(self,
             traj_preds,
             traj_targets,
             gt_fut_masks):
        """"Choose best preds among all modes.
        Args:
            traj_preds (Tensor): MultiModal traj preds with shape (num_box_preds, fut_mode, fut_ts, 2).
            traj_targets (Tensor): Ground truth traj for each pred box with shape (num_box_preds, fut_ts, 2).
            gt_fut_masks (Tensor): Ground truth traj mask with shape (num_box_preds, fut_ts).
            pred_box_centers (Tensor): Pred box centers with shape (num_box_preds, 2).
            gt_box_centers (Tensor): Ground truth box centers with shape (num_box_preds, 2).

        Returns:
            best_traj_preds (Tensor): best traj preds (min displacement error with gt)
                with shape (num_box_preds, fut_ts*2).
        """

        cum_traj_preds = traj_preds.cumsum(dim=-2)
        cum_traj_targets = traj_targets.cumsum(dim=-2)

        # Get min pred mode indices.
        # (num_box_preds, fut_mode, fut_ts)
        dist = torch.linalg.norm(cum_traj_targets[:, None, :, :] - cum_traj_preds, dim=-1)
        dist = dist * gt_fut_masks[:, None, :]
        dist = dist[..., -1]
        dist[torch.isnan(dist)] = dist[torch.isnan(dist)] * 0
        min_mode_idxs = torch.argmin(dist, dim=-1).tolist()
        box_idxs = torch.arange(traj_preds.shape[0]).tolist()
        best_traj_preds = traj_preds[box_idxs, min_mode_idxs, :, :].reshape(-1, self.fut_ts*2)

        return best_traj_preds

    def get_traj_cls_target(self,
             traj_preds,
             traj_targets,
             gt_fut_masks,
             neg_inds):
        """"Get Trajectory mode classification target.
        Args:
            traj_preds (Tensor): MultiModal traj preds with shape (num_box_preds, fut_mode, fut_ts, 2).
            traj_targets (Tensor): Ground truth traj for each pred box with shape (num_box_preds, fut_ts, 2).
            gt_fut_masks (Tensor): Ground truth traj mask with shape (num_box_preds, fut_ts).
            neg_inds (Tensor): Negtive indices with shape (num_box_preds,)

        Returns:
            traj_labels (Tensor): traj cls labels (num_box_preds,).
        """

        cum_traj_preds = traj_preds.cumsum(dim=-2)
        cum_traj_targets = traj_targets.cumsum(dim=-2)

        # Get min pred mode indices.
        # (num_box_preds, fut_mode, fut_ts)
        dist = torch.linalg.norm(cum_traj_targets[:, None, :, :] - cum_traj_preds, dim=-1)
        dist = dist * gt_fut_masks[:, None, :]
        dist = dist[..., -1]
        dist[torch.isnan(dist)] = dist[torch.isnan(dist)] * 0
        traj_labels = torch.argmin(dist, dim=-1)
        traj_labels[neg_inds] = self.fut_mode

        return traj_labels

    def map_loss_single(self,
                    cls_scores,
                    bbox_preds,
                    pts_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_pts_list (list[Tensor]): Ground truth pts for each image
                with shape (num_gts, fixed_num, 2) in [x,y] format.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.map_get_targets(cls_scores_list, bbox_preds_list,pts_preds_list,
                                           gt_bboxes_list, gt_labels_list,gt_shifts_pts_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pts_targets_list, pts_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
 
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.map_cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.map_bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_map_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_2d_bbox(bbox_targets, self.pc_range)
        # normalized_bbox_targets = bbox_targets
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.map_code_weights

        loss_bbox = self.loss_map_bbox(
            bbox_preds[isnotnan, :4],
            normalized_bbox_targets[isnotnan,:4],
            bbox_weights[isnotnan, :4],
            avg_factor=num_total_pos)

        # regression pts CD loss
        # num_samples, num_order, num_pts, num_coords
        normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range)

        # num_samples, num_pts, num_coords
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2), pts_preds.size(-1))
        if self.map_num_pts_per_vec != self.map_num_pts_per_gt_vec:
            pts_preds = pts_preds.permute(0,2,1)
            pts_preds = F.interpolate(pts_preds, size=(self.map_num_pts_per_gt_vec), mode='linear',
                                    align_corners=True)
            pts_preds = pts_preds.permute(0,2,1).contiguous()

        loss_pts = self.loss_map_pts(
            pts_preds[isnotnan,:,:],
            normalized_pts_targets[isnotnan,:,:], 
            pts_weights[isnotnan,:,:],
            avg_factor=num_total_pos)

        dir_weights = pts_weights[:, :-self.map_dir_interval,0]
        denormed_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)
        denormed_pts_preds_dir = denormed_pts_preds[:,self.map_dir_interval:,:] - \
            denormed_pts_preds[:,:-self.map_dir_interval,:]
        pts_targets_dir = pts_targets[:, self.map_dir_interval:,:] - pts_targets[:,:-self.map_dir_interval,:]

        loss_dir = self.loss_map_dir(
            denormed_pts_preds_dir[isnotnan,:,:],
            pts_targets_dir[isnotnan,:,:],
            dir_weights[isnotnan,:],
            avg_factor=num_total_pos)

        bboxes = denormalize_2d_bbox(bbox_preds, self.pc_range)
        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_map_iou(
            bboxes[isnotnan, :4],
            bbox_targets[isnotnan, :4],
            bbox_weights[isnotnan, :4], 
            avg_factor=num_total_pos)

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_iou = torch.nan_to_num(loss_iou)
            loss_pts = torch.nan_to_num(loss_pts)
            loss_dir = torch.nan_to_num(loss_dir)

        return loss_cls, loss_bbox, loss_iou, loss_pts, loss_dir

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             map_gt_bboxes_list,
             map_gt_labels_list,
             preds_dicts,
             gt_attr_labels,
             gt_bboxes_ignore=None,
             map_gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        map_gt_vecs_list = copy.deepcopy(map_gt_bboxes_list)

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        map_all_cls_scores = preds_dicts['map_all_cls_scores']
        map_all_bbox_preds = preds_dicts['map_all_bbox_preds']
        map_all_pts_preds = preds_dicts['map_all_pts_preds']
        map_enc_cls_scores = preds_dicts['map_enc_cls_scores']
        map_enc_bbox_preds = preds_dicts['map_enc_bbox_preds']
        map_enc_pts_preds = preds_dicts['map_enc_pts_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_attr_labels_list = [gt_attr_labels for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox  = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds, all_gt_bboxes_list, all_gt_labels_list,
            all_gt_attr_labels_list, all_gt_bboxes_ignore_list)
        

        num_dec_layers = len(map_all_cls_scores)
        device = map_gt_labels_list[0].device

        map_gt_bboxes_list = [
            map_gt_bboxes.bbox.to(device) for map_gt_bboxes in map_gt_vecs_list]
        map_gt_pts_list = [
            map_gt_bboxes.fixed_num_sampled_points.to(device) for map_gt_bboxes in map_gt_vecs_list]
        if self.map_gt_shift_pts_pattern == 'v0':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v1':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v1.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v2':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v2.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v3':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v3.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v4':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v4.to(device) for gt_bboxes in map_gt_vecs_list]
        else:
            raise NotImplementedError
        map_all_gt_bboxes_list = [map_gt_bboxes_list for _ in range(num_dec_layers)]
        map_all_gt_labels_list = [map_gt_labels_list for _ in range(num_dec_layers)]
        map_all_gt_pts_list = [map_gt_pts_list for _ in range(num_dec_layers)]
        map_all_gt_shifts_pts_list = [map_gt_shifts_pts_list for _ in range(num_dec_layers)]
        map_all_gt_bboxes_ignore_list = [
            map_gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        map_losses_cls, map_losses_bbox, map_losses_iou, \
            map_losses_pts, map_losses_dir = multi_apply(
            self.map_loss_single, map_all_cls_scores, map_all_bbox_preds,
            map_all_pts_preds, map_all_gt_bboxes_list, map_all_gt_labels_list,
            map_all_gt_shifts_pts_list, map_all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        # loss from the last decoder layer
        loss_dict['loss_map_cls'] = map_losses_cls[-1]
        loss_dict['loss_map_bbox'] = map_losses_bbox[-1]
        loss_dict['loss_map_iou'] = map_losses_iou[-1]
        loss_dict['loss_map_pts'] = map_losses_pts[-1]
        loss_dict['loss_map_dir'] = map_losses_dir[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        # loss from other decoder layers
        num_dec_layer = 0
        for map_loss_cls_i, map_loss_bbox_i, map_loss_iou_i, map_loss_pts_i, map_loss_dir_i in zip(
            map_losses_cls[:-1],
            map_losses_bbox[:-1],
            map_losses_iou[:-1],
            map_losses_pts[:-1],
            map_losses_dir[:-1]
        ):
            loss_dict[f'd{num_dec_layer}.loss_map_cls'] = map_loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_map_bbox'] = map_loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_map_iou'] = map_loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_map_pts'] = map_loss_pts_i
            loss_dict[f'd{num_dec_layer}.loss_map_dir'] = map_loss_dir_i
            num_dec_layer += 1

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list,
                                 gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        if map_enc_cls_scores is not None:
            map_binary_labels_list = [
                torch.zeros_like(map_gt_labels_list[i])
                for i in range(len(map_all_gt_labels_list))
            ]
            # TODO bug here, but we dont care enc_loss now
            map_enc_loss_cls, map_enc_loss_bbox, map_enc_loss_iou, \
                 map_enc_loss_pts, map_enc_loss_dir = \
                self.map_loss_single(
                    map_enc_cls_scores, map_enc_bbox_preds,
                    map_enc_pts_preds, map_gt_bboxes_list,
                    map_binary_labels_list, map_gt_pts_list,
                    map_gt_bboxes_ignore
                )
            loss_dict['enc_loss_map_cls'] = map_enc_loss_cls
            loss_dict['enc_loss_map_bbox'] = map_enc_loss_bbox
            loss_dict['enc_loss_map_iou'] = map_enc_loss_iou
            loss_dict['enc_loss_map_pts'] = map_enc_loss_pts
            loss_dict['enc_loss_map_dir'] = map_enc_loss_dir

        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        det_preds_dicts = self.bbox_coder.decode(preds_dicts)
        # map_bboxes: xmin, ymin, xmax, ymax
        map_preds_dicts = self.map_bbox_coder.decode(preds_dicts)

        num_samples = len(det_preds_dicts)
        assert len(det_preds_dicts) == len(map_preds_dicts), \
             'len(preds_dict) should be equal to len(map_preds_dicts)'
        ret_list = []
        for i in range(num_samples):
            preds = det_preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            code_size = bboxes.shape[-1]
            bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds['scores']
            labels = preds['labels']
            trajs = preds['trajs']

            map_preds = map_preds_dicts[i]
            map_bboxes = map_preds['map_bboxes']
            map_scores = map_preds['map_scores']
            map_labels = map_preds['map_labels']
            map_pts = map_preds['map_pts']

            ret_list.append([bboxes, scores, labels, trajs, map_bboxes,
                             map_scores, map_labels, map_pts])

        return ret_list

    def select_and_pad_pred_map(
        self,
        motion_pos,
        map_query,
        map_score,
        map_pos,
        map_thresh=0.5,
        dis_thresh=None,
        pe_normalization=True,
        use_fix_pad=False
    ):
        """select_and_pad_pred_map.
        Args:
            motion_pos: [B, A, 2]
            map_query: [B, P, D].
            map_score: [B, P, 3].
            map_pos: [B, P, pts, 2].
            map_thresh: map confidence threshold for filtering low-confidence preds
            dis_thresh: distance threshold for masking far maps for each agent in cross-attn
            use_fix_pad: always pad one lane instance for each batch
        Returns:
            selected_map_query: [B*A, P1(+1), D], P1 is the max inst num after filter and pad.
            selected_map_pos: [B*A, P1(+1), 2]
            selected_padding_mask: [B*A, P1(+1)]
        """
        
        if dis_thresh is None:
            raise NotImplementedError('Not implement yet')

        # use the most close pts pos in each map inst as the inst's pos
        batch, num_map = map_pos.shape[:2]
        map_dis = torch.sqrt(map_pos[..., 0]**2 + map_pos[..., 1]**2)
        min_map_pos_idx = map_dis.argmin(dim=-1).flatten()  # [B*P]
        min_map_pos = map_pos.flatten(0, 1)  # [B*P, pts, 2]
        min_map_pos = min_map_pos[range(min_map_pos.shape[0]), min_map_pos_idx]  # [B*P, 2]
        min_map_pos = min_map_pos.view(batch, num_map, 2)  # [B, P, 2]

        # select & pad map vectors for different batch using map_thresh
        map_score = map_score.sigmoid()
        map_max_score = map_score.max(dim=-1)[0]
        map_idx = map_max_score > map_thresh
        batch_max_pnum = 0
        for i in range(map_score.shape[0]):
            pnum = map_idx[i].sum()
            if pnum > batch_max_pnum:
                batch_max_pnum = pnum

        selected_map_query, selected_map_pos, selected_padding_mask = [], [], []
        for i in range(map_score.shape[0]):
            dim = map_query.shape[-1]
            valid_pnum = map_idx[i].sum()
            valid_map_query = map_query[i, map_idx[i]]
            valid_map_pos = min_map_pos[i, map_idx[i]]
            pad_pnum = batch_max_pnum - valid_pnum
            padding_mask = torch.tensor([False], device=map_score.device).repeat(batch_max_pnum)
            if pad_pnum != 0:
                valid_map_query = torch.cat([valid_map_query, torch.zeros((pad_pnum, dim), device=map_score.device)], dim=0)
                valid_map_pos = torch.cat([valid_map_pos, torch.zeros((pad_pnum, 2), device=map_score.device)], dim=0)
                padding_mask[valid_pnum:] = True
            selected_map_query.append(valid_map_query)
            selected_map_pos.append(valid_map_pos)
            selected_padding_mask.append(padding_mask)

        selected_map_query = torch.stack(selected_map_query, dim=0)
        selected_map_pos = torch.stack(selected_map_pos, dim=0)
        selected_padding_mask = torch.stack(selected_padding_mask, dim=0)

        # generate different pe for map vectors for each agent
        num_agent = motion_pos.shape[1]
        selected_map_query = selected_map_query.unsqueeze(1).repeat(1, num_agent, 1, 1)  # [B, A, max_P, D]
        selected_map_pos = selected_map_pos.unsqueeze(1).repeat(1, num_agent, 1, 1)  # [B, A, max_P, 2]
        selected_padding_mask = selected_padding_mask.unsqueeze(1).repeat(1, num_agent, 1)  # [B, A, max_P]
        # move lane to per-car coords system
        selected_map_dist = selected_map_pos - motion_pos[:, :, None, :]  # [B, A, max_P, 2]
        if pe_normalization:
            selected_map_pos = selected_map_pos - motion_pos[:, :, None, :]  # [B, A, max_P, 2]

        # filter far map inst for each agent
        map_dis = torch.sqrt(selected_map_dist[..., 0]**2 + selected_map_dist[..., 1]**2)
        valid_map_inst = (map_dis <= dis_thresh)  # [B, A, max_P]
        invalid_map_inst = (valid_map_inst == False)
        selected_padding_mask = selected_padding_mask + invalid_map_inst

        selected_map_query = selected_map_query.flatten(0, 1)
        selected_map_pos = selected_map_pos.flatten(0, 1)
        selected_padding_mask = selected_padding_mask.flatten(0, 1)

        num_batch = selected_padding_mask.shape[0]
        feat_dim = selected_map_query.shape[-1]
        if use_fix_pad:
            pad_map_query = torch.zeros((num_batch, 1, feat_dim), device=selected_map_query.device)
            pad_map_pos = torch.ones((num_batch, 1, 2), device=selected_map_pos.device)
            pad_lane_mask = torch.tensor([False], device=selected_padding_mask.device).unsqueeze(0).repeat(num_batch, 1)
            selected_map_query = torch.cat([selected_map_query, pad_map_query], dim=1)
            selected_map_pos = torch.cat([selected_map_pos, pad_map_pos], dim=1)
            selected_padding_mask = torch.cat([selected_padding_mask, pad_lane_mask], dim=1)

        return selected_map_query, selected_map_pos, selected_padding_mask


    def select_and_pad_query(
        self,
        query,
        query_pos,
        query_score,
        score_thresh=0.5,
        use_fix_pad=True
    ):
        """select_and_pad_query.
        Args:
            query: [B, Q, D].
            query_pos: [B, Q, 2]
            query_score: [B, Q, C].
            score_thresh: confidence threshold for filtering low-confidence query
            use_fix_pad: always pad one query instance for each batch
        Returns:
            selected_query: [B, Q', D]
            selected_query_pos: [B, Q', 2]
            selected_padding_mask: [B, Q']
        """

        # select & pad query for different batch using score_thresh
        query_score = query_score.sigmoid()
        query_score = query_score.max(dim=-1)[0]
        query_idx = query_score > score_thresh
        batch_max_qnum = 0
        for i in range(query_score.shape[0]):
            qnum = query_idx[i].sum()
            if qnum > batch_max_qnum:
                batch_max_qnum = qnum

        selected_query, selected_query_pos, selected_padding_mask = [], [], []
        for i in range(query_score.shape[0]):
            dim = query.shape[-1]
            valid_qnum = query_idx[i].sum()
            valid_query = query[i, query_idx[i]]
            valid_query_pos = query_pos[i, query_idx[i]]
            pad_qnum = batch_max_qnum - valid_qnum
            padding_mask = torch.tensor([False], device=query_score.device).repeat(batch_max_qnum)
            if pad_qnum != 0:
                valid_query = torch.cat([valid_query, torch.zeros((pad_qnum, dim), device=query_score.device)], dim=0)
                valid_query_pos = torch.cat([valid_query_pos, torch.zeros((pad_qnum, 2), device=query_score.device)], dim=0)
                padding_mask[valid_qnum:] = True
            selected_query.append(valid_query)
            selected_query_pos.append(valid_query_pos)
            selected_padding_mask.append(padding_mask)

        selected_query = torch.stack(selected_query, dim=0)
        selected_query_pos = torch.stack(selected_query_pos, dim=0)
        selected_padding_mask = torch.stack(selected_padding_mask, dim=0)

        num_batch = selected_padding_mask.shape[0]
        feat_dim = selected_query.shape[-1]
        if use_fix_pad:
            pad_query = torch.zeros((num_batch, 1, feat_dim), device=selected_query.device)
            pad_query_pos = torch.ones((num_batch, 1, 2), device=selected_query_pos.device)
            pad_mask = torch.tensor([False], device=selected_padding_mask.device).unsqueeze(0).repeat(num_batch, 1)
            selected_query = torch.cat([selected_query, pad_query], dim=1)
            selected_query_pos = torch.cat([selected_query_pos, pad_query_pos], dim=1)
            selected_padding_mask = torch.cat([selected_padding_mask, pad_mask], dim=1)

        return selected_query, selected_query_pos, selected_padding_mask

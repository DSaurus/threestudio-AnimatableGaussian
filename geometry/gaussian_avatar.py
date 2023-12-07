#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import NamedTuple

import numpy as np
import threestudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from threestudio.models.geometry.base import BaseGeometry
from threestudio.utils.typing import *
from .poser import Skeleton
from .network import NomalNet

C0 = 0.28209479177387814


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def safe_state(silent):
    old_f = sys.stdout

    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(
                        x.replace(
                            "\n",
                            " [{}]\n".format(
                                str(datetime.now().strftime("%d/%m %H:%M:%S"))
                            ),
                        )
                    )
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


class Camera(NamedTuple):
    FoVx: torch.Tensor
    FoVy: torch.Tensor
    camera_center: torch.Tensor
    image_width: int
    image_height: int
    world_view_transform: torch.Tensor
    full_proj_transform: torch.Tensor


def save_ply(vertices):
    import pandas as pd
    # Convert the numpy array to a DataFrame
    df = pd.DataFrame(vertices, columns=['x', 'y', 'z'])
    # Create a PlyElement from the DataFrame
    element = PlyElement.describe(df.to_records(index=False), 'vertex')
    # Create a PlyData object
    ply_data = PlyData([element])
    # Write the ply file
    ply_data.write('.threestudio_cache/output.ply')

@threestudio.register("avatar-gaussian")
class GaussianAvatarModel(BaseGeometry):
    @dataclass
    class Config(BaseGeometry.Config):
        sh_degree: int = 0
        feature_network_lr: float = 0.0

        smplx_path: str = "/path/to/smplx/model"
        disable_hand_densification: bool = False
        smplx_hand_radius: float = 0.05
        smplx_gender: str = 'neutral'
        smplx_apose: bool = True

        mesh_renderer_config: dict = field(default_factory=dict)

    cfg: Config

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def configure(self) -> None:
        super().configure()
        self.active_sh_degree = 0
        self.max_sh_degree = self.cfg.sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.skel = Skeleton(humansd_style=True, apose=self.cfg.smplx_apose)
        self.skel.load_smplx(self.cfg.smplx_path, gender=self.cfg.smplx_gender)
        self.skel.scale(-10)

        self.mesh_renderer = threestudio.find("smplx-mesh-renderer")(
            self.cfg.mesh_renderer_config,
            self.skel.vertices,
            self.skel.faces
        )

        output = self.mesh_renderer(512, 512)
        self.front_view_position = output["comp_rgb"][0].permute(2, 0, 1).unsqueeze(0)
        self.back_view_position = output["comp_rgb"][1].permute(2, 0, 1).unsqueeze(0)
        self.front_view_mask = output["opacity"][0].permute(2, 0, 1).unsqueeze(0)
        self.back_view_mask = output["opacity"][1].permute(2, 0, 1).unsqueeze(0)

        self.feature_cache = False
        # xyz scaling rotation opacity features
        self.feature_unet_front = NormalNet(input_nc=3, output_nc=3+3+4+1+3)
        self.feature_unet_back = NormalNet(input_nc=3, output_nc=3+3+4+1+3)


        self.setup_functions()

    def get_features(self):
        self.feature_cache = True
        features_front = self.feature_unet_front(self.front_view_position)
        feature_back = self.feature_unet_back(self.back_view_position)
        xyz_front = 0.1*features_front[:, :3] + self.front_view_position
        xyz_back = 0.1*feature_back[:, :3] + self.back_view_position
        scale_front = features_front[:, 3:6]
        scale_back = feature_back[:, 3:6]
        rotation_front = features_front[:, 6:10]
        rotation_back = feature_back[:, 6:10]
        opacity_front = features_front[:, 10:11]
        opacity_back = feature_back[:, 10:11]
        color_front = features_front[:, 11:]
        color_back = feature_back[:, 11:]
        mask_front = self.front_view_mask[0, 0]
        mask_back = self.back_view_mask[0, 0]
        self._xyz = torch.cat((xyz_front.permute(0, 2, 3, 1)[0, mask_front], xyz_back.permute(0, 2, 3, 1)[0, mask_back]), dim=0)
        self._scaling = torch.cat((scale_front.permute(0, 2, 3, 1)[0, mask_front], scale_back.permute(0, 2, 3, 1)[0, mask_back]), dim=0)
        self._rotation = torch.cat((rotation_front.permute(0, 2, 3, 1)[0, mask_front], rotation_back.permute(0, 2, 3, 1)[0, mask_back]), dim=0)
        self._opacity = torch.cat((opacity_front.permute(0, 2, 3, 1)[0, mask_front], opacity_back.permute(0, 2, 3, 1)[0, mask_back]), dim=0)
        self._features_dc = torch.cat((color_front.permute(0, 2, 3, 1)[0, mask_front], color_back.permute(0, 2, 3, 1)[0, mask_back]), dim=0)

    @property
    def get_scaling(self):
        if not self.feature_cache:
            self.get_features()
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        if not self.feature_cache:
            self.get_features()
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        if not self.feature_cache:
            self.get_features()
        return self._xyz

    @property
    def get_features(self):
        if not self.feature_cache:
            self.get_features()
        features_dc = self._features_dc.unsqueeze(1)
        # features_rest = self._features_rest
        return features_dc

    @property
    def get_opacity(self):
        if not self.feature_cache:
            self.get_features()
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )


    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def to(self, device="cpu"):
        self._xyz = self._xyz.to(device)
        self._features_dc = self._features_dc.to(device)
        self._features_rest = self._features_rest.to(device)
        self._opacity = self._opacity.to(device)
        self._scaling = self._scaling.to(device)
        self._rotation = self._rotation.to(device)

    @torch.no_grad()
    def update_states(
        self,
        iteration,
        visibility_filter,
        radii,
        viewspace_point_tensor,
    ):
        self.feature_cache = False
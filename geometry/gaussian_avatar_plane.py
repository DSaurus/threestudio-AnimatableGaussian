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
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import threestudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from threestudio.models.geometry.base import BaseGeometry
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.typing import *
from torch.cuda.amp import autocast

from .gaussian_function import *
from .network import NormalNet
from .poser import Skeleton
from .styleunet.styleunet import StyleUNet


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
    df = pd.DataFrame(vertices, columns=["x", "y", "z"])
    # Create a PlyElement from the DataFrame
    element = PlyElement.describe(df.to_records(index=False), "vertex")
    # Create a PlyData object
    ply_data = PlyData([element])
    # Write the ply file
    ply_data.write(".threestudio_cache/output.ply")


@threestudio.register("avatar-gaussian-plane")
class GaussianAvatarModel(BaseGeometry):
    @dataclass
    class Config(BaseGeometry.Config):
        sh_degree: int = 0
        feature_network_lr: float = 0.0

        pos_encoding_config: dict = field(
            default_factory=lambda: {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.447269237440378,
            }
        )
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )

        smplx_path: str = "/path/to/smplx/model"
        disable_hand_densification: bool = False
        smplx_hand_radius: float = 0.05
        smplx_gender: str = "neutral"
        smplx_apose: bool = True

        mesh_renderer_config: dict = field(default_factory=dict)

        opacity_init: float = 0.9

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
        self.optimizer = None

        self.skel = Skeleton(humansd_style=True, apose=self.cfg.smplx_apose)
        self.skel.load_smplx(self.cfg.smplx_path, gender=self.cfg.smplx_gender)
        self.skel.scale(-10)
        self.mesh_renderer = threestudio.find("smplx-mesh-renderer")(
            self.cfg.mesh_renderer_config, self.skel.vertices, self.skel.faces
        )
        output = self.mesh_renderer(512, 512)
        self.cache_data = {
            "front_view_position": output["comp_rgb"][0].permute(2, 0, 1).unsqueeze(0),
            "back_view_position": output["comp_rgb"][1].permute(2, 0, 1).unsqueeze(0),
            "front_view_mask": output["opacity"][0].permute(2, 0, 1).unsqueeze(0)
            > 0.99,
            "back_view_mask": output["opacity"][1].permute(2, 0, 1).unsqueeze(0) > 0.99,
        }
        self.front_hash_encoding = get_encoding(3, self.cfg.pos_encoding_config).to(
            self.device
        )
        self.front_hash_network = get_mlp(
            self.front_hash_encoding.n_output_dims,
            3 + 4 + 1 + 3,
            self.cfg.mlp_network_config,
        ).to(self.device)
        self.front_pos_encoding = get_encoding(3, self.cfg.pos_encoding_config).to(
            self.device
        )
        self.front_pos_network = get_mlp(
            self.front_pos_encoding.n_output_dims, 3, self.cfg.mlp_network_config
        ).to(self.device)

        self.back_hash_encoding = get_encoding(3, self.cfg.pos_encoding_config).to(
            self.device
        )
        self.back_hash_network = get_mlp(
            self.back_hash_encoding.n_output_dims,
            3 + 4 + 1 + 3,
            self.cfg.mlp_network_config,
        ).to(self.device)
        self.back_pos_encoding = get_encoding(3, self.cfg.pos_encoding_config).to(
            self.device
        )
        self.back_pos_network = get_mlp(
            self.back_pos_encoding.n_output_dims, 3, self.cfg.mlp_network_config
        ).to(self.device)

        self.get_network_features()

        # self.base_xyz = torch.from_numpy(self.skel.sample_smplx_points(512*512)).float().to(self.device)
        self.base_xyz = self.cache_data["_xyz"].detach()
        dist2 = torch.clamp_min(
            distCUDA2(self.base_xyz.float().cuda()),
            0.0000001,
        )
        self.base_scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3) - 1
        self.base_opacity = inverse_sigmoid(
            self.cfg.opacity_init * torch.ones((1, 1), dtype=torch.float, device="cuda")
        )

        # import cv2
        # cv2.imwrite(".threestudio_cache/rgb_0.jpg", (output["comp_rgb"][0].detach().cpu().numpy()*0.5+0.5)*255)
        # cv2.imwrite(".threestudio_cache/rgb_1.jpg", (output["comp_rgb"][1].detach().cpu().numpy()*0.5+0.5)*255)
        # exit(0)

        # xyz scaling rotation opacity features
        # self.feature_unet = NormalNet(input_nc=6, output_nc=32, ngf=32, n_blocks=3, last_op=nn.Identity())

        self.setup_functions()

    def get_network_features(self):
        front_view_position = self.cache_data["front_view_position"]
        back_view_position = self.cache_data["back_view_position"]
        mask_front = self.cache_data["front_view_mask"][0, 0]
        mask_back = self.cache_data["back_view_mask"][0, 0]

        xyz_front = front_view_position
        xyz_back = back_view_position
        xyz_front = xyz_front.permute(0, 2, 3, 1)[0, mask_front]
        xyz_back = xyz_back.permute(0, 2, 3, 1)[0, mask_back]

        features_front = self.front_hash_network(self.front_hash_encoding(xyz_front))
        features_back = self.back_hash_network(self.back_hash_encoding(xyz_back))
        pos_front = (
            0.002 * self.front_pos_network(self.front_pos_encoding(xyz_front))
            + xyz_front
        )
        pos_back = (
            0.002 * self.back_pos_network(self.back_pos_encoding(xyz_back)) + xyz_back
        )
        scale_front = features_front[:, :3]
        scale_back = features_back[:, :3]
        rotation_front = features_front[:, 3:7]
        rotation_back = features_back[:, 3:7]
        opacity_front = features_front[:, 7:8]
        opacity_back = features_back[:, 7:8]
        color_front = features_front[:, 8:]
        color_back = features_back[:, 8:]
        self.cache_data.update(
            {
                "_xyz": torch.cat(
                    (pos_front, pos_back),
                    dim=0,
                ),
                "_scaling": torch.cat(
                    (scale_front, scale_back),
                    dim=0,
                ),
                "_rotation": torch.cat(
                    (rotation_front, rotation_back),
                    dim=0,
                ),
                "_opacity": torch.cat(
                    (opacity_front, opacity_back),
                    dim=0,
                ),
                "_features_dc": torch.cat(
                    (color_front, color_back),
                    dim=0,
                ),
            }
        )

    @property
    def get_scaling(self):
        return self.scaling_activation(
            F.tanh(self.cache_data["_scaling"]) * 2 + self.base_scales
        )

    @property
    def get_rotation(self):
        return self.rotation_activation(self.cache_data["_rotation"])

    @property
    def get_xyz(self):
        return self.cache_data["_xyz"]

    @property
    def get_features(self):
        features_dc = self.cache_data["_features_dc"].unsqueeze(1)
        # features_rest = self._features_rest
        return F.tanh(features_dc)

    @property
    def get_opacity(self):
        return self.opacity_activation(self.cache_data["_opacity"] + self.base_opacity)

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
        pass

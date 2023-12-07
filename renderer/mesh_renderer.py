import math
from dataclasses import dataclass

import nerfacc
import numpy as np
import threestudio
import torch
import torch.nn.functional as F
from threestudio.models.renderers.base import Rasterizer, VolumeRenderer
from threestudio.utils.misc import get_device
from threestudio.utils.ops import get_mvp_matrix, get_projection_matrix
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *


@threestudio.register("smplx-mesh-renderer")
class SMPLXMeshRenderer(Rasterizer):
    @dataclass
    class Config(VolumeRenderer.Config):
        context_type: str = "cuda"

    cfg: Config

    def configure(self, vertices, faces) -> None:
        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, get_device())

        # configuration for rendering front and back views
        azimuth_deg: Float[Tensor, "B"]
        azimuth_deg = torch.linspace(-90.0, 90.0, 2)
        print(azimuth_deg)
        elevation_deg: Float[Tensor, "B"] = torch.full_like(azimuth_deg, 0)
        camera_distances: Float[Tensor, "B"] = torch.full_like(elevation_deg, 7.5)

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180

        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(azimuth_deg.shape[0], 1)

        fovy_deg: Float[Tensor, "B"] = torch.full_like(elevation_deg, 15)
        fovy = fovy_deg * math.pi / 180

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, 1.0, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        self.mvp_mtx = mvp_mtx.to(self.device)
        self.c2w = c2w.to(self.device)
        self.v_pos = torch.tensor(vertices, dtype=torch.float32).to(self.device)
        self.t_pos_idx = torch.tensor(faces.astype(np.int32), dtype=torch.int64).to(
            self.device
        )

    def forward(self, height: int, width: int) -> Dict[str, Any]:
        batch_size = self.mvp_mtx.shape[0]

        v_pos_clip: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
            self.v_pos, self.mvp_mtx
        )
        rast, _ = self.ctx.rasterize(v_pos_clip, self.t_pos_idx, (height, width))
        mask = rast[..., 3:] > 0
        mask_aa = self.ctx.antialias(mask.float(), rast, v_pos_clip, self.t_pos_idx)

        out = {"opacity": mask_aa}

        gb_rgb_fg, _ = self.ctx.interpolate_one(self.v_pos, rast, self.t_pos_idx)
        out.update({"comp_rgb": gb_rgb_fg})

        return out

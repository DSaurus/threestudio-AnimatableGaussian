# from shap_e.diffusion.sample import sample_latents
# from shap_e.diffusion.gaussian_diffusion import diffusion_from_config as diffusion_from_config_shape
# from shap_e.models.download import load_model, load_config
# from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
# from shap_e.util.notebooks import decode_latent_mesh
import io
import os
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import threestudio
import torch
import torch.nn.functional as F
from gaussiansplatting.arguments import (
    ModelParams,
    OptimizationParams,
    PipelineParams,
    get_combined_args,
)
from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene import GaussianModel, Scene
from gaussiansplatting.scene.cameras import Camera, MiniCam
from gaussiansplatting.scene.gaussian_model import BasicPointCloud
from gaussiansplatting.utils.sh_utils import SH2RGB
from PIL import Image
from plyfile import PlyData, PlyElement
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot, get_cam_info_gaussian
from threestudio.utils.poser import Skeleton
from threestudio.utils.typing import *

# import open3d as o3d


def load_ply(path, save_path):
    C0 = 0.28209479177387814

    def SH2RGB(sh):
        return sh * C0 + 0.5

    plydata = PlyData.read(path)

    xyz = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ),
        axis=1,
    )

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
    color = SH2RGB(features_dc[:, :, 0])

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    point_cloud.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_point_cloud(save_path, point_cloud)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


@threestudio.register("avatar-system")
class GaussianDreamer(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        texture_structure_joint: bool = False
        controlnet: bool = False

        size_threshold_fix_step: int = 1500
        half_scheduler_max_step: int = 1500

        bg_white: bool = False

    cfg: Config

    def configure(self) -> None:
        self.gaussian = GaussianModel(sh_degree=0)
        self.background_tensor = (
            torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
            if self.cfg.bg_white
            else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        )

        self.parser = ArgumentParser(description="Training script parameters")
        self.pipe = PipelineParams(self.parser)

        self.texture_structure_joint = self.cfg.texture_structure_joint
        self.controlnet = self.cfg.controlnet

    def save_gif_to_file(self, images, output_file):
        with io.BytesIO() as writer:
            images[0].save(
                writer,
                format="GIF",
                save_all=True,
                append_images=images[1:],
                duration=100,
                loop=0,
            )
            writer.seek(0)
            with open(output_file, "wb") as file:
                file.write(writer.read())

    def forward(self, batch: Dict[str, Any], renderbackground=None) -> Dict[str, Any]:
        if renderbackground is None:
            renderbackground = self.background_tensor

        images = []
        depths = []
        pose_images = []
        self.viewspace_point_list = []

        for batch_idx in range(batch["c2w"].shape[0]):
            batch["batch_idx"] = batch_idx
            fovy = batch["fovy"][batch_idx]
            w2c, proj, cam_p = get_cam_info_gaussian(
                c2w=batch["c2w"][batch_idx], fovx=fovy, fovy=fovy, znear=0.1, zfar=100
            )

            # import pdb; pdb.set_trace()
            viewpoint_cam = Camera(
                FoVx=fovy,
                FoVy=fovy,
                image_width=batch["width"],
                image_height=batch["height"],
                world_view_transform=w2c,
                full_proj_transform=proj,
                camera_center=cam_p,
            )

            render_pkg = self.renderer(viewpoint_cam, self.background_tensor, **batch)

            depth = render_pkg["depth"]
            image = render_pkg["render"]

            # import kiui
            # kiui.vis.plot_image(image)

            depth = depth.permute(1, 2, 0)
            image = image.permute(1, 2, 0)
            images.append(image)
            depths.append(depth)

            if self.texture_structure_joint:
                backview = abs(batch["azimuth"][id]) > 120 * np.pi / 180
                mvp = batch["mvp_mtx"][id].detach().cpu().numpy()  # [4, 4]
                pose_image, _ = self.geometry.skel.humansd_draw(
                    mvp, 512, 512, backview
                )  # [512, 512, 3], fixed pose image resolution
                # kiui.vis.plot_image(pose_image)
                pose_image = torch.from_numpy(pose_image).to(self.device)  # [H, W, 3]
                pose_images.append(pose_image)
            else:
                # render pose image
                backview = abs(batch["azimuth"][id]) > 120 * np.pi / 180
                mvp = batch["mvp_mtx"][id].detach().cpu().numpy()  # [4, 4]
                pose_image, _ = self.geometry.skel.draw(
                    mvp, 512, 512, backview
                )  # [512, 512, 3], fixed pose image resolution
                # kiui.vis.plot_image(pose_image)
                pose_image = torch.from_numpy(pose_image).to(self.device)  # [H, W, 3]
                pose_images.append(pose_image)

        images = torch.stack(images, 0)
        depths = torch.stack(depths, 0)
        pose_images = torch.stack(pose_images, 0)

        self.visibility_filter = self.radii > 0.0

        render_pkg["comp_rgb"] = images
        render_pkg["depth"] = depths
        render_pkg["pose"] = pose_images
        render_pkg["opacity"] = depths / (depths.max() + 1e-5)

        return {
            **render_pkg,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

    def training_step(self, batch, batch_idx):
        self.gaussian.update_learning_rate(self.true_global_step)

        if self.true_global_step > self.cfg.half_scheduler_max_step:
            self.guidance.set_min_max_steps(
                min_step_percent=0.02, max_step_percent=0.55
            )

        self.gaussian.update_learning_rate(self.true_global_step)

        out = self(batch)

        prompt_utils = self.prompt_processor()
        images = out["comp_rgb"]
        depth_images = out["depth"]
        depth_min = torch.amin(depth_images, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_images, dim=[1, 2, 3], keepdim=True)
        depth_images = (depth_images - depth_min) / (
            depth_max - depth_min + 1e-10
        )  # to [0, 1]
        depth_images = depth_images.repeat(1, 1, 1, 3)  # to 3-channel
        control_images = out["pose"]

        # guidance_eval = (self.true_global_step % 200 == 0)
        guidance_eval = False

        if self.texture_structure_joint:
            guidance_out = self.guidance(
                control_images,
                images,
                depth_images,
                prompt_utils,
                **batch,
                rgb_as_latents=False,
                guidance_eval=guidance_eval,
            )
        elif self.controlnet:
            guidance_out = self.guidance(
                control_images,
                images,
                prompt_utils,
                **batch,
                rgb_as_latents=False,
                guidance_eval=guidance_eval,
            )
        else:
            guidance_out = self.guidance(
                images,
                prompt_utils,
                **batch,
                rgb_as_latents=False,
                guidance_eval=guidance_eval,
            )

        loss = 0.0

        loss = loss + guidance_out["loss_sds"] * self.C(self.cfg.loss["lambda_sds"])

        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)
        if guidance_eval:
            self.guidance_evaluation_save(
                out["comp_rgb"].detach()[: guidance_out["eval"]["bs"]],
                guidance_out["eval"],
            )
        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "rgb" in batch
                else []
            )
            + [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            ),
            name="validation_step",
            step=self.true_global_step,
        )
        # save_path = self.get_save_path(f"it{self.true_global_step}-val.ply")
        # self.gaussian.save_ply(save_path)
        # load_ply(save_path,self.get_save_path(f"it{self.true_global_step}-val-color.ply"))

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        only_rgb = True
        bg_color = [1, 1, 1] if self.cfg.bg_white else [0, 0, 0]

        testbackground_tensor = torch.tensor(
            bg_color, dtype=torch.float32, device="cuda"
        )

        out = self(batch, testbackground_tensor)
        if only_rgb:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                ),
                name="test_step",
                step=self.true_global_step,
            )
        else:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "grayscale",
                            "img": out["depth"][0],
                            "kwargs": {},
                        }
                    ]
                    if "depth" in out
                    else []
                )
                + [
                    {
                        "type": "grayscale",
                        "img": out["opacity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ],
                name="test_step",
                step=self.true_global_step,
            )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
        save_path = self.get_save_path(f"last.ply")
        self.gaussian.save_ply(save_path)
        # self.pointefig.savefig(self.get_save_path("pointe.png"))
        # o3d.io.write_point_cloud(self.get_save_path("shape.ply"), self.point_cloud)
        # self.save_gif_to_file(self.shapeimages, self.get_save_path("shape.gif"))
        # load_ply(save_path,self.get_save_path(f"it{self.true_global_step}-test-color.ply"))

    def configure_optimizers(self):
        opt = OptimizationParams(self.parser)

        point_cloud = self.pcb()
        self.gaussian.create_from_pcd(point_cloud, self.cameras_extent)

        self.gaussian.training_setup(opt)

        ret = {
            "optimizer": self.gaussian.optimizer,
        }

        return ret

    def guidance_evaluation_save(self, comp_rgb, guidance_eval_out):
        B, size = comp_rgb.shape[:2]
        resize = lambda x: F.interpolate(
            x.permute(0, 3, 1, 2), (size, size), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        filename = f"it{self.true_global_step}-train.png"

        def merge12(x):
            return x.reshape(-1, *x.shape[2:])

        self.save_image_grid(
            filename,
            [
                {
                    "type": "rgb",
                    "img": merge12(comp_rgb),
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_noisy"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_1step"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_1orig"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_final"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(
                            resize(guidance_eval_out["midas_depth_imgs_noisy"])
                        ),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(
                            resize(guidance_eval_out["midas_depth_imgs_1step"])
                        ),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(
                            resize(guidance_eval_out["midas_depth_imgs_1orig"])
                        ),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(
                            resize(guidance_eval_out["midas_depth_imgs_final"])
                        ),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            ),
            name="train_step",
            step=self.true_global_step,
            texts=guidance_eval_out["texts"],
        )

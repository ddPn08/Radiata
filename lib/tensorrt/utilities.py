#! fork: https://github.com/NVIDIA/TensorRT/blob/main/demo/Diffusion/utilities.py

#
# Copyright 2022 The HuggingFace Inc. team.
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import gc
from collections import OrderedDict
from copy import copy
from typing import *

import diffusers
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import tensorrt as trt
import torch
from cuda import cudart
from PIL import Image
from polygraphy import cuda
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import (
    CreateConfig,
    Profile,
    engine_from_bytes,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)
from polygraphy.backend.trt import util as trt_util
from transformers import CLIPTextConfig

from .models import CLIP, VAE, BaseModel, UNet, VAEEncoder

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool

# Map of torch dtype -> numpy dtype
torch_to_numpy_dtype_dict = {
    value: key for (key, value) in numpy_to_torch_dtype_dict.items()
}


def device_view(t: torch.Tensor, dtype: Optional[np.dtype] = None):
    return cuda.DeviceView(
        ptr=t.data_ptr(),
        shape=t.shape,
        dtype=dtype or torch_to_numpy_dtype_dict[t.dtype],
    )


class Engine:
    def __init__(
        self,
        engine_path: str,
    ):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()

    def __del__(self):
        [
            buf.free()
            for buf in self.buffers.values()
            if isinstance(buf, cuda.DeviceArray)
        ]
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def refit(self, onnx_path: str, onnx_refit_path: str):
        def convert_int64(arr: torch.Tensor):
            # TODO: smarter conversion
            if len(arr.shape) == 0:
                return np.int32(arr)
            return arr

        def add_to_map(refit_dict, name, values):
            if name in refit_dict:
                assert refit_dict[name] is None
                if values.dtype == np.int64:
                    values = convert_int64(values)
                refit_dict[name] = values

        print(f"Refitting TensorRT engine with {onnx_refit_path} weights")
        refit_nodes = gs.import_onnx(onnx.load(onnx_refit_path)).toposort().nodes

        # Construct mapping from weight names in refit model -> original model
        name_map = {}
        for n, node in enumerate(gs.import_onnx(onnx.load(onnx_path)).toposort().nodes):
            refit_node = refit_nodes[n]
            assert node.op == refit_node.op
            # Constant nodes in ONNX do not have inputs but have a constant output
            if node.op == "Constant":
                name_map[refit_node.outputs[0].name] = node.outputs[0].name
            # Handle scale and bias weights
            elif node.op == "Conv":
                if node.inputs[1].__class__ == gs.Constant:
                    name_map[refit_node.name + "_TRTKERNEL"] = node.name + "_TRTKERNEL"
                if node.inputs[2].__class__ == gs.Constant:
                    name_map[refit_node.name + "_TRTBIAS"] = node.name + "_TRTBIAS"
            # For all other nodes: find node inputs that are initializers (gs.Constant)
            else:
                for i, inp in enumerate(node.inputs):
                    if inp.__class__ == gs.Constant:
                        name_map[refit_node.inputs[i].name] = inp.name

        def map_name(name: str):
            if name in name_map:
                return name_map[name]
            return name

        # Construct refit dictionary
        refit_dict = {}
        refitter = trt.Refitter(self.engine, TRT_LOGGER)
        all_weights = refitter.get_all()
        for layer_name, role in zip(all_weights[0], all_weights[1]):
            # for speciailized roles, use a unique name in the map:
            if role == trt.WeightsRole.KERNEL:
                name = layer_name + "_TRTKERNEL"
            elif role == trt.WeightsRole.BIAS:
                name = layer_name + "_TRTBIAS"
            else:
                name = layer_name

            assert name not in refit_dict, "Found duplicate layer: " + name
            refit_dict[name] = None

        for n in refit_nodes:
            # Constant nodes in ONNX do not have inputs but have a constant output
            if n.op == "Constant":
                name = map_name(n.outputs[0].name)
                print(f"Add Constant {name}\n")
                add_to_map(refit_dict, name, n.outputs[0].values)

            # Handle scale and bias weights
            elif n.op == "Conv":
                if n.inputs[1].__class__ == gs.Constant:
                    name = map_name(n.name + "_TRTKERNEL")
                    add_to_map(refit_dict, name, n.inputs[1].values)

                if n.inputs[2].__class__ == gs.Constant:
                    name = map_name(n.name + "_TRTBIAS")
                    add_to_map(refit_dict, name, n.inputs[2].values)

            # For all other nodes: find node inputs that are initializers (AKA gs.Constant)
            else:
                for inp in n.inputs:
                    name = map_name(inp.name)
                    if inp.__class__ == gs.Constant:
                        add_to_map(refit_dict, name, inp.values)

        for layer_name, weights_role in zip(all_weights[0], all_weights[1]):
            if weights_role == trt.WeightsRole.KERNEL:
                custom_name = layer_name + "_TRTKERNEL"
            elif weights_role == trt.WeightsRole.BIAS:
                custom_name = layer_name + "_TRTBIAS"
            else:
                custom_name = layer_name

            # Skip refitting Trilu for now; scalar weights of type int64 value 1 - for clip model
            if layer_name.startswith("onnx::Trilu"):
                continue

            if refit_dict[custom_name] is not None:
                refitter.set_weights(layer_name, weights_role, refit_dict[custom_name])
            else:
                print(f"[W] No refit weights for layer: {layer_name}")

        if not refitter.refit_cuda_engine():
            print("Failed to refit!")
            exit(0)

    def build(
        self,
        onnx_path: str,
        fp16: bool,
        input_profile: Dict[str, list] = None,
        enable_refit: bool = False,
        enable_preview: bool = False,
        enable_all_tactics: bool = False,
        timing_cache=None,
        workspace_size: int = 0,
    ):
        print(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")
        p = Profile()
        if input_profile:
            for name, dims in input_profile.items():
                assert len(dims) == 3
                p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        config_kwargs = {}

        config_kwargs["preview_features"] = [
            trt.PreviewFeature.DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805
        ]
        if enable_preview:
            # Faster dynamic shapes made optional since it increases engine build time.
            config_kwargs["preview_features"].append(
                trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805
            )
        if workspace_size > 0:
            config_kwargs["memory_pool_limits"] = {
                trt.MemoryPoolType.WORKSPACE: workspace_size
            }
        if not enable_all_tactics:
            config_kwargs["tactic_sources"] = []

        engine = engine_from_network(
            network_from_onnx_path(onnx_path),
            config=CreateConfig(
                fp16=fp16,
                refittable=enable_refit,
                profiles=[p],
                load_timing_cache=timing_cache,
                **config_kwargs,
            ),
            save_timing_cache=timing_cache,
        )
        save_engine(engine, path=self.engine_path)

    def load(self):
        print(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self):
        self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict=None, device: torch.device = "cuda"):
        for idx in range(trt_util.get_bindings_per_profile(self.engine)):
            binding = self.engine[idx]
            if shape_dict and binding in shape_dict:
                shape = shape_dict[binding]
            else:
                shape = self.engine.get_binding_shape(binding)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            if self.engine.binding_is_input(binding):
                self.context.set_binding_shape(idx, shape)
            tensor = torch.empty(
                tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]
            ).to(device=device)
            self.tensors[binding] = tensor
            self.buffers[binding] = cuda.DeviceView(
                ptr=tensor.data_ptr(), shape=shape, dtype=dtype
            )

    def infer(self, feed_dict: Dict[str, cuda.DeviceView], stream: cuda.Stream):
        start_binding, _ = trt_util.get_active_profile_bindings(self.context)
        # shallow copy of ordered dict
        device_buffers = copy(self.buffers)
        for name, buf in feed_dict.items():
            assert isinstance(buf, cuda.DeviceView)
            device_buffers[name] = buf
        bindings = [0] * start_binding + [buf.ptr for buf in device_buffers.values()]
        noerror = self.context.execute_async_v2(
            bindings=bindings, stream_handle=stream.ptr
        )
        if not noerror:
            raise ValueError(f"ERROR: inference failed.")

        return self.tensors


def decode_images(images: torch.Tensor):
    images = (
        ((images + 1) * 255 / 2)
        .clamp(0, 255)
        .detach()
        .permute(0, 2, 3, 1)
        .round()
        .type(torch.uint8)
        .cpu()
        .numpy()
    )
    return [Image.fromarray(x) for x in images]


def preprocess_image(image: Image.Image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h))
    init_image = np.array(image).astype(np.float32) / 255.0
    init_image = init_image[None].transpose(0, 3, 1, 2)
    init_image = torch.from_numpy(init_image).contiguous()
    return 2.0 * init_image - 1.0


def prepare_mask_and_masked_image(image: Image.Image, mask: Image.Image):
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32).contiguous() / 127.5 - 1.0
    if isinstance(mask, Image.Image):
        mask = np.array(mask.convert("L"))
        mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask).to(dtype=torch.float32).contiguous()

    masked_image = image * (mask < 0.5)

    return mask, masked_image


def create_models(
    model_id: str,
    use_auth_token: Optional[str],
    device: Union[str, torch.device],
    max_batch_size: int,
    hf_cache_dir: Optional[str] = None,
):
    text_encoder_config = CLIPTextConfig.from_pretrained(
        model_id,
        subfolder="text_encoder",
        use_auth_token=use_auth_token,
        cache_dir=hf_cache_dir,
    )
    unet = diffusers.UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet",
        use_auth_token=use_auth_token,
        cache_dir=hf_cache_dir,
    )
    unet_dim = unet.in_channels
    embedding_dim = text_encoder_config.hidden_size
    models = {
        "clip": CLIP(
            hf_token=use_auth_token,
            device=device,
            path=model_id,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
        ),
        "unet": UNet(
            hf_token=use_auth_token,
            fp16=True,
            device=device,
            path=model_id,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
            unet_dim=unet_dim,
        ),
        "vae": VAE(
            hf_token=use_auth_token,
            device=device,
            path=model_id,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
        ),
        "vae_encoder": VAEEncoder(
            hf_token=use_auth_token,
            device=device,
            path=model_id,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
        ),
    }
    del unet, text_encoder_config
    torch.cuda.empty_cache()
    gc.collect()
    return models


def build_engine(
    engine_path: str,
    onnx_opt_path: str,
    model_data: BaseModel,
    opt_image_height: int,
    opt_image_width: int,
    build_static_batch: bool = False,
    build_dynamic_shape: bool = False,
    build_all_tactics: bool = False,
    build_enable_refit: bool = False,
    build_preview_features: bool = False,
):
    _, free_mem, _ = cudart.cudaMemGetInfo()
    GiB = 2**30
    if free_mem > 6 * GiB:
        activation_carveout = 4 * GiB
        max_workspace_size = free_mem - activation_carveout
    else:
        max_workspace_size = 0
    engine = Engine(engine_path)
    engine.build(
        onnx_opt_path,
        fp16=True,
        input_profile=model_data.get_input_profile(
            1,
            opt_image_height,
            opt_image_width,
            static_batch=build_static_batch,
            static_shape=not build_dynamic_shape,
        ),
        enable_all_tactics=build_all_tactics,
        enable_refit=build_enable_refit,
        enable_preview=build_preview_features,
        workspace_size=max_workspace_size,
    )

    return engine


def export_onnx(
    onnx_path: str,
    model_data: BaseModel,
    opt_image_height: int,
    opt_image_width: int,
    onnx_opset: int,
    hf_cache_dir: Optional[str] = None,
):
    model = model_data.get_model(hf_cache_dir=hf_cache_dir)
    with torch.inference_mode(), torch.autocast("cuda"):
        inputs = model_data.get_sample_input(1, opt_image_height, opt_image_width)
        torch.onnx.export(
            model,
            inputs,
            onnx_path,
            export_params=True,
            opset_version=onnx_opset,
            do_constant_folding=True,
            input_names=model_data.get_input_names(),
            output_names=model_data.get_output_names(),
            dynamic_axes=model_data.get_dynamic_axes(),
        )
    del model
    torch.cuda.empty_cache()
    gc.collect()


def optimize_onnx(
    onnx_path: str,
    onnx_opt_path: str,
    model_data: BaseModel,
):
    onnx_opt_graph = model_data.optimize(onnx.load(onnx_path))
    onnx.save(onnx_opt_graph, onnx_opt_path)
    del onnx_opt_graph
    torch.cuda.empty_cache()
    gc.collect()

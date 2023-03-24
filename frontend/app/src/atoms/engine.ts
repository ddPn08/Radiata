import type { BuildEngineOptions } from 'internal:api'
import { atom } from 'jotai'

const OPTIONS: Required<BuildEngineOptions> = {
    hf_token: '',
    subfolder: '',
    max_batch_size: 1,
    opt_image_height: 512,
    opt_image_width: 512,
    min_latent_resolution: 256,
    max_latent_resolution: 1024,
    build_enable_refit: true,
    build_static_batch: false,
    build_dynamic_shape: true,
    build_all_tactics: false,
    build_preview_features: false,
    onnx_opset: 16,
    force_engine_build: false,
    force_onnx_export: false,
    force_onnx_optimize: false,
}

export const buildEngineOptions = atom(OPTIONS)

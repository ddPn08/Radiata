import { BuildRequest } from 'internal:api'
import { atom } from 'jotai'

export const engineFormAtom = atom<BuildRequest>({
    model_id: '',
    subfolder: '',
    hf_token: '',
    fp16: false,
    opt_image_height: 512,
    opt_image_width: 512,
    max_batch_size: 1,
    onnx_opset: 16,
    build_static_batch: false,
    build_dynamic_shape: true,
    build_preview_features: false,
    force_engine_build: false,
    force_onnx_export: false,
    force_onnx_optimize: false,
    onnx_minimal_optimization: false,
})

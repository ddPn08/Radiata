import gradio as gr

from modules import model_manager
from modules.model import DiffusersModel


def model_list_str():
    return [x.model_id for x in model_manager.sd_models]


def change_model(model_id: str):
    if model_id not in model_list_str():
        raise ValueError("Model not found.")
    model_manager.set_model(model_id)
    return model_id


def add_model(model_id: str):
    if model_id not in model_list_str():
        searched = model_manager.search_model(model_id)
        if len(searched) < 1:
            raise ValueError("Model not found.")
        model_manager.add_model(model_id)
    return gr.Dropdown.update(choices=model_list_str())


def diffusers_ui():
    model_id = (
        model_manager.sd_model.model_id if model_manager.sd_model is not None else None
    )
    with gr.Row(visible=model_manager.mode == "diffusers") as row:
        with gr.Column(scale=0.25):
            with gr.Row():
                model_id_dropdown = gr.Dropdown(
                    value=model_id,
                    choices=model_list_str(),
                    show_label=False,
                )
                reload_models_button = gr.Button("🔄", elem_classes=["tool-button"])

        with gr.Column(scale=0.25):
            with gr.Row():
                add_model_textbox = gr.Textbox(
                    placeholder="Add model",
                    show_label=False,
                )
                add_model_button = gr.Button("💾", elem_classes=["tool-button"])

    model_id_dropdown.change(
        fn=change_model, inputs=[model_id_dropdown], outputs=[model_id_dropdown]
    )
    reload_models_button.click(
        fn=lambda: gr.Dropdown.update(
            choices=model_list_str(),
            value=model_manager.sd_model.model_id
            if model_manager.sd_model is not None
            else None,
        ),
        inputs=[],
        outputs=[model_id_dropdown],
    )
    add_model_button.click(
        fn=add_model, inputs=[add_model_textbox], outputs=[model_id_dropdown]
    )

    return row


def tensorrt_ui():
    def tensorrt_models():
        return [x for x in model_manager.sd_models if x.trt_available()]

    model_id = (
        model_manager.sd_model.model_id
        if model_manager.sd_model is not None and model_manager.sd_model.trt_available()
        else None
    )
    with gr.Row(visible=model_manager.mode == "tensorrt") as row:
        with gr.Column(scale=0.25):
            with gr.Row():
                model_id_dropdown = gr.Dropdown(
                    value=model_id,
                    choices=[x.model_id for x in tensorrt_models()],
                    show_label=False,
                )
                reload_models_button = gr.Button("🔄", elem_classes=["tool-button"])

                reload_models_button.click(
                    fn=lambda: gr.Dropdown.update(
                        choices=[x.model_id for x in tensorrt_models()],
                        value=model_manager.sd_model.model_id
                        if model_manager.sd_model is not None
                        else None,
                    ),
                    inputs=[],
                    outputs=[model_id_dropdown],
                )
                model_id_dropdown.change(
                    fn=change_model,
                    inputs=[model_id_dropdown],
                    outputs=[model_id_dropdown],
                )

    return row


def deepfloyd_if_ui():
    with gr.Row(visible=model_manager.mode == "deepfloyd_if") as box:
        stage_1_textbox = gr.Textbox(
            placeholder="Stage 1", show_label=False, value="DeepFloyd/IF-I-L-v1.0"
        )
        stage_2_textbox = gr.Textbox(
            placeholder="Stage 2", show_label=False, value="DeepFloyd/IF-II-L-v1.0"
        )
        stage_3_textbox = gr.Textbox(
            placeholder="Stage 3",
            show_label=False,
            value="stabilityai/stable-diffusion-x4-upscaler",
        )
        mode = gr.Radio(
            choices=["auto", "lowvram", "medvram", "highvram"], value="auto"
        )
        apply_button = gr.Button("💾", elem_classes=["tool-button"])

    def apply(stage_1: str, stage_2: str, stage_3: str, mode: str):
        model = DiffusersModel(
            model_id="deepfloyd_if",
            IF_model_id_1=stage_1,
            IF_model_id_2=stage_2,
            IF_model_id_3=stage_3,
        )
        model_manager._set_model(model, load_after=True)
        model_manager.runner.mode = mode
        return stage_1, stage_2, stage_3, mode

    apply_button.click(
        fn=apply,
        inputs=[stage_1_textbox, stage_2_textbox, stage_3_textbox, mode],
        outputs=[stage_1_textbox, stage_2_textbox, stage_3_textbox, mode],
    )

    return box


def ui():
    with gr.Box():
        with gr.Row():
            with gr.Column(scale=3):
                diffusers = diffusers_ui()
                tensorrt = tensorrt_ui()
                deepfloyd_if = deepfloyd_if_ui()
            with gr.Row():
                mode = gr.Radio(
                    choices=model_manager.available_mode,
                    value=model_manager.mode,
                    show_label=False,
                )
                reload_button = gr.Button(
                    "🔄",
                    elem_classes=["tool-button"],
                    elem_id="inference-mode-reload-button",
                )

    def reload():
        return (
            model_manager.mode,
            gr.Row.update(visible=model_manager.mode == "diffusers"),
            gr.Row.update(visible=model_manager.mode == "tensorrt"),
            gr.Box.update(visible=model_manager.mode == "deepfloyd_if"),
        )

    def on_change(mode: str):
        model_manager.set_mode(mode)
        return (
            mode,
            gr.Row.update(visible=mode == "diffusers"),
            gr.Row.update(visible=mode == "tensorrt"),
            gr.Box.update(visible=mode == "deepfloyd_if"),
        )

    reload_button.click(
        fn=reload,
        inputs=[],
        outputs=[mode, diffusers, tensorrt, deepfloyd_if],
    )

    mode.change(
        fn=on_change,
        inputs=[mode],
        outputs=[mode, diffusers, tensorrt, deepfloyd_if],
    )

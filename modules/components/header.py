import gradio as gr

from modules import model_manager


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


def ui():
    with gr.Box():
        with gr.Row():
            diffusers = diffusers_ui()
            tensorrt = tensorrt_ui()
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
        )

    def on_change(mode: str):
        model_manager.set_mode(mode)
        return (
            mode,
            gr.Row.update(visible=mode == "diffusers"),
            gr.Row.update(visible=mode == "tensorrt"),
        )

    reload_button.click(fn=reload, inputs=[], outputs=[mode, diffusers, tensorrt])

    mode.change(fn=on_change, inputs=[mode], outputs=[mode, diffusers, tensorrt])

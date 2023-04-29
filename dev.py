import modules.ui as ui
import webui

webui.pre_load()
demo = ui.create_ui().queue(64)

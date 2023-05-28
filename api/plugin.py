import inspect


def get_plugin_id(frm=None):
    if frm is None:
        frm = inspect.stack()[1]
    mod = inspect.getmodule(frm[0])
    return mod.__name__.split(".")[1]

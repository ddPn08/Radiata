import inspect


def get_plugin_id():
    frm = inspect.stack()[1]
    mod = inspect.getmodule(frm[0])
    return mod.__name__.split(".")[1]

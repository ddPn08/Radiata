import base64
import io

from PIL import Image


def img2b64(img: Image.Image, format="png"):
    buf = io.BytesIO()
    img.save(buf, format)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return b64


def b642img(img: Image.Image):
    return Image.open(io.BytesIO(base64.b64decode(img)))

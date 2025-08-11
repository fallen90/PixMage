# main.py
# PixMage is a FastAPI-powered image compression and optimization proxy inspired by Bandwidth Hero, built for speed, quality control, and minimal bandwidth usage. It supports dynamic resizing, format conversion, grayscale filtering, and intelligent text/image detection to choose the optimal compression strategy on the fly.
# - Legacy root "/" supports PixMage params and 302-redirects to /compress
# - Main endpoint:
#     /compress?url=<image>
#       [&quality=1-100&format=webp|jpeg|png&grayscale=0|1&max_width&max_height&text=0|1]
#
# SPECIAL BEHAVIOR
# - SUPER mode: quality=40
#   * Text mode is DISABLED by default (unless text=1).
#   * If source width > 850px and caller didn't set max dims, downscale to 75% (LANCZOS).
#   * Iteratively reduce quality to reach ~50% size reduction (for lossy formats).
#
# - DOUBLE mode: quality=95
#   * Text mode is ENABLED by default (unless text=0).
#   * Upscales image by 1.5x (LANCZOS) before encoding.
#   * Bumps quality by ~10% (capped at 100).
#   * NEW: Apply MedianFilter to de-grain, then a slightly stronger UnsharpMask.
#
# - Passthrough: WEBP→WEBP if NOT SUPER/DOUBLE, NOT text/grayscale/resize
#
# Run: uvicorn main:app --host 0.0.0.0 --port 3000

import io
import os
import sys
import base64
import hashlib
from urllib.parse import urlparse, urlunparse, urlencode

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, Response, RedirectResponse
from PIL import Image, ImageOps, ImageStat, ImageFilter

# -----------------------------
# Config
# -----------------------------
DEFAULT_QUALITY = int(os.getenv("DEFAULT_QUALITY", "35"))
DEFAULT_FORMAT = os.getenv("DEFAULT_FORMAT", "webp").lower()  # webp|jpeg|png
DEFAULT_CACHE_SECONDS = int(os.getenv("CACHE_SECONDS", "86400"))
MAX_BYTES = int(os.getenv("MAX_SOURCE_BYTES", str(20 * 1024 * 1024)))  # 20 MB cap
TIMEOUT = float(os.getenv("FETCH_TIMEOUT", "15"))
ALLOW_HTTP = os.getenv("ALLOW_HTTP", "1") == "1"
FORWARD_COOKIES = os.getenv("FORWARD_COOKIES", "1") == "1"
FORWARD_UA = os.getenv("FORWARD_UA", "1") == "1"
FORWARD_REFERER = os.getenv("FORWARD_REFERER", "1") == "1"
DISABLE_ETAGS = os.getenv("DISABLE_ETAGS", "0") == "1"

USERNAME = os.getenv("PROXY_USERNAME")
PASSWORD = os.getenv("PROXY_PASSWORD")

ACCEPTED_FORMATS = {"webp", "jpeg", "jpg", "png"}

app = FastAPI(title="pix-mage-py-fastapi")

client = httpx.Client(
    headers={
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Encoding": "identity",  # avoid double-compression
        "Connection": "close",
    },
    timeout=TIMEOUT
)

# -----------------------------
# Helpers
# -----------------------------
def parse_bool(val, default=False):
    if val is None:
        return default
    return str(val).lower() in ("1", "true", "yes", "on")

def to_int(x, default):
    try:
        return int(x)
    except Exception:
        return default

def safe_origin_url(u: str) -> str | None:
    try:
        p = urlparse(u)
    except Exception:
        return None
    if p.scheme not in ("http", "https"):
        return None
    if p.scheme == "http" and not ALLOW_HTTP:
        return None
    p = p._replace(netloc=p.hostname + (f":{p.port}" if p.port else ""), fragment="")
    return urlunparse(p)

def make_etag_token(payload: bytes) -> str:
    # raw token; we'll format as weak ETag header later
    return hashlib.sha256(payload).hexdigest()[:32]

def etag_header(token: str, weak: bool = True) -> str:
    # Properly quoted per RFC; weak example: W/"<token>"
    return f'W/"{token}"' if weak else f'"{token}"'

def client_forward_headers(req: Request) -> dict:
    headers = {}
    if FORWARD_UA and "user-agent" in req.headers:
        headers["User-Agent"] = req.headers["user-agent"]
    if FORWARD_REFERER and "referer" in req.headers:
        headers["Referer"] = req.headers["referer"]
    if FORWARD_COOKIES and "cookie" in req.headers:
        headers["Cookie"] = req.headers["cookie"]
    if "accept" in req.headers:
        headers["Accept"] = req.headers["accept"]
    return headers

def is_svg(content_type: str, head: bytes) -> bool:
    if content_type and "svg" in content_type.lower():
        return True
    sample = head[:256].lstrip().lower()
    return sample.startswith(b"<svg") or b"<svg" in sample

def fetch_image_bytes(url: str, req: Request) -> tuple[bytes, str]:
    headers = client_forward_headers(req)
    with client.stream("GET", url, headers=headers) as r:
        r.raise_for_status()
        content_type = r.headers.get("Content-Type", "").lower()

        buf = io.BytesIO()
        total = 0
        for chunk in r.iter_bytes(64 * 1024):
            total += len(chunk)
            if total > MAX_BYTES:
                raise HTTPException(status_code=413, detail="Source image too large")
            buf.write(chunk)
    return buf.getvalue(), content_type

def pillow_open_safely(data: bytes) -> Image.Image:
    Image.MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "178956970"))
    img = Image.open(io.BytesIO(data))
    img.load()
    return img

def detect_texty(img: Image.Image) -> bool:
    """
    Heuristic: mark as text/graphics (screenshots, UI, manga bubbles, diagrams)
    if low color complexity AND (edges or strong black/white share).
    """
    try:
        thumb = img.convert("RGB")
        max_side = 512
        w, h = thumb.size
        if max(w, h) > max_side:
            ratio = max_side / float(max(w, h))
            thumb = thumb.resize((max(1, int(w * ratio)), max(1, int(h * ratio))), Image.BOX)

        colors = thumb.getcolors(maxcolors=1000000)  # None if too many
        color_cnt = len(colors) if colors is not None else 1000001

        edges = thumb.convert("L").filter(ImageFilter.FIND_EDGES)
        edge_mean = ImageStat.Stat(edges).mean[0]  # 0..255

        g = thumb.convert("L")
        hist = g.histogram()
        total_px = sum(hist) if hist else 1
        black_px = sum(hist[:8])
        white_px = sum(hist[-8:])
        bw_share = (black_px + white_px) / total_px

        is_low_colors = color_cnt <= 256
        is_edgy = edge_mean >= 10.0
        is_bw_heavy = bw_share >= 0.35

        return (is_low_colors and (is_edgy or is_bw_heavy))
    except Exception:
        return False

def check_basic_auth(req: Request) -> bool:
    if not USERNAME and not PASSWORD:
        return True
    auth = req.headers.get("authorization")
    if not auth or not auth.lower().startswith("basic "):
        return False
    try:
        decoded = base64.b64decode(auth.split(" ", 1)[1]).decode("utf-8")
        user, pw = decoded.split(":", 1)
        return (user == USERNAME and pw == PASSWORD)
    except Exception:
        return False

# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/", include_in_schema=False)
async def legacy_root(request: Request):
    # Support PixMage legacy params:
    # ?jpg=1&l=70&bw=0&url=<src>&w=<max_width>&h=<max_height>
    qp = request.query_params
    if "url" in qp and any(k in qp for k in ("jpg", "l", "bw", "w", "h")):
        q = {}
        q["url"] = qp.get("url", "")
        if qp.get("l"):
            q["quality"] = qp.get("l")
        if qp.get("bw") in ("1", "true", "True"):
            q["grayscale"] = "1"
        if qp.get("jpg") in ("1", "true", "True"):
            q["format"] = "jpeg"
        if qp.get("w"):
            q["max_width"] = qp.get("w")
        if qp.get("h"):
            q["max_height"] = qp.get("h")
        return RedirectResponse(url=f"/compress?{urlencode(q)}", status_code=302)

    example = str(request.base_url).rstrip("/") + "/compress?url=" + \
        base64.urlsafe_b64encode(b"https://upload.wikimedia.org/wikipedia/commons/7/77/Delete_key1.jpg").decode()
    return JSONResponse({
        "name": "pix-mage-py-fastapi",
        "endpoints": {
            "compress": "/compress?url=<image-url>[&quality=1-100&format=webp|jpeg|png&grayscale=0|1&max_width=<px>&max_height=<px>&text=0|1]"
        },
        "example": example,
        "auth": "Set PROXY_USERNAME and PROXY_PASSWORD to enable HTTP Basic Auth"
    })

@app.api_route("/compress", methods=["GET", "HEAD"])
async def compress(request: Request):
    if not check_basic_auth(request):
        return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="Proxy"'})

    qp = request.query_params

    src_url = qp.get("url", "").strip()
    if not src_url:
        raise HTTPException(status_code=400, detail="Missing url parameter")

    src_url = safe_origin_url(src_url)
    if not src_url:
        raise HTTPException(status_code=400, detail="Invalid or unsupported URL")

    out_format = qp.get("format", DEFAULT_FORMAT).lower()
    if out_format == "jpg":
        out_format = "jpeg"
    if out_format not in ACCEPTED_FORMATS:
        out_format = DEFAULT_FORMAT

    requested_quality = to_int(qp.get("quality"), DEFAULT_QUALITY)
    requested_quality = min(100, max(1, requested_quality))
    grayscale = parse_bool(qp.get("grayscale"), False)
    text_param_raw = qp.get("text")  # raw string so we can detect explicit "1"/"0"
    text_flag = parse_bool(text_param_raw, False)
    max_w = max(0, to_int(qp.get("max_width"), 0))
    max_h = max(0, to_int(qp.get("max_height"), 0))

    # Modes
    super_mode = (requested_quality == 40)   # aggressive compression
    double_mode = (requested_quality == 95)  # upscale & enhance

    # HEAD: just probe upstream
    if request.method == "HEAD":
        try:
            head = client.head(src_url, headers=client_forward_headers(request), follow_redirects=True)
            head.raise_for_status()
        except Exception:
            return Response(status_code=404)
        return Response(status_code=200)

    # Fetch
    try:
        data, content_type = fetch_image_bytes(src_url, request)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {type(e).__name__}")

    # SVG passthrough
    if is_svg(content_type, data[:512]):
        payload = data
        mime = "image/svg+xml"
        headers = {
            "Cache-Control": f"public, max-age={DEFAULT_CACHE_SECONDS}",
            "X-Compressed-By": "pix-mage-py",
            "X-Original-Content-Type": content_type or "",
            "X-Original-Size": str(len(data)),
            "X-Compressed-Size": str(len(payload)),
        }
        if not DISABLE_ETAGS:
            token = make_etag_token(payload)
            etag = etag_header(token, weak=True)
            if request.headers.get("if-none-match", "") and token in request.headers.get("if-none-match", ""):
                return Response(status_code=304)
            headers["ETag"] = etag
        return Response(content=payload, media_type=mime, headers=headers)

    if not (content_type or "").startswith("image/"):
        raise HTTPException(status_code=415, detail="Source is not an image")

    # Passthrough: WEBP->WEBP when NOT SUPER/DOUBLE, NOT text/grayscale/resize
    if (not super_mode and not double_mode and not grayscale and text_param_raw is None and
        max_w == 0 and max_h == 0 and out_format == "webp" and
        (content_type or "").startswith("image/webp")):
        payload = data
        mime = "image/webp"
        orig_size = len(data); out_size = len(payload); saved_size = max(0, orig_size - out_size)
        print("\033[1;36m[==] Passthrough WEBP: Original=%dKB, Output=%dKB, Saved=%dKB\033[0m" %
              (orig_size // 1024, out_size // 1024, saved_size // 1024))
        headers = {
            "Cache-Control": f"public, max-age={DEFAULT_CACHE_SECONDS}",
            "X-Compressed-By": "pix-mage-py",
            "X-Original-Content-Type": content_type or "",
            "X-Original-Size": str(orig_size),
            "X-Compressed-Size": str(out_size),
        }
        if not DISABLE_ETAGS:
            token = make_etag_token(payload)
            etag = etag_header(token, weak=True)
            if request.headers.get("if-none-match", "") and token in request.headers.get("if-none-match", ""):
                return Response(status_code=304)
            headers["ETag"] = etag
        return Response(content=payload, media_type=mime, headers=headers)

    # Decode
    try:
        img = pillow_open_safely(data)
    except Exception:
        payload = data
        mime = content_type or "application/octet-stream"
        headers = {"Cache-Control": f"public, max-age={DEFAULT_CACHE_SECONDS}"}
        if not DISABLE_ETAGS:
            token = make_etag_token(payload)
            etag = etag_header(token, weak=True)
            if request.headers.get("if-none-match", "") and token in request.headers.get("if-none-match", ""):
                return Response(status_code=304)
            headers["ETag"] = etag
        return Response(content=payload, media_type=mime, headers=headers)

    # -----------------------------
    # Text-mode decision
    # - DOUBLE: default ON unless text=0
    # - SUPER : default OFF unless text=1
    # - NORMAL: auto-detect unless explicitly set
    # -----------------------------
    if double_mode:
        if text_param_raw == "0":
            text_mode = False
        elif text_param_raw == "1":
            text_mode = True
        else:
            text_mode = True  # default ON
    elif super_mode:
        text_mode = (text_param_raw == "1") or (text_param_raw is None and detect_texty(img))
    else:
        auto_text = detect_texty(img) if text_param_raw is None else text_flag
        text_mode = text_flag if text_param_raw is not None else auto_text

    # Flatten animation
    try:
        if getattr(img, "is_animated", False):
            img.seek(0)
            img = img.convert("RGBA")
    except Exception:
        pass

    # Grayscale
    if grayscale:
        if img.mode in ("RGBA", "LA"):
            alpha = img.split()[-1]
            base = ImageOps.grayscale(img.convert("RGB"))
            img = Image.merge("LA", (base, alpha))
        else:
            img = ImageOps.grayscale(img.convert("RGB"))

    # DOUBLE mode: upscale by 1.5x, bump quality ~10% (cap 100)
    quality = requested_quality
    if double_mode:
        iw, ih = img.size
        new_w = max(1, int(iw * 1.5))
        new_h = max(1, int(ih * 1.5))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        quality = min(int(quality * 1.1), 100)
        print(f"\033[1;35m[++] DOUBLE mode: upscaled {iw}x{ih} -> {new_w}x{new_h}, quality={quality}\033[0m")
        # NEW: MedianFilter to reduce grain before sharpening
        try:
            img = img.filter(ImageFilter.MedianFilter(size=3))
        except Exception:
            pass
        # NEW: Slightly stronger UnsharpMask for cleaner edges/text without halos
        try:
            img = img.filter(ImageFilter.UnsharpMask(radius=0.8, percent=150, threshold=3))
        except Exception:
            pass

    # SUPER downscale if width > 850px and no explicit max dims
    if super_mode and max_w == 0 and max_h == 0:
        iw, ih = img.size
        if iw > 850:
            factor = 0.75
            target_w = max(1, int(iw * factor))
            target_h = max(1, int(ih * factor))
            img = img.resize((target_w, target_h), resample=Image.LANCZOS)

    # If explicit resize hints are provided, apply them
    if max_w > 0 or max_h > 0:
        iw, ih = img.size
        target_w = max_w if max_w > 0 else iw
        target_h = max_h if max_h > 0 else ih
        if text_mode:
            img = img.resize((target_w, target_h), resample=Image.BOX)
        else:
            img.thumbnail((target_w, target_h), Image.LANCZOS)

    # -----------------------------
    # Encode
    # -----------------------------
    orig_size = len(data)
    if super_mode and not text_mode:
        quality = min(quality, 30)  # start aggressive

    def encode_once(img_obj, fmt, q_val, for_text: bool):
        """Encode once and return (bytes, mime)."""
        buf = io.BytesIO()
        m = None
        kwargs = {}

        if fmt == "jpeg":
            base = img_obj
            if base.mode in ("RGBA", "LA", "P"):
                bg = Image.new("RGB", base.size, (255, 255, 255))
                if base.mode in ("RGBA", "LA"):
                    alpha = base.split()[-1]
                    bg.paste(base.convert("RGB"), mask=alpha)
                else:
                    bg.paste(base.convert("RGB"))
                base = bg
            else:
                base = base.convert("RGB")
            kwargs.update(dict(optimize=True, progressive=True))
            if for_text:
                kwargs["quality"] = max(80, q_val)
                kwargs["subsampling"] = 0  # 4:4:4
            else:
                kwargs["quality"] = q_val
                if super_mode:
                    kwargs["subsampling"] = 2  # 4:2:0
            base.save(buf, format="JPEG", **kwargs)
            m = "image/jpeg"

        elif fmt == "png":
            base = img_obj
            if for_text:
                try:
                    base = base.convert("RGB").quantize(colors=64, method=Image.MEDIANCUT, dither=Image.NONE)
                except Exception:
                    base = base.convert("RGBA")
            else:
                if base.mode not in ("RGB", "RGBA", "L", "LA", "P"):
                    base = base.convert("RGBA")
            kwargs.update(dict(optimize=True))
            base.save(buf, format="PNG", **kwargs)
            m = "image/png"

        else:
            # WEBP
            base = img_obj
            if base.mode in ("P",):
                base = base.convert("RGBA")
            if for_text:
                kwargs.update(dict(lossless=True, method=6))
            else:
                kwargs.update(dict(quality=q_val, method=6, exact=False))
            base.save(buf, format="WEBP", **kwargs)
            m = "image/webp"

        return buf.getvalue(), m

    # Encode once with current mode
    try:
        payload, mime = encode_once(img, out_format, quality, text_mode)
    except Exception:
        payload, mime = data, (content_type or "application/octet-stream")

    # If SUPER and NOT text_mode, try to reach ~50% of original by lowering quality
    if super_mode and not text_mode:
        target_size = int(orig_size * 0.50)
        if out_format == "png":
            if len(payload) > target_size:
                try:
                    img_q = img.convert("RGB").quantize(colors=32, method=Image.MEDIANCUT, dither=Image.NONE)
                    payload, mime = encode_once(img_q, "png", quality, False)
                except Exception:
                    pass
        else:
            floor = 10
            step = 5
            q_iter = quality
            while len(payload) > target_size and q_iter > floor:
                q_iter = max(floor, q_iter - step)
                try:
                    payload, mime = encode_once(img, out_format, q_iter, False)
                except Exception:
                    break

    # Logging
    out_size = len(payload)
    saved_size = max(0, orig_size - out_size)
    ratio = (out_size / orig_size) if orig_size else 1.0
    tag = ("DOUBLE" if double_mode else ("SUPER" if super_mode else ("TEXT" if text_mode else "STD")))
    print(
        ("\033[1;32m[>>] %s Image compressed: Original=%dKB, Output=%dKB, Saved=%dKB, "
         "Ratio=%.2fx\033[0m") %
        (tag, orig_size // 1024, out_size // 1024, saved_size // 1024, ratio)
    )

    # Build response
    headers = {
        "Cache-Control": f"public, max-age={DEFAULT_CACHE_SECONDS}",
        "X-Compressed-By": "pix-mage-py",
        "X-Original-Content-Type": content_type or "",
        "X-Original-Size": str(orig_size),
        "X-Compressed-Size": str(out_size),
        "X-Mode": tag,
    }
    if not DISABLE_ETAGS:
        token = make_etag_token(payload)
        etag = etag_header(token, weak=True)
        if request.headers.get("if-none-match", "") and token in request.headers.get("if-none-match", ""):
            return Response(status_code=304)
        headers["ETag"] = etag

    return Response(content=payload, media_type=mime, headers=headers)


# -----------------------------
# Entrypoint note (run with uvicorn)
# -----------------------------
if __name__ == "__main__":
    if "webp" not in Image.registered_extensions().values():
        print("Warning: Pillow WebP support not detected. Ensure libwebp is installed.", file=sys.stderr)
    import uvicorn
    uvicorn.run("main:app", host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "3000")), reload=False)

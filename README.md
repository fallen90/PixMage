# PixMage

**PixMage** is a **FastAPI-powered image compression and optimization proxy** inspired by *Bandwidth Hero*.  
It reduces bandwidth usage by dynamically resizing, reformatting, and optimizing images in real time — without sacrificing visible quality.  
PixMage includes special modes for aggressive compression and enhanced upscaling, making it perfect for low-bandwidth environments and image-heavy apps.

---

## ✨ Features

- **SUPER mode** – Aggressive compression with smart downscaling for large images.
- **DOUBLE mode** – Upscales images with grain reduction and enhanced sharpening for crisp text/images.
- **Automatic text/graphics detection** – Adapts compression for screenshots, manga, UI elements, and diagrams.
- **Passthrough mode** – Zero re-encoding for already-optimized WebP files.
- Supports **WebP**, **JPEG**, and **PNG** output formats.
- Optional **grayscale** conversion.
- Configurable **HTTP Basic Auth** for restricted access.
- Fully configurable via environment variables.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
````

### 2. Run the server

```bash
uvicorn app:app --host 0.0.0.0 --port 3000
```

### 3. Example request

```bash
curl "http://localhost:3000/compress?url=https://example.com/image.jpg&quality=40&format=webp"
```

---

## ⚙️ Environment Variables

| Variable           | Default    | Description                                    |
| ------------------ | ---------- | ---------------------------------------------- |
| `DEFAULT_QUALITY`  | `35`       | Default compression quality (1–100).           |
| `DEFAULT_FORMAT`   | `webp`     | Default output format (`webp`, `jpeg`, `png`). |
| `CACHE_SECONDS`    | `86400`    | Cache duration in seconds.                     |
| `MAX_SOURCE_BYTES` | `20971520` | Max source file size in bytes (20 MB default). |
| `FETCH_TIMEOUT`    | `15`       | Timeout for fetching images in seconds.        |
| `ALLOW_HTTP`       | `1`        | Allow HTTP (non-HTTPS) sources.                |
| `FORWARD_COOKIES`  | `1`        | Forward client cookies to source.              |
| `FORWARD_UA`       | `1`        | Forward client User-Agent header.              |
| `FORWARD_REFERER`  | `1`        | Forward client Referer header.                 |
| `DISABLE_ETAGS`    | `0`        | Disable ETag generation.                       |
| `PROXY_USERNAME`   | *(empty)*  | Username for HTTP Basic Auth.                  |
| `PROXY_PASSWORD`   | *(empty)*  | Password for HTTP Basic Auth.                  |

---

## 🖥 Special Modes

### SUPER Mode

* Activated with `quality=40`
* Text mode disabled by default (unless `text=1`)
* Downscales large images (> 850px wide) to 75% of size
* Iteratively reduces quality to target \~50% original size

### DOUBLE Mode

* Activated with `quality=95`
* Text mode enabled by default (unless `text=0`)
* Upscales image by 1.5x (LANCZOS)
* Applies **MedianFilter** to remove grain
* Applies **UnsharpMask** (radius=0.8, percent=150, threshold=3) for cleaner edges/text
* Increases quality by \~10% (max 100)

---

## 🔒 Authentication

To enable HTTP Basic Auth:

```bash
export PROXY_USERNAME=myuser
export PROXY_PASSWORD=mypass
```

Requests without valid credentials will return `401 Unauthorized`.

---

## 📜 License

MIT License © 2025
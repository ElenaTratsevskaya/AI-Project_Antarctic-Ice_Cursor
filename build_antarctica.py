"""Build a responsive HTML animation of Antarctic ice-surface height change."""

import base64
import json
import math
import os
from datetime import datetime, timedelta

import h5py
import numpy as np
from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "antarctica")
FRAMES = os.path.join(OUT, "frames")
STEP = 20000.0
SCALE_M = 5.0
PIXELS = 5
EPOCH = datetime(2018, 1, 1)
ROSE = np.array([27, 73, 129], dtype=np.float32)
ZERO = np.array([244, 247, 251], dtype=np.float32)
FELL = np.array([176, 32, 48], dtype=np.float32)


def tile_files():
    return [
        os.path.join(ROOT, name)
        for name in sorted(os.listdir(ROOT))
        if name.startswith("ATL15_A") and name.endswith(".nc")
    ]


def load_mosaic(paths):
    tiles = []
    times = None
    for path in paths:
        with h5py.File(path, "r") as handle:
            x = handle["delta_h/x"][:].astype(np.float64)
            y = handle["delta_h/y"][:].astype(np.float64)
            height = handle["delta_h/delta_h"][:].astype(np.float32)
            ice = handle["delta_h/ice_area"][:].astype(np.float32)
            time = handle["delta_h/time"][:].astype(np.float64)
            if times is None:
                times = time
            elif not np.array_equal(times, time):
                raise SystemExit("Time axes differ between Antarctic tiles")
            valid = (height < 1e30) & (ice > 0) & (ice < 1e30)
            height = np.where(valid, height, np.nan).astype(np.float32)
            tiles.append((x, y, height))

    x0 = min(float(x.min()) for x, _, _ in tiles)
    x1 = max(float(x.max()) for x, _, _ in tiles)
    y0 = min(float(y.min()) for _, y, _ in tiles)
    y1 = max(float(y.max()) for _, y, _ in tiles)
    nx = int(round((x1 - x0) / STEP)) + 1
    ny = int(round((y1 - y0) / STEP)) + 1
    mosaic = np.full((len(times), ny, nx), np.nan, dtype=np.float32)
    for x, y, height in tiles:
        cols = np.rint((x - x0) / STEP).astype(int)
        rows = np.rint((y1 - y) / STEP).astype(int)
        mosaic[:, rows[:, None], cols[None, :]] = height
    change = mosaic - mosaic[0]
    return change, times, (x0, y1)


def _neighbor_median(frame, cells):
    height, width = frame.shape
    ring = []
    for y, x in cells:
        for ny in range(y - 1, y + 2):
            for nx in range(x - 1, x + 2):
                if 0 <= ny < height and 0 <= nx < width and np.isfinite(frame[ny, nx]):
                    ring.append(float(frame[ny, nx]))
    if not ring:
        return None
    return float(np.median(ring))


def fill_gaps(frame):
    """Paint enclosed gaps with the ice color beside each one."""
    height, width = frame.shape
    ocean = np.zeros(frame.shape, dtype=bool)
    stack = []
    for x in range(width):
        stack.append((0, x))
        stack.append((height - 1, x))
    for y in range(height):
        stack.append((y, 0))
        stack.append((y, width - 1))
    while stack:
        y, x = stack.pop()
        if y < 0 or x < 0 or y >= height or x >= width:
            continue
        if ocean[y, x] or np.isfinite(frame[y, x]):
            continue
        ocean[y, x] = True
        stack.extend(((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)))

    filled = frame.copy()
    hole = ~np.isfinite(frame) & ~ocean
    seen = np.zeros(frame.shape, dtype=bool)
    ys, xs = np.where(hole)
    for start_y, start_x in zip(ys, xs):
        if seen[start_y, start_x]:
            continue
        cells = []
        stack = [(int(start_y), int(start_x))]
        while stack:
            y, x = stack.pop()
            if y < 0 or x < 0 or y >= height or x >= width or seen[y, x] or not hole[y, x]:
                continue
            seen[y, x] = True
            cells.append((y, x))
            stack.extend(((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)))
        color = _neighbor_median(frame, cells)
        if color is None:
            continue
        for y, x in cells:
            filled[y, x] = color

    for _ in range(8):
        finite = np.isfinite(filled).astype(np.uint8)
        pad = np.pad(finite, 1, constant_values=0)
        neighbors = (
            pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
            + pad[:-2, :-2] + pad[:-2, 2:] + pad[2:, :-2] + pad[2:, 2:]
        )
        candidates = np.argwhere(~np.isfinite(filled) & (neighbors >= 6))
        if len(candidates) == 0:
            break
        for y, x in candidates:
            block = filled[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
            values = block[np.isfinite(block)]
            if values.size:
                filled[y, x] = np.median(values)

    ice = np.isfinite(filled)
    grown = ice
    for _ in range(2):
        pad = np.pad(grown, 1, constant_values=False)
        grown = np.zeros_like(ice)
        for dy in range(3):
            for dx in range(3):
                grown |= pad[dy:dy + height, dx:dx + width]
    closed = grown
    for _ in range(2):
        pad = np.pad(closed, 1, constant_values=False)
        closed = np.ones_like(ice)
        for dy in range(3):
            for dx in range(3):
                closed &= pad[dy:dy + height, dx:dx + width]
    for _ in range(6):
        finite = np.isfinite(filled)
        pad = np.pad(finite, 1, constant_values=False)
        touch = np.zeros_like(ice)
        for dy in range(3):
            for dx in range(3):
                if dy == 1 and dx == 1:
                    continue
                touch |= pad[dy:dy + height, dx:dx + width]
        candidates = np.argwhere(closed & ~finite & touch)
        if len(candidates) == 0:
            break
        for y, x in candidates:
            block = filled[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
            values = block[np.isfinite(block)]
            if values.size:
                filled[y, x] = np.median(values)
    return filled


def quiet_blue_squares(frame):
    """Repaint small saturated blue clumps with the color of the ice beside them."""
    height, width = frame.shape
    hot = np.isfinite(frame) & (frame >= 3.0)
    seen = np.zeros(frame.shape, dtype=bool)
    painted = frame.copy()
    ys, xs = np.where(hot)
    for start_y, start_x in zip(ys, xs):
        if seen[start_y, start_x]:
            continue
        cells = []
        stack = [(int(start_y), int(start_x))]
        while stack:
            y, x = stack.pop()
            if y < 0 or x < 0 or y >= height or x >= width or seen[y, x] or not hot[y, x]:
                continue
            seen[y, x] = True
            cells.append((y, x))
            stack.extend(((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)))
        if len(cells) > 12:
            continue
        color = _neighbor_median(frame, cells)
        if color is None or color >= 3.0:
            ring = []
            for y, x in cells:
                for ny in range(y - 1, y + 2):
                    for nx in range(x - 1, x + 2):
                        if (
                            0 <= ny < height
                            and 0 <= nx < width
                            and np.isfinite(frame[ny, nx])
                            and frame[ny, nx] < 3.0
                        ):
                            ring.append(float(frame[ny, nx]))
            if not ring:
                continue
            color = float(np.median(ring))
        for y, x in cells:
            painted[y, x] = color

    for margin, level, limit in ((28, 1.5, 16), (32, 2.5, 6)):
        east = np.zeros(frame.shape, dtype=bool)
        east[:, width - margin :] = np.isfinite(painted[:, width - margin :]) & (
            painted[:, width - margin :] >= level
        )
        seen[:] = False
        ys, xs = np.where(east)
        for start_y, start_x in zip(ys, xs):
            if seen[start_y, start_x]:
                continue
            cells = []
            stack = [(int(start_y), int(start_x))]
            while stack:
                y, x = stack.pop()
                if y < 0 or x < 0 or y >= height or x >= width or seen[y, x] or not east[y, x]:
                    continue
                seen[y, x] = True
                cells.append((y, x))
                stack.extend(((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)))
            if len(cells) > limit:
                continue
            chosen = set(cells)
            ring = []
            for y, x in cells:
                for ny in range(max(0, y - 2), min(height, y + 3)):
                    for nx in range(max(0, x - 2), min(width, x + 3)):
                        if (ny, nx) not in chosen and np.isfinite(painted[ny, nx]) and painted[ny, nx] < level:
                            ring.append(float(painted[ny, nx]))
            if not ring:
                continue
            color = float(np.median(ring))
            for y, x in cells:
                painted[y, x] = color
    return painted


def right_coast_square(change):
    """The same right-coast block is high in most quarters. Keep its cells fixed."""
    height, width = change.shape[1:]
    strong = np.sum(np.isfinite(change) & (change >= 4.0), axis=0)
    mild = np.sum(np.isfinite(change) & (change >= 3.0), axis=0) >= 3
    core = strong >= (change.shape[0] // 2)
    core[:, : int(width * 0.9)] = False
    seen = np.zeros_like(core)
    best = []
    ys, xs = np.where(core)
    for start_y, start_x in zip(ys, xs):
        if seen[start_y, start_x]:
            continue
        cells = []
        stack = [(int(start_y), int(start_x))]
        while stack:
            y, x = stack.pop()
            if y < 0 or x < 0 or y >= height or x >= width or seen[y, x] or not core[y, x]:
                continue
            seen[y, x] = True
            cells.append((y, x))
            stack.extend(((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)))
        if len(cells) > len(best):
            best = cells
    mask = np.zeros_like(core)
    for y, x in best:
        mask[y, x] = True
    for _ in range(2):
        pad = np.pad(mask, 1, constant_values=False)
        near = pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:]
        mask |= near & mild
        mask[:, : int(width * 0.9)] = False
    return mask


def paint_square(frame, mask):
    """Repaint that block in this quarter with the ice color around it."""
    height, width = frame.shape
    pad = np.pad(mask, 1, constant_values=False)
    near = pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:]
    cover = mask | (near & np.isfinite(frame) & (frame >= 2.5))
    painted = frame.copy()
    ring = []
    ys, xs = np.where(cover)
    for y, x in zip(ys, xs):
        for ny in range(max(0, int(y) - 2), min(height, int(y) + 3)):
            for nx in range(max(0, int(x) - 2), min(width, int(x) + 3)):
                if not cover[ny, nx] and np.isfinite(frame[ny, nx]) and frame[ny, nx] < 2.5:
                    ring.append(float(frame[ny, nx]))
    if not ring:
        return painted
    color = float(np.median(ring))
    finite = cover & np.isfinite(frame)
    painted[finite] = color
    return painted


def colorize(frame):
    amount = np.clip(frame / SCALE_M, -1, 1)
    strength = np.abs(amount)[..., None]
    rose = amount >= 0
    rgb = np.empty(frame.shape + (3,), dtype=np.float32)
    rgb[rose] = ZERO + (ROSE - ZERO) * strength[rose]
    rgb[~rose] = ZERO + (FELL - ZERO) * strength[~rose]
    image = np.zeros(frame.shape + (4,), dtype=np.uint8)
    valid = np.isfinite(frame)
    image[valid, :3] = np.clip(rgb[valid], 0, 255).astype(np.uint8)
    image[valid, 3] = 255
    picture = Image.fromarray(image, "RGBA")
    picture = picture.resize(
        (picture.width * PIXELS, picture.height * PIXELS),
        Image.Resampling.NEAREST,
    )
    return smooth_contour(picture)


def smooth_contour(picture):
    """Round the coastline. Interior cells stay flat blocks of color."""
    array = np.array(picture)
    rgb = array[:, :, :3]
    opaque = array[:, :, 3] > 0
    height, width = opaque.shape
    blurred = np.array(
        Image.fromarray((opaque.astype(np.uint8) * 255)).filter(
            ImageFilter.GaussianBlur(radius=2.2)
        )
    )
    smooth = blurred >= 128
    edge = np.array(
        Image.fromarray((smooth.astype(np.uint8) * 255)).filter(
            ImageFilter.GaussianBlur(radius=1.15)
        )
    )
    colored = opaque.copy()
    filled = rgb.copy()
    for _ in range(PIXELS * 3):
        added = False
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                src_y0, dst_y0 = max(0, -dy), max(0, dy)
                src_y1, dst_y1 = height - max(0, dy), height - max(0, -dy)
                src_x0, dst_x0 = max(0, -dx), max(0, dx)
                src_x1, dst_x1 = width - max(0, dx), width - max(0, -dx)
                take = colored[src_y0:src_y1, src_x0:src_x1] & ~colored[dst_y0:dst_y1, dst_x0:dst_x1]
                take &= edge[dst_y0:dst_y1, dst_x0:dst_x1] > 8
                if not np.any(take):
                    continue
                ys, xs = np.nonzero(take)
                filled[ys + dst_y0, xs + dst_x0] = filled[ys + src_y0, xs + src_x0]
                colored[ys + dst_y0, xs + dst_x0] = True
                added = True
        if not added:
            break
    out = np.zeros_like(array)
    out[:, :, :3] = filled
    out[:, :, 3] = edge
    out[edge <= 8, 3] = 0
    return Image.fromarray(out, "RGBA")


def labels_for(times):
    labels = []
    for days in times:
        moment = EPOCH + timedelta(days=float(days))
        labels.append(moment.strftime("%-d %B %Y") if os.name != "nt" else moment.strftime("%#d %B %Y"))
    return labels


def embedded_frames():
    """Pack every frame into the page so one HTML file opens anywhere."""
    sources = []
    index = 0
    while True:
        path = os.path.join(FRAMES, f"frame-{index:02d}.png")
        if not os.path.exists(path):
            break
        with open(path, "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("ascii")
        sources.append("data:image/png;base64," + encoded)
        index += 1
    return json.dumps(sources)


def write_page(labels, x0, y1, columns, rows):
    payload = json.dumps(labels)
    pictures = embedded_frames()
    width = columns * PIXELS
    height = rows * PIXELS
    pole_x = (0 - x0) / STEP * PIXELS
    pole_y = y1 / STEP * PIXELS

    def border(dx, dy):
        limits = []
        if dx:
            limits.append(((width if dx > 0 else 0) - pole_x) / dx)
        if dy:
            limits.append(((height if dy > 0 else 0) - pole_y) / dy)
        span = min(value for value in limits if value > 1)
        return pole_x + span * dx, pole_y + span * dy

    lines = []
    for lon, strong in ((0, True), (45, False), (90, True), (135, False)):
        angle = math.radians(lon)
        dx, dy = math.sin(angle), -math.cos(angle)
        x1, y1 = border(dx, dy)
        x2, y2 = border(-dx, -dy)
        opacity = "0.55" if strong else "0.28"
        stroke = "1.6" if strong else "1"
        lines.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#f4f7fb" stroke-opacity="{opacity}" stroke-width="{stroke}"/>'
        )
    grid_lines = "\n".join(lines)
    pole_x_pct = pole_x / width * 100
    pole_y_pct = pole_y / height * 100
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Antarctic Land Ice Height Change</title>
  <style>
    :root {{
      --bg: #07131c;
      --ink: #e7eef5;
      --muted: #b7c5d2;
      --line: #2c4558;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; height: 100%; }}
    body {{
      min-height: 100dvh;
      display: flex;
      flex-direction: column;
      background: radial-gradient(circle at 50% 30%, #123044 0%, var(--bg) 58%);
      color: var(--ink);
      font-family: "Segoe UI", "Helvetica Neue", sans-serif;
    }}
    header, .controls {{
      padding: clamp(0.7rem, 1.8vw, 1.35rem) clamp(0.9rem, 3vw, 2rem);
    }}
    h1 {{
      margin: 0;
      font-family: Georgia, "Palatino Linotype", Palatino, serif;
      font-size: clamp(0.9rem, 3vw, 1.7rem);
      font-weight: 600;
      letter-spacing: -0.02em;
      line-height: 1.12;
    }}
    .source {{
      margin: 0.6rem 0 0;
      max-width: 68rem;
      color: var(--muted);
      font-size: clamp(0.78rem, 1.4vw, 0.95rem);
      line-height: 1.45;
      overflow-wrap: anywhere;
    }}
    .source a {{ color: #d7e7f6; }}
    main {{
      flex: 1 1 auto;
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      grid-template-rows: minmax(0, 1fr);
      gap: clamp(0.4rem, 1.5vw, 1.2rem);
      align-items: center;
      padding: 0 clamp(0.9rem, 3vw, 2rem) 0.4rem;
      overflow: visible;
    }}
    .map-wrap {{
      min-width: 0;
      min-height: 0;
      height: 100%;
      display: grid;
      place-items: center;
      width: 100%;
      justify-self: stretch;
      container-type: size;
      overflow: visible;
      padding: 1.5rem 5.4rem 0.2rem 0;
      perspective: 1200px;
      perspective-origin: 50% 42%;
    }}
    .map-frame {{
      position: relative;
      aspect-ratio: {width} / {height};
      width: min(100cqw, calc(100cqh * {width} / {height}));
      height: auto;
      transform-origin: 50% 78%;
      transform: rotateX(50deg) rotateZ(-8deg);
      transition: transform 2.6s cubic-bezier(0.22, 0.61, 0.24, 1);
    }}
    .map-frame.settled {{
      transform: rotateX(0deg) rotateZ(0deg);
    }}
    .map-frame img,
    .map-frame svg {{
      display: block;
      width: 100%;
      height: 100%;
    }}
    .map-frame img {{
      filter: drop-shadow(0 28px 18px rgba(0, 0, 0, 0.35));
      transition: filter 2.6s ease;
    }}
    .map-frame.settled img {{
      filter: none;
    }}
    .map-frame svg {{
      position: absolute;
      inset: 0;
      pointer-events: none;
    }}
    .dir {{
      position: absolute;
      z-index: 1;
      color: #f4f7fb;
      font-size: clamp(0.75rem, 1.3vw, 0.88rem);
      letter-spacing: 0.03em;
      line-height: 1.35;
      text-shadow: 0 0 4px #07131c, 0 1px 2px #07131c;
      white-space: nowrap;
      pointer-events: none;
    }}
    .dir.north {{ left: {pole_x_pct:.2f}%; top: 0; transform: translate(-50%, calc(-100% - 0.28rem)); }}
    .dir.east {{ left: 100%; top: {pole_y_pct:.2f}%; transform: translate(0.45rem, -50%); }}
    .dir.west {{ left: 0.4rem; top: {pole_y_pct:.2f}%; transform: translateY(-50%); }}
    .dir.far {{ left: {pole_x_pct:.2f}%; bottom: 0.35rem; transform: translateX(-50%); }}
    .dir.pole {{
      left: {pole_x_pct:.2f}%;
      top: {pole_y_pct:.2f}%;
      transform: translate(-50%, 0.55rem);
    }}
    .legend {{
      display: flex;
      gap: 0.45rem;
      align-items: stretch;
      align-self: center;
      width: 6.8rem;
      height: min(70%, 26rem);
      color: var(--muted);
      font-size: clamp(0.72rem, 1.2vw, 0.82rem);
      line-height: 1.2;
    }}
    .bar {{
      width: 0.8rem;
      flex: 0 0 auto;
      border-radius: 999px;
      background: linear-gradient(#1b4981, #f4f7fb 50%, #b02030);
    }}
    .ticks {{
      display: flex;
      flex: 1 1 auto;
      flex-direction: column;
      justify-content: space-between;
      min-height: 10rem;
    }}
    .controls {{
      display: grid;
      grid-template-columns: auto minmax(8rem, 1fr) auto;
      gap: 0.7rem 0.9rem;
      align-items: center;
      border-top: 1px solid var(--line);
      background: rgba(7, 19, 28, 0.92);
    }}
    button {{
      min-height: 2.75rem;
      min-width: 5.6rem;
      border: 1px solid #9bb6cb;
      border-radius: 999px;
      background: #173044;
      color: var(--ink);
      font: inherit;
      padding: 0 1rem;
      touch-action: manipulation;
    }}
    button:focus-visible, input:focus-visible {{
      outline: 2px solid #f4f7fb;
      outline-offset: 2px;
    }}
    .scale {{
      position: relative;
      min-width: 0;
    }}
    .marks {{
      position: absolute;
      inset: 0;
      pointer-events: none;
    }}
    .mark {{
      position: absolute;
      top: 50%;
      width: 1.6px;
      height: 1.15rem;
      background: rgba(244, 247, 251, 0.55);
      transform: translate(-50%, -50%);
    }}
    .mark span {{
      position: absolute;
      left: 50%;
      bottom: calc(100% + 1px);
      color: #f4f7fb;
      font-size: clamp(calc(0.75rem - 2pt), calc(1.3vw - 2pt), calc(0.88rem - 2pt));
      letter-spacing: 0.02em;
      line-height: 1;
      white-space: nowrap;
      text-shadow: 0 0 4px #07131c, 0 1px 2px #07131c;
    }}
    input[type="range"] {{
      display: block;
      width: 100%;
      min-height: 2.75rem;
      margin: 0;
      accent-color: #9db1ca;
    }}
    .when {{
      margin: 0;
      min-width: 9.5rem;
      text-align: right;
      font-variant-numeric: tabular-nums;
      font-size: clamp(0.9rem, 1.6vw, 1.05rem);
    }}
    .note {{
      grid-column: 1 / -1;
      margin: 0;
      color: var(--muted);
      font-size: clamp(0.75rem, 1.3vw, 0.88rem);
      line-height: 1.35;
    }}
    @media (max-width: 760px) {{
      main {{
        grid-template-columns: minmax(0, 1fr);
        grid-template-rows: minmax(0, 1fr) auto;
        justify-items: center;
      }}
      .legend {{
        width: min(100%, 18rem);
        height: auto;
        flex-direction: column;
        align-items: stretch;
      }}
      .bar {{
        width: 100%;
        height: 0.8rem;
        background: linear-gradient(90deg, #b02030, #f4f7fb 50%, #1b4981);
      }}
      .ticks {{
        display: none;
      }}
      .ends {{
        display: flex;
        justify-content: space-between;
        width: 100%;
      }}
      .controls {{
        grid-template-columns: auto minmax(0, 1fr);
      }}
      .when {{
        grid-column: 1 / -1;
        min-width: 0;
        text-align: left;
      }}
    }}
    @media (min-width: 761px) {{
      .ends {{ display: none; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Antarctic Land Ice Height Change</h1>
    <p class="source">Source: NASA ATLAS/ICESat-2 L3B Gridded Land Ice Height Change (ATL15), Version 006, National Snow and Ice Data Center. Quarterly maps, January 2019–April 2026, 20 km grid. <a href="https://doi.org/10.5067/ATLAS/ATL15.006">https://doi.org/10.5067/ATLAS/ATL15.006</a></p>
  </header>
  <main>
    <div class="map-wrap">
      <div class="map-frame">
        <img id="map" alt="Map of Antarctic land ice height change">
        <svg viewBox="0 0 {width} {height}" aria-hidden="true">
          {grid_lines}
        </svg>
        <span class="dir north">North · 0°</span>
        <span class="dir east">East · 90°E</span>
        <span class="dir west">West · 90°W</span>
        <span class="dir far">180°</span>
        <span class="dir pole">South Pole</span>
      </div>
    </div>
    <div class="legend" aria-hidden="true">
      <div class="bar"></div>
      <div class="ticks">
        <span>+5 m rose</span>
        <span>0</span>
        <span>−5 m fell</span>
      </div>
      <div class="ends">
        <span>−5 m fell</span>
        <span>0</span>
        <span>+5 m rose</span>
      </div>
    </div>
  </main>
  <div class="controls">
    <button id="play" type="button" aria-pressed="false">Play</button>
    <div class="scale">
      <input id="quarter" type="range" min="0" max="{len(labels) - 1}" value="0" aria-label="Quarter">
      <div class="marks" id="marks" aria-hidden="true"></div>
    </div>
    <p class="when" id="when"></p>
    <p class="note">Change in ice-surface height since 1 January 2019. Each cell is 20 km across. Changes stronger than 5 m keep the darkest color.</p>
  </div>
  <script>
    const labels = {payload};
    const map = document.getElementById("map");
    const play = document.getElementById("play");
    const quarter = document.getElementById("quarter");
    const when = document.getElementById("when");
    const marks = document.getElementById("marks");
    const yearMarks = [];
    {{
      let previous = "";
      labels.forEach((label, frameIndex) => {{
        const text = label.slice(-4);
        if (text !== previous) {{
          yearMarks.push({{ text, frameIndex }});
          previous = text;
        }}
      }});
    }}
    const frames = {pictures}.map((src) => {{
      const image = new Image();
      image.src = src;
      return image;
    }});
    const frame = document.querySelector(".map-frame");
    let index = 0;
    let timer = null;
    let laid = false;
    let playWhenFlat = true;

    function show(next) {{
      index = next;
      map.src = frames[index].src;
      map.alt = "Antarctic land ice height change, " + labels[index];
      quarter.value = String(index);
      when.textContent = labels[index];
      placeYears();
    }}

    function placeYears() {{
      const max = Number(quarter.max) || 1;
      const thumb = 16;
      const width = quarter.clientWidth;
      marks.replaceChildren();
      yearMarks.forEach((mark) => {{
        if (mark.frameIndex > index) return;
        const ratio = mark.frameIndex / max;
        const center = thumb / 2 + ratio * Math.max(width - thumb, 0);
        const node = document.createElement("span");
        node.className = "mark";
        const label = document.createElement("span");
        label.textContent = mark.text;
        node.appendChild(label);
        node.style.left = center + "px";
        marks.appendChild(node);
        const half = label.offsetWidth / 2;
        const shift = center < half ? half - center : center > width - half ? width - half - center : 0;
        label.style.transform = "translateX(calc(-50% + " + shift + "px))";
      }});
    }}

    window.addEventListener("resize", placeYears);

    function pause() {{
      clearInterval(timer);
      timer = null;
      play.textContent = "Play";
      play.setAttribute("aria-pressed", "false");
    }}

    function start() {{
      if (timer) return;
      play.textContent = "Pause";
      play.setAttribute("aria-pressed", "true");
      timer = setInterval(() => show((index + 1) % labels.length), 670);
    }}

    function settle() {{
      frame.classList.add("settled");
    }}

    function takeOver() {{
      playWhenFlat = false;
      settle();
      pause();
    }}

    play.addEventListener("click", () => {{
      if (!laid) {{
        playWhenFlat = true;
        settle();
        return;
      }}
      timer ? pause() : start();
    }});
    quarter.addEventListener("input", () => {{
      takeOver();
      show(Number(quarter.value));
    }});
    document.addEventListener("keydown", (event) => {{
      if (event.key === "ArrowRight") {{
        takeOver();
        show((index + 1) % labels.length);
      }}
      if (event.key === "ArrowLeft") {{
        takeOver();
        show((index - 1 + labels.length) % labels.length);
      }}
      if (event.key === " ") {{
        event.preventDefault();
        if (!laid) {{
          playWhenFlat = true;
          settle();
          return;
        }}
        timer ? pause() : start();
      }}
    }});

    frame.addEventListener("transitionend", (event) => {{
      if (event.target !== frame || event.propertyName !== "transform" || laid) return;
      laid = true;
      if (playWhenFlat) start();
    }});

    show(0);
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {{
      settle();
      laid = true;
      start();
    }} else {{
      window.setTimeout(settle, 700);
    }}
  </script>
</body>
</html>
"""
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8", newline="\n") as handle:
        handle.write(html)


def main():
    os.makedirs(FRAMES, exist_ok=True)
    change, times, (x0, y1) = load_mosaic(tile_files())
    names = labels_for(times)
    square = right_coast_square(change)
    for index in range(change.shape[0]):
        frame = quiet_blue_squares(fill_gaps(change[index]))
        picture = colorize(paint_square(frame, square))
        picture.save(os.path.join(FRAMES, f"frame-{index:02d}.png"), optimize=True)
    write_page(names, x0, y1, change.shape[2], change.shape[1])
    print(f"frames {change.shape[0]} grid {change.shape[2]}x{change.shape[1]} labels {names[0]} -> {names[-1]}")


if __name__ == "__main__":
    main()

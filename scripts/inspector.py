"""
Interactive activation inspector for STS-Net v0.2 (ClipClassifier).

Takes a list of video (or .pose) files on the command line, runs each
through the model, and serves a web GUI showing the video alongside
per-frame activation heatmaps for every head, plus the attention trace —
synced to the video playhead. No CSV / dataset metadata required.

Usage:
    python scripts/inspect.py clip1.mp4 clip2.mp4 clip3.pose \\
        --ckpt checkpoints/stsnet_v02.pt
"""

import argparse
import mimetypes
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, Response, jsonify, render_template_string, request

app = Flask(__name__)

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# ---------------------------------------------------------------------------
# Global state (populated in main())
# ---------------------------------------------------------------------------
CLIPS: list[dict] = []          # [{idx, name, video_path, pose_path}]
MODEL   = None
VOCAB: dict = {}                # idx_to_shape/att/motion/cloc/ctype
DEVICE  = torch.device("cpu")
HANDEDNESS = "right"
ACT_CACHE: dict[int, dict] = {}


# ---------------------------------------------------------------------------
# Pose extraction
# ---------------------------------------------------------------------------

def extract_pose(video_path: Path, pose_path: Path) -> bool:
    """Run video_to_pose (MediaPipe Holistic) on a video file. Returns success."""
    print(f"  Extracting pose from {video_path.name}...", end=" ", flush=True)
    t0 = time.time()
    result = subprocess.run(
        ["video_to_pose", "-i", str(video_path), "-o", str(pose_path),
         "--format", "mediapipe"],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"FAILED\n    {result.stderr.decode()[-300:]}")
        return False
    print(f"done ({time.time() - t0:.1f}s)")
    return True


# ---------------------------------------------------------------------------
# Flask routes — API
# ---------------------------------------------------------------------------

@app.route("/api/clips")
def api_clips():
    return jsonify([
        {"idx": c["idx"], "name": c["name"],
         "has_pose": c["pose_path"] is not None,
         "has_video": c["video_path"] is not None}
        for c in CLIPS
    ])


@app.route("/api/clip/<int:idx>/activations")
def api_clip_activations(idx):
    if idx in ACT_CACHE:
        return jsonify(ACT_CACHE[idx])

    if idx < 0 or idx >= len(CLIPS):
        return jsonify({"error": "not found"}), 404
    clip = CLIPS[idx]
    if clip["pose_path"] is None:
        return jsonify({"error": "pose extraction failed for this clip"}), 400
    if MODEL is None:
        return jsonify({"error": "no model loaded"}), 400

    from stsnet.data.pose_io import load_pose_streams
    streams = load_pose_streams(clip["pose_path"], HANDEDNESS, mirror_left=True)
    if streams is None:
        return jsonify({"error": "pose load failed"}), 500

    dom    = torch.from_numpy(streams["dominant"]).unsqueeze(0).to(DEVICE)
    nondom = torch.from_numpy(streams["nondominant"]).unsqueeze(0).to(DEVICE)
    body   = torch.from_numpy(streams["body"]).unsqueeze(0).to(DEVICE)
    face   = torch.from_numpy(streams["face"]).unsqueeze(0).to(DEVICE) \
             if "face" in streams else None

    T = dom.shape[1]
    full_t = torch.tensor([T], dtype=torch.long, device=DEVICE)
    zero_t = torch.zeros(1, dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        frame_feats = MODEL.frame_features(dom, nondom, body, face)  # (1, T, D)
        f = frame_feats[0]

        def _sigmoid_probs(head):
            return torch.sigmoid(head(f)).cpu().numpy()

        heads_probs = {
            "shape":  _sigmoid_probs(MODEL.shape_head),
            "att":    _sigmoid_probs(MODEL.att_head),
            "cloc":   _sigmoid_probs(MODEL.cloc_head),
            "ctype":  _sigmoid_probs(MODEL.ctype_head),
            "motion": _sigmoid_probs(MODEL.motion_head),
            "hand_type": torch.softmax(MODEL.hand_type_head(f), dim=-1).cpu().numpy(),
        }
        if MODEL.has_nondom_shape:
            heads_probs["nondom_shape"] = _sigmoid_probs(MODEL.nondom_shape_head)
        if MODEL.has_nondom_att:
            heads_probs["nondom_att"] = _sigmoid_probs(MODEL.nondom_att_head)

        # Full forward (whole-clip attention window) for the attention trace.
        out = MODEL(dom, nondom, body, face,
                     sign_start=zero_t, sign_end=full_t, lengths=full_t)
        attn = out["attn_weights"][0].cpu().numpy()  # (T,)

    def _round_list(arr):
        return [[round(float(v), 4) for v in row] for row in arr]

    def _round_1d(arr):
        return [round(float(v), 4) for v in arr]

    label_maps = {
        "shape": VOCAB["idx_to_shape"], "att": VOCAB["idx_to_att"],
        "cloc": VOCAB["idx_to_cloc"],   "ctype": VOCAB["idx_to_ctype"],
        "motion": VOCAB["idx_to_motion"],
        "hand_type": {0: "one", 1: "two"},
        "nondom_shape": VOCAB["idx_to_shape"], "nondom_att": VOCAB["idx_to_att"],
    }

    result = {
        "T": T,
        "heads": {
            name: {
                "probs":  _round_list(probs),
                "labels": [label_maps[name].get(i, str(i)) for i in range(probs.shape[1])],
            }
            for name, probs in heads_probs.items()
        },
        "attn": _round_1d(attn),
    }
    ACT_CACHE[idx] = result
    return jsonify(result)


# ---------------------------------------------------------------------------
# Video serving with HTTP Range support (required for <video> seeking)
# ---------------------------------------------------------------------------

@app.route("/video/<int:idx>")
def serve_video(idx):
    if idx < 0 or idx >= len(CLIPS):
        return Response("not found", 404)
    path = CLIPS[idx]["video_path"]
    if path is None or not path.exists():
        return Response("not found", 404)

    content_type = mimetypes.guess_type(str(path))[0] or "video/mp4"
    file_size    = path.stat().st_size
    range_header = request.headers.get("Range")

    if range_header:
        m     = re.match(r"bytes=(\d+)-(\d*)", range_header)
        start = int(m.group(1))
        end   = int(m.group(2)) if m.group(2) else file_size - 1
        end   = min(end, file_size - 1)
        length = end - start + 1

        def _gen_range():
            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        return Response(
            _gen_range(), 206,
            headers={
                "Content-Range":  f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges":  "bytes",
                "Content-Length": str(length),
                "Content-Type":   content_type,
            },
        )

    def _gen_full():
        with open(path, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                yield chunk

    return Response(
        _gen_full(),
        headers={
            "Content-Length": str(file_size),
            "Accept-Ranges":  "bytes",
            "Content-Type":   content_type,
        },
    )


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

HEAD_TITLES = {
    "shape": "Shape", "att": "Attitude", "cloc": "Contact Location",
    "ctype": "Contact Type", "motion": "Motion", "hand_type": "Hand Type",
    "nondom_shape": "Shape (non-dom.)", "nondom_att": "Attitude (non-dom.)",
}
HEAD_ORDER = ["shape", "att", "cloc", "ctype", "motion", "hand_type",
              "nondom_shape", "nondom_att"]
N_SHOW = {"shape": 14, "att": 10, "cloc": 14, "ctype": 4, "motion": 8,
          "hand_type": 2, "nondom_shape": 14, "nondom_att": 10}

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>STS-Net Inspector</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Segoe UI', system-ui, sans-serif;
  background: #1a1a2e;
  color: #e8e8e8;
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* ── header ── */
#header {
  padding: 8px 12px;
  background: #16213e;
  border-bottom: 1px solid #0f3460;
  display: flex;
  align-items: center;
  gap: 10px;
  flex-shrink: 0;
}
#header h1 { font-size: 15px; font-weight: 600; color: #e94560; flex-shrink: 0; }
#search {
  flex: 1;
  background: #0f3460;
  border: 1px solid #1a4080;
  color: #e8e8e8;
  padding: 5px 8px;
  border-radius: 4px;
  font-size: 13px;
}
#search::placeholder { color: #888; }
#count { font-size: 11px; color: #888; flex-shrink: 0; }

/* ── main layout ── */
#main { display: flex; flex: 1; min-height: 0; }

/* ── sidebar ── */
#sidebar { width: 260px; flex-shrink: 0; overflow-y: auto; border-right: 1px solid #0f3460; }
.clip-item { padding: 7px 10px; cursor: pointer; border-bottom: 1px solid #0f3460; font-size: 13px; word-break: break-all; }
.clip-item:hover { background: #16213e; }
.clip-item.active { background: #0f3460; border-left: 3px solid #e94560; }
.clip-item.no-pose { color: #666; cursor: not-allowed; }

/* ── viewer ── */
#viewer {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
  padding: 8px 10px;
  gap: 6px;
  overflow-y: auto;
}
#videoWrap {
  flex-shrink: 0;
  background: #000;
  border-radius: 4px;
  overflow: hidden;
  display: flex;
  justify-content: center;
  max-height: 38vh;
}
video { max-height: 38vh; max-width: 100%; display: block; }

#empty { display: flex; align-items: center; justify-content: center; flex: 1; color: #555; font-size: 14px; }

/* ── activation sections ── */
.act-section { flex-shrink: 0; border: 1px solid #1a3060; border-radius: 5px; overflow: hidden; }
.act-header {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 5px 10px;
  background: #16213e;
  cursor: pointer;
  user-select: none;
  font-size: 12px;
  font-weight: 600;
  color: #c0d0f0;
  border-bottom: 1px solid #1a3060;
}
.act-header:hover { background: #1a2a4a; }
.act-arrow { font-size: 10px; transition: transform 0.15s; }
.act-section.collapsed .act-arrow { transform: rotate(-90deg); }
.act-body {
  max-height: 600px;
  overflow: hidden;
  transition: max-height 0.2s ease;
  background: #12192e;
  padding: 6px 8px;
}
.act-section.collapsed .act-body { max-height: 0; padding: 0 8px; }
.act-loading { font-size: 11px; color: #888; padding: 8px 0; }
.canvas-wrap { width: 100%; }
canvas.act-canvas { width: 100%; display: block; }
</style>
</head>
<body>

<div id="header">
  <h1>STS-Net Inspector</h1>
  <input id="search" type="text" placeholder="filter filenames…" autocomplete="off">
  <span id="count"></span>
</div>

<div id="main">
  <div id="sidebar"></div>

  <div id="viewer">
    <div id="empty">← select a clip</div>

    <div id="videoWrap" style="display:none">
      <video id="video" controls preload="metadata"></video>
    </div>

    <div id="sections"></div>
  </div><!-- #viewer -->
</div><!-- #main -->

<script>
// ── constants ──────────────────────────────────────────────────────────────
const LABEL_W = 90;   // px for row label column
const AXIS_H  = 18;   // px for time axis
const ROW_H   = 14;   // px per class row in heatmap
const N_SHOW  = __N_SHOW__;
const HEAD_ORDER  = __HEAD_ORDER__;
const HEAD_TITLES = __HEAD_TITLES__;

// ── state ──────────────────────────────────────────────────────────────────
let clips      = [];
let activeIdx  = null;
let actData    = null;   // current activations JSON
let headKeys   = [];     // heads present in the current actData, in display order
const collapseState = {};

// ── YlOrRd colormap ────────────────────────────────────────────────────────
function ylOrRd(t) {
  const stops = [
    [1.0,  0.988, 0.910],  // 0.0
    [0.996, 0.769, 0.310], // 0.25
    [0.988, 0.553, 0.349], // 0.5
    [0.890, 0.290, 0.200], // 0.75
    [0.545, 0.0,   0.0  ], // 1.0
  ];
  const scaled = Math.min(1, Math.max(0, t)) * 4;
  const lo = Math.floor(scaled), hi = Math.min(4, lo + 1);
  const f  = scaled - lo;
  const r = stops[lo][0] + f * (stops[hi][0] - stops[lo][0]);
  const g = stops[lo][1] + f * (stops[hi][1] - stops[lo][1]);
  const b = stops[lo][2] + f * (stops[hi][2] - stops[lo][2]);
  return `rgb(${Math.round(r*255)},${Math.round(g*255)},${Math.round(b*255)})`;
}

// ── collapsible sections ───────────────────────────────────────────────────
function toggleSection(key) {
  collapseState[key] = !collapseState[key];
  document.getElementById('sec-' + key).classList.toggle('collapsed', collapseState[key]);
  if (!collapseState[key] && actData) {
    if (key === 'attn') redrawAttn(); else redrawHeatmap(key);
  }
}

// ── clip list ──────────────────────────────────────────────────────────────
async function fetchClips() {
  const r = await fetch('/api/clips');
  clips = await r.json();
  renderList();
}

function renderList() {
  const q  = document.getElementById('search').value.trim().toLowerCase();
  const sb = document.getElementById('sidebar');
  const shown = clips.filter(c => !q || c.name.toLowerCase().includes(q));
  document.getElementById('count').textContent = shown.length + ' / ' + clips.length + ' clips';
  sb.innerHTML = '';
  shown.forEach(c => {
    const el = document.createElement('div');
    el.className = 'clip-item'
      + (activeIdx === c.idx ? ' active' : '')
      + (c.has_pose ? '' : ' no-pose');
    el.textContent = c.name + (c.has_pose ? '' : '  (pose extraction failed)');
    el.onclick = () => { if (c.has_pose) selectClip(c); };
    sb.appendChild(el);
  });
}

document.getElementById('search').addEventListener('input', renderList);

// ── video element ──────────────────────────────────────────────────────────
const video = document.getElementById('video');
video.addEventListener('timeupdate', () => { if (actData) redrawAll(); });

// ── sections DOM (built dynamically from the heads present) ────────────────
function buildSections(keys) {
  const wrap = document.getElementById('sections');
  wrap.innerHTML = '';
  keys.concat(['attn']).forEach(key => {
    if (!(key in collapseState)) collapseState[key] = false;
    const title = key === 'attn' ? 'Attention' : (HEAD_TITLES[key] || key);
    const sec = document.createElement('div');
    sec.className = 'act-section' + (collapseState[key] ? ' collapsed' : '');
    sec.id = 'sec-' + key;
    sec.innerHTML = `
      <div class="act-header" onclick="toggleSection('${key}')">
        <span class="act-arrow">▾</span> ${title}
      </div>
      <div class="act-body">
        <div class="act-loading" id="loading-${key}">Loading…</div>
        <div class="canvas-wrap" id="wrap-${key}" style="display:none"><canvas class="act-canvas" id="canvas-${key}"></canvas></div>
      </div>`;
    wrap.appendChild(sec);
  });
}

// ── clip selection ─────────────────────────────────────────────────────────
async function selectClip(c) {
  activeIdx = c.idx;
  actData   = null;
  renderList();

  document.getElementById('empty').style.display    = 'none';
  document.getElementById('videoWrap').style.display = c.has_video ? 'flex' : 'none';

  if (c.has_video) {
    video.src = '/video/' + c.idx;
    video.load();
  } else {
    video.removeAttribute('src');
  }

  // Placeholder sections until we know which heads this checkpoint has.
  buildSections(headKeys.length ? headKeys : HEAD_ORDER);

  try {
    const ra = await fetch('/api/clip/' + c.idx + '/activations');
    if (!ra.ok) {
      const err = await ra.json();
      document.querySelectorAll('.act-loading').forEach(el => el.textContent = 'Error: ' + (err.error || ra.status));
      return;
    }
    actData  = await ra.json();
    headKeys = HEAD_ORDER.filter(k => k in actData.heads);
    buildSections(headKeys);
    headKeys.concat(['attn']).forEach(key => {
      document.getElementById('loading-' + key).style.display = 'none';
      document.getElementById('wrap-' + key).style.display = 'block';
    });
    redrawAll();
  } catch (e) {
    document.querySelectorAll('.act-loading').forEach(el => el.textContent = 'Error: ' + e.message);
  }
}

// ── drawing helpers ────────────────────────────────────────────────────────

function drawTimeAxis(ctx, W, H, axisY, T) {
  ctx.fillStyle = '#0a1a32';
  ctx.fillRect(LABEL_W, axisY, W - LABEL_W, AXIS_H);

  const plotW = W - LABEL_W;
  const dur   = T;

  const candidates = [1, 2, 5, 10, 20, 50, 100, 200];
  let interval = candidates[candidates.length - 1];
  for (const iv of candidates) {
    if (iv / dur * plotW >= 40) { interval = iv; break; }
  }

  ctx.textAlign    = 'center';
  ctx.textBaseline = 'middle';
  for (let t = 0; t <= dur; t += interval) {
    const x = LABEL_W + t / dur * plotW;
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth   = 1;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, axisY); ctx.stroke();
    ctx.strokeStyle = 'rgba(255,255,255,0.35)';
    ctx.lineWidth   = 1;
    ctx.beginPath(); ctx.moveTo(x, axisY); ctx.lineTo(x, axisY + 5); ctx.stroke();
    ctx.fillStyle = 'rgba(200,200,200,0.6)';
    ctx.font = '9px system-ui';
    ctx.fillText(String(t), x, axisY + AXIS_H / 2 + 1);
  }
}

function drawPlayhead(ctx, H, axisY, W) {
  if (!video.duration) return;
  const plotW = W - LABEL_W;
  const x = LABEL_W + (video.currentTime / video.duration) * plotW;
  ctx.strokeStyle = 'rgba(233,69,96,0.4)';
  ctx.lineWidth   = 1;
  ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
  ctx.strokeStyle = '#e94560';
  ctx.lineWidth   = 1.5;
  ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, axisY); ctx.stroke();
  ctx.fillStyle = '#e94560';
  ctx.beginPath();
  ctx.moveTo(x - 4, 0); ctx.lineTo(x + 4, 0); ctx.lineTo(x, 6);
  ctx.closePath(); ctx.fill();
}

// Pick which class rows to show: highest max-activation first, up to nShow.
function pickRows(probs, nShow, pinFirst) {
  const T = probs.length;
  const C = probs[0].length;
  const maxAct = new Array(C).fill(0);
  for (let t = 0; t < T; t++)
    for (let c = 0; c < C; c++)
      if (probs[t][c] > maxAct[c]) maxAct[c] = probs[t][c];

  const pinned = pinFirst != null ? [pinFirst] : [];
  const others = [];
  for (let c = 0; c < C; c++)
    if (c !== pinFirst) others.push(c);
  others.sort((a, b) => maxAct[b] - maxAct[a]);

  return [...pinned, ...others].slice(0, nShow);
}

function drawHeatmap(canvasEl, headData, T) {
  const headKey = canvasEl.id.replace('canvas-', '');
  const nShow   = N_SHOW[headKey] || 14;
  const probs   = headData.probs;   // (T, C)
  const labels  = headData.labels;  // C

  // pin "none" (index 0) as first row for cloc/ctype/motion so the
  // no-activity case is always visible.
  const pinFirst = (headKey === 'motion' || headKey === 'cloc' || headKey === 'ctype') ? 0 : null;
  const rows = pickRows(probs, nShow, pinFirst);

  const plotH = rows.length * ROW_H;
  const totalH = plotH + AXIS_H;
  const W = canvasEl.width = canvasEl.offsetWidth;
  canvasEl.height = totalH;

  const ctx   = canvasEl.getContext('2d');
  const plotW = W - LABEL_W;
  ctx.clearRect(0, 0, W, totalH);

  const cellW = plotW / T;
  rows.forEach((ci, ri) => {
    const y0 = ri * ROW_H;
    for (let t = 0; t < T; t++) {
      ctx.fillStyle = ylOrRd(probs[t][ci]);
      ctx.fillRect(LABEL_W + t * cellW, y0, Math.ceil(cellW), ROW_H);
    }
    ctx.fillStyle    = 'rgba(200,200,200,0.45)';
    ctx.font         = '9px system-ui';
    ctx.textAlign    = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(labels[ci] || String(ci), LABEL_W - 4, y0 + ROW_H / 2);
  });

  drawTimeAxis(ctx, W, totalH, plotH, T);
  drawPlayhead(ctx, totalH, plotH, W);
}

function drawAttn(canvasEl, attn, T) {
  const PLOT_H = 50;
  const totalH = PLOT_H + AXIS_H;
  const W      = canvasEl.width = canvasEl.offsetWidth;
  canvasEl.height = totalH;

  const ctx   = canvasEl.getContext('2d');
  const plotW = W - LABEL_W;
  ctx.clearRect(0, 0, W, totalH);

  ctx.fillStyle = '#0d1a30';
  ctx.fillRect(LABEL_W, 0, plotW, PLOT_H);

  ctx.fillStyle    = 'rgba(200,200,200,0.45)';
  ctx.font         = '9px system-ui';
  ctx.textAlign    = 'right';
  ctx.textBaseline = 'middle';
  ctx.fillText('attn', LABEL_W - 4, PLOT_H / 2);

  const maxVal = Math.max(...attn, 1e-9);
  const norm   = attn.map(v => v / maxVal);

  const cellW = plotW / T;
  ctx.fillStyle = 'steelblue';
  ctx.beginPath();
  ctx.moveTo(LABEL_W, PLOT_H);
  for (let t = 0; t < T; t++) {
    const x = LABEL_W + (t + 0.5) * cellW;
    const y = PLOT_H * (1 - norm[t]);
    if (t === 0) ctx.moveTo(LABEL_W, PLOT_H);
    ctx.lineTo(x, y);
  }
  ctx.lineTo(LABEL_W + plotW, PLOT_H);
  ctx.closePath();
  ctx.fill();

  drawTimeAxis(ctx, W, totalH, PLOT_H, T);
  drawPlayhead(ctx, totalH, PLOT_H, W);
}

// ── redraw functions (check collapsed state) ──
function redrawHeatmap(key) {
  if (collapseState[key] || !actData) return;
  const canvas = document.getElementById('canvas-' + key);
  const head   = actData.heads[key];
  if (!canvas || !head) return;
  drawHeatmap(canvas, head, actData.T);
}

function redrawAttn() {
  if (collapseState['attn'] || !actData) return;
  const canvas = document.getElementById('canvas-attn');
  if (!canvas) return;
  drawAttn(canvas, actData.attn, actData.T);
}

function redrawAll() {
  headKeys.forEach(k => redrawHeatmap(k));
  redrawAttn();
}

// Redraw the active section's canvas when its container is resized.
new ResizeObserver(() => { if (actData) redrawAll(); }).observe(document.getElementById('sections'));

// ── init ───────────────────────────────────────────────────────────────────
fetchClips();
</script>
</body>
</html>
"""

HTML = (HTML
        .replace("__N_SHOW__", str(N_SHOW).replace("'", '"'))
        .replace("__HEAD_ORDER__", str(HEAD_ORDER).replace("'", '"'))
        .replace("__HEAD_TITLES__", str(HEAD_TITLES).replace("'", '"')))


@app.route("/")
def index():
    return render_template_string(HTML)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global CLIPS, MODEL, VOCAB, DEVICE, HANDEDNESS

    ap = argparse.ArgumentParser(
        description="Interactive STS-Net v0.2 activation inspector")
    ap.add_argument("videos", nargs="+",
                    help="Video (.mp4, .mov, ...) or .pose files to inspect")
    ap.add_argument("--ckpt", default="checkpoints/stsnet_v02.pt",
                    help="ClipClassifier checkpoint (default: checkpoints/stsnet_v02.pt)")
    ap.add_argument("--handedness", default="right", choices=["right", "left"])
    ap.add_argument("--device", default="cpu",
                    help="Torch device (default: cpu; use cuda for GPU)")
    ap.add_argument("--pose_cache_dir", default=None,
                    help="Directory to cache extracted .pose files "
                         "(default: alongside each video)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    HANDEDNESS = args.handedness
    pose_cache_dir = Path(args.pose_cache_dir) if args.pose_cache_dir else None
    if pose_cache_dir:
        pose_cache_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Resolve videos → pose files (extracting where needed) ────────────
    print(f"Preparing {len(args.videos)} clip(s)…")
    for i, raw in enumerate(args.videos):
        path = Path(raw)
        if not path.exists():
            print(f"  WARNING: not found, skipping: {path}")
            continue

        if path.suffix.lower() == ".pose":
            pose_path, video_path = path, None
        elif path.suffix.lower() in VIDEO_SUFFIXES:
            video_path = path
            cache_dir  = pose_cache_dir or path.parent
            pose_path  = cache_dir / (path.name + ".pose")
            if not pose_path.exists():
                ok = extract_pose(path, pose_path)
                pose_path = pose_path if ok else None
        else:
            print(f"  WARNING: unsupported file type, skipping: {path}")
            continue

        CLIPS.append({
            "idx": len(CLIPS), "name": path.name,
            "video_path": video_path, "pose_path": pose_path,
        })

    n_ok = sum(1 for c in CLIPS if c["pose_path"] is not None)
    print(f"  {n_ok}/{len(CLIPS)} clips ready")

    # ── 2. Load ClipClassifier checkpoint ────────────────────────────────────
    DEVICE = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    print(f"Loading ClipClassifier from {args.ckpt} on {DEVICE}…")
    from stsnet.clip_classifier import ClipClassifier
    MODEL, vocab_meta = ClipClassifier.from_checkpoint(args.ckpt, map_location=str(DEVICE))
    MODEL.to(DEVICE)
    MODEL.eval()

    def _invert(d):
        return {v: k for k, v in d.items()}

    VOCAB["idx_to_shape"]  = _invert(vocab_meta.get("shape_to_idx",  {}))
    VOCAB["idx_to_att"]    = _invert(vocab_meta.get("att_to_idx",    {}))
    VOCAB["idx_to_motion"] = _invert(vocab_meta.get("motion_to_idx", {}))
    VOCAB["idx_to_cloc"]   = _invert(vocab_meta.get("cloc_to_idx",   {}))
    VOCAB["idx_to_ctype"]  = _invert(vocab_meta.get("ctype_to_idx",  {}))

    print(f"\nStarting Inspector on http://{args.host}:{args.port}/")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()

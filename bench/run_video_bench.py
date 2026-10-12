#!/usr/bin/env python3
"""img2img video benchmark over the bundles built by bench/build_fast_models.sh.

For every (model, steps) in RUNS it runs the matching C++ harness on the same input clip
(validation_cpp/sd_video_harness for SD/SDXL, validation_cpp/klein_fps_harness for klein), then:
  <out>/clips/<run>.mp4             the output frames, burned-in model / steps / measured fps
  <out>/compare_<family>_<n>step.mp4 every model of a family side by side, frame-aligned
  <out>/results.json, results.md    throughput table (mean fps, p50 / p99 frame time)

Only the per-frame pipeline call is timed (upload, encode, denoise, decode, download, sync); writing
frames out happens after the timed loop. Runs are sequential, so they never share the GPU.

  bench/run_video_bench.py --engines /media/data2/lrd-engines-fast --video dance.mp4 \
      --harness-dir /media/data1/trt-install/harness-build --out /media/data3/lrd-benchmark/fast
Stdlib + ffmpeg only.
"""
import argparse, json, os, shutil, subprocess, sys, time

SDXL_PROMPT = "a vivid impressionist oil painting, bold brushstrokes, vibrant saturated color"
KLEIN_PROMPT = "turn this into " + SDXL_PROMPT
# Indices into the 50-entry scheduler table (t = 999 - 20*i). "preset" is the node presets' img2img
# default (t=359): it keeps the footage and only lightly restyles it. "strong" starts at t=599, where
# the prompt visibly takes over, which is where the models' quality actually differs.
SDXL_TIMESTEPS = {"preset": {1: "32", 2: "32, 45"}, "strong": {1: "20", 2: "20, 35"}}

# name -> (family, bundle subdir, extra) ; extra: sd "model" / klein "transformer"
MODELS = {
    "sdxl-turbo":      ("sdxl", "sdxl-turbo",      {"model": "sdxl"}),
    "dmd2-sdxl-1step": ("sdxl", "dmd2-sdxl-1step", {"model": "sdxl"}),
    "dmd2-sdxl-4step": ("sdxl", "dmd2-sdxl-4step", {"model": "sdxl"}),
    "nitrosd-realism": ("sdxl", "nitrosd-realism", {"model": "sdxl"}),
    "nitrosd-vibrant": ("sdxl", "nitrosd-vibrant", {"model": "sdxl"}),
    "klein":           ("klein", "klein",          {"transformer": "transformer_fp8_calib.plan"}),
    "klein-sprint":    ("klein", "klein-sprint",   {"transformer": "transformer_fp8_calib.plan"}),
}
GEOMETRY = {"sdxl": (1024, 1024), "klein": (320, 576)}


def sh(cmd, **kw):
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def prepare_video(src, w, h, seconds, start, dst):
    """Decode `seconds` of `src` from `start`, 25 fps, cover-cropped to WxH, as raw RGBA."""
    if not os.path.exists(dst):
        sh(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(seconds), "-i", src,
            "-vf", f"fps=25,scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}",
            "-pix_fmt", "rgba", "-f", "rawvideo", dst])
    return dst


def encode(raw_or_dir, w, h, title, dst, fps=25):
    """Frames -> mp4 at the source rate (so motion matches the input clip) with a title bar."""
    text = title.replace(":", r"\:").replace("'", "’")
    vf = (f"drawtext=text='{text}':x=8:y=8:fontsize={max(14, w // 40)}:fontcolor=white:"
          "box=1:boxcolor=black@0.6:boxborderw=6")
    if os.path.isdir(raw_or_dir):
        inp = ["-framerate", str(fps), "-i", os.path.join(raw_or_dir, "frame_%05d.ppm")]
    else:
        inp = ["-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-framerate", str(fps),
               "-i", raw_or_dir]
    sh(["ffmpeg", "-v", "error", "-y", *inp, "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18", dst])


def run_sd(args, name, bundle, extra, steps, video, w, h, workdir):
    raw = os.path.join(workdir, "out.rgba")
    js = os.path.join(workdir, "result.json")
    sh([os.path.join(args.harness_dir, "sd_video_harness"), "--bundle", bundle,
        "--model", extra["model"], "--video", video, "--size", f"{w}x{h}",
        "--timesteps", SDXL_TIMESTEPS[args.strength][steps], "--prompt", SDXL_PROMPT,
        "--warmup", str(args.warmup), "--out", raw, "--json", js, "--label", name])
    r = json.load(open(js))
    return r, raw


def run_klein(args, name, bundle, extra, steps, video, w, h, workdir, n_frames):
    clip_dir = os.path.join(workdir, "frames")
    os.makedirs(clip_dir, exist_ok=True)
    env = dict(os.environ, LRD_KLEIN_REF=video, LRD_KLEIN_STEPS=str(steps),
               LRD_KLEIN_WARMUP=str(args.warmup), LRD_KLEIN_PROMPT=KLEIN_PROMPT,
               LRD_CLIP_DIR=clip_dir, LRD_CLIP_MAX=str(n_frames))
    B = lambda f: os.path.join(bundle, f)
    # sync-noint = the node's sync path without RIFE: one diffusion per input frame, n_frames ticks.
    p = subprocess.run(
        [os.path.join(args.harness_dir, "klein_fps_harness"), B(extra["transformer"]),
         B("qwen3_encoder_bf16.plan"), B("vae_decoder_bf16.plan"), B("vae_encoder_bf16.plan"),
         B("tokenizer.json"), B("bn_mean.bin"), B("bn_std.bin"), B("rife_ifnet_fp16.plan"),
         "sync-noint", "0", str(n_frames), str(w), str(h)],
        env=env, check=True, capture_output=True, text=True)
    print(p.stdout)
    fps = None
    for line in p.stdout.splitlines():
        if "diffusion fps =" in line:
            fps = float(line.split("diffusion fps =")[1].split()[0])
    if fps is None:
        raise RuntimeError(f"{name}: no fps in klein_fps_harness output")
    r = {"label": name, "bundle": bundle, "model": "klein", "width": w, "height": h,
         "steps": steps, "frames": n_frames, "fps": fps, "mean_ms": 1000.0 / fps,
         "p50_ms": None, "p99_ms": None}
    return r, clip_dir


def side_by_side(clips, dst, cols):
    """Frame-aligned grid of already-titled clips (xstack)."""
    n = len(clips)
    rows = (n + cols - 1) // cols
    cells = []  # xstack layout, e.g. 0_0|w0_0|0_h0
    for i in range(n):
        c, r = i % cols, i // cols
        x = "+".join(["w0"] * c) or "0"
        y = "+".join(["h0"] * r) or "0"
        cells.append(f"{x}_{y}")
    inputs = sum((["-i", c] for c in clips), [])
    fill = "" if rows * cols == n else ":fill=black"
    sh(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex",
        f"xstack=inputs={n}:layout={'|'.join(cells)}{fill}", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-crf", "20", dst])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engines", required=True, help="dir holding one bundle per MODELS entry")
    ap.add_argument("--video", required=True, help="source clip (any ffmpeg-readable file)")
    ap.add_argument("--harness-dir", required=True, help="build dir of validation_cpp")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="*", default=list(MODELS), help="subset of MODELS")
    ap.add_argument("--steps", nargs="*", type=int, default=[1, 2])
    ap.add_argument("--seconds", type=float, default=6)
    ap.add_argument("--start", type=float, default=4)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--strength", choices=list(SDXL_TIMESTEPS), default="preset",
                    help="SD/SDXL img2img start timestep (klein always edits from noise)")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "clips"), exist_ok=True)
    results_path = os.path.join(args.out, "results.json")
    results = json.load(open(results_path)) if os.path.exists(results_path) else {}
    suffix = "" if args.strength == "preset" else "_" + args.strength

    for name in args.models:
        family, sub, extra = MODELS[name]
        bundle = os.path.join(args.engines, sub)
        if not os.path.isdir(bundle):
            print(f"[bench] skip {name}: no bundle at {bundle}")
            continue
        w, h = GEOMETRY[family]
        video = prepare_video(args.video, w, h, args.seconds, args.start,
                              os.path.join(args.out, f"input_{w}x{h}.rgba"))
        n_frames = os.path.getsize(video) // (w * h * 4)
        for steps in args.steps:
            run = f"{name}_{steps}step{suffix}"
            workdir = os.path.join("/dev/shm", "lrd_bench_" + run)
            shutil.rmtree(workdir, ignore_errors=True)
            os.makedirs(workdir)
            try:
                if family == "klein":
                    r, frames = run_klein(args, name, bundle, extra, steps, video, w, h, workdir, n_frames)
                else:
                    r, frames = run_sd(args, name, bundle, extra, steps, video, w, h, workdir)
                r.update(family=family, run=run, strength=args.strength if family != "klein" else "-",
                         date=time.strftime("%Y-%m-%d %H:%M"))
                encode(frames, w, h, f"{name}  {steps}-step  {r['fps']:.1f} fps",
                       os.path.join(args.out, "clips", run + ".mp4"))
                results[run] = r
                json.dump(results, open(results_path, "w"), indent=1)
            finally:
                shutil.rmtree(workdir, ignore_errors=True)

    # side-by-side per family and step count, the source clip first for reference
    for family in sorted({MODELS[m][0] for m in args.models}):
        w, h = GEOMETRY[family]
        src = os.path.join(args.out, "clips", f"input_{family}.mp4")
        raw = os.path.join(args.out, f"input_{w}x{h}.rgba")
        if os.path.exists(raw) and not os.path.exists(src):
            encode(raw, w, h, "input", src)
        for steps in args.steps:
            clips = [os.path.join(args.out, "clips", f"{m}_{steps}step{suffix if family != 'klein' else ''}.mp4")
                     for m in MODELS if MODELS[m][0] == family]
            clips = [c for c in clips if os.path.exists(c)]
            if clips:
                cols = 3 if family == "sdxl" else 6
                side_by_side([src] + clips, os.path.join(args.out, f"compare_{family}_{steps}step{suffix if family != 'klein' else ''}.mp4"),
                             cols)

    with open(os.path.join(args.out, "results.md"), "w") as f:
        f.write("| model | steps | strength | resolution | fps | mean ms | p50 ms | p99 ms |\n"
                "|---|---|---|---|---|---|---|---|\n")
        for run in sorted(results, key=lambda k: (results[k]["family"], results[k].get("strength", "preset"),
                                                  results[k]["steps"], -results[k]["fps"])):
            r = results[run]
            fmt = lambda v: "-" if v is None else f"{v:.1f}"
            strength = "-" if r["family"] == "klein" else r.get("strength", "preset")
            f.write(f"| {r['label']} | {r['steps']} | {strength} | "
                    f"{r['width']}x{r['height']} | {r['fps']:.1f} | "
                    f"{fmt(r['mean_ms'])} | {fmt(r['p50_ms'])} | {fmt(r['p99_ms'])} |\n")
    print(open(os.path.join(args.out, "results.md")).read())


if __name__ == "__main__":
    sys.exit(main())

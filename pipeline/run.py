"""Namma Space reconstruction pipeline: phone video -> camera poses -> Gaussian splat.

    python pipeline/run.py data/<scene>/video.mp4            # full run
    python pipeline/run.py data/<scene>/frames --from images  # already have photos

Stages (each skipped if its output already exists; --force redoes everything):
  1. frames   pick the sharpest frame in every window of the video
  2. features ALIKED keypoints (learned; copes with blank indoor walls better than SIFT)
  3. match    LightGlue, sequential pairs for video / exhaustive for small photo sets
  4. sfm      COLMAP global mapper (GLOMAP) -> camera poses + sparse points; if the poses are
              jittery (camera turned on the spot) redo with the slower incremental mapper
  5. undistort  to PINHOLE so the splat trainer sees ideal cameras
  6. splat    Brush trains 3D Gaussians -> splat.ply
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
COLMAP = ROOT / "tools/colmap/bin/colmap.exe"
BRUSH = ROOT / "tools/brush/brush_app.exe"

# COLMAP's ONNX models (ALIKED, LightGlue) run on GPU via CUDA 12 DLLs from the
# pip nvidia-* wheels in .venv, so no CUDA toolkit install is needed.
_nv = Path(sys.prefix) / "Lib/site-packages/nvidia"
os.environ["PATH"] = os.pathsep.join([str(p) for p in _nv.glob("*/bin")] + [os.environ["PATH"]])


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, logfile):
    log("  $ " + " ".join(str(c) for c in cmd[:2]) + " ...")
    with open(logfile, "a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        r = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        sys.exit(f"FAILED (exit {r.returncode}) - see {logfile}")


def sharpness(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2))
    return cv2.Laplacian(g, cv2.CV_64F).var()


def extract_frames(video, out_dir, fps, max_side):
    """Keep the sharpest frame from each 1/fps-second window (drops motion blur)."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"OpenCV cannot open {video} (HEVC? re-encode: ffmpeg -i in.mov -c:v libx264 out.mp4)")
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30
    window = max(1, round(vfps / fps))
    out_dir.mkdir(parents=True, exist_ok=True)
    best, best_s, i, n = None, -1, 0, 0
    scores = []
    while True:
        ok, img = cap.read()
        if ok:
            s = sharpness(img)
            if s > best_s:
                best, best_s = img, s
        if (not ok or (i + 1) % window == 0) and best is not None:
            h, w = best.shape[:2]
            k = max_side / max(h, w)
            if k < 1:
                best = cv2.resize(best, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / f"{n:05d}.jpg"), best, [cv2.IMWRITE_JPEG_QUALITY, 95])
            scores.append(best_s)
            best, best_s, n = None, -1, n + 1
        if not ok:
            break
        i += 1
    cap.release()
    s = np.array(scores)
    log(f"  {i} video frames @ {vfps:.1f}fps -> {n} keyframes; sharpness median {np.median(s):.0f}, "
        f"{(s < 0.3 * np.median(s)).sum()} look blurry")
    return n


def largest_model(sparse_root):
    models = [d for d in sparse_root.iterdir() if (d / "images.bin").exists()]
    if not models:
        sys.exit("SfM produced no model - capture probably lacks overlap")
    sizes = {d: (d / "images.bin").stat().st_size for d in models}
    big = [s for s in sizes.values() if s > 0.05 * max(sizes.values())]  # ignore 2-3 frame scraps
    if len(big) > 1:
        log(f"  WARNING: scene split into {len(big)} pieces; using the largest. "
            "Re-shoot the joins (doorways) with more overlap.")
    return max(sizes, key=sizes.get)


def read_poses(model, logfile):
    """[(name, center, R_world_to_cam)] for every registered image, sorted by name (= video order)."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        run([COLMAP, "model_converter", "--input_path", model, "--output_path", tmp,
             "--output_type", "TXT"], logfile)
        lines = [l for l in Path(tmp, "images.txt").read_text().splitlines() if l and not l.startswith("#")]
    poses = []
    for l in lines[::2]:  # every image has a pose line followed by a 2D-points line
        f = l.split()
        qw, qx, qy, qz, tx, ty, tz = map(float, f[1:8])
        R = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
        poses.append((f[9], -R.T @ [tx, ty, tz], R))
    return sorted(poses, key=lambda p: p[0])


def pose_jitter(model, logfile):
    """(fraction of steps > 5x median, number of >8 deg turns) between consecutive video frames.
    The global mapper solves positions from pairwise directions, which are undefined when the
    camera turns on the spot (no parallax); that shows up as jumps and sudden turns here."""
    poses = read_poses(model, logfile)
    c = np.array([p[1] for p in poses])
    f = np.array([p[2].T[:, 2] for p in poses])
    step = np.linalg.norm(np.diff(c, axis=0), axis=1)
    turn = np.degrees(np.arccos(np.clip((f[1:] * f[:-1]).sum(1), -1, 1)))
    return (step > 5 * np.median(step)).mean(), int((turn > 8).sum())


def drop_pose_outliers(model, out_model, logfile, k=10):
    """Video frames are ~10 cm apart, so a camera far from BOTH time-neighbours is a bad registration.
    Such poses make the splat trainer smear floaters across the room, so delete them."""
    poses = read_poses(model, logfile)
    keep = list(range(len(poses)))
    while len(keep) > 3:
        c = np.array([poses[i][1] for i in keep])
        step = np.linalg.norm(np.diff(c, axis=0), axis=1)
        med = np.median(step)
        near = np.minimum(np.r_[np.inf, step], np.r_[step, np.inf])  # distance to closer neighbour
        worst = int(np.argmax(near))
        if near[worst] <= k * med:
            break
        keep.pop(worst)
    bad = sorted(set(range(len(poses))) - set(keep))
    if out_model.exists():
        shutil.rmtree(out_model)
    out_model.mkdir(parents=True)
    names = out_model.parent / "outliers.txt"
    names.write_text("\n".join(poses[i][0] for i in bad))
    run([COLMAP, "image_deleter", "--input_path", model, "--output_path", out_model,
         "--image_names_path", names], logfile)
    log(f"  dropped {len(bad)} outlier poses: {', '.join(poses[i][0] for i in bad) or 'none'}")


def export_cameras(model, out_json, logfile):
    """Camera centres + view directions (COLMAP world frame) so the viewer can start inside the room."""
    import json
    import tempfile
    cams = [{"name": n, "center": c.round(4).tolist(), "forward": R.T[:, 2].round(4).tolist(),
             "up": (-R.T[:, 1]).round(4).tolist()} for n, c, R in read_poses(model, logfile)]
    with tempfile.TemporaryDirectory() as tmp:
        run([COLMAP, "model_converter", "--input_path", model, "--output_path", tmp,
             "--output_type", "TXT"], logfile)
        cam = next(l for l in Path(tmp, "cameras.txt").read_text().splitlines() if l and l[0] != "#").split()
    # OPENCV/PINHOLE params start fx, fy; SIMPLE_* start f
    w, h, fy = int(cam[2]), int(cam[3]), float(cam[5] if cam[1] in ("OPENCV", "PINHOLE") else cam[4])
    out_json.write_text(json.dumps({"width": w, "height": h, "fy": round(fy, 2), "cams": cams}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="video file, or folder of images with --from images")
    ap.add_argument("--from", dest="src", choices=["video", "images"], default="video")
    ap.add_argument("--out", type=Path, help="scene folder (default: next to input)")
    ap.add_argument("--fps", type=float, default=3, help="keyframes per second of video")
    ap.add_argument("--max-side", type=int, default=1600, help="resize frames so longest side <= this")
    ap.add_argument("--matcher", choices=["sequential", "exhaustive"], help="default: sequential for video")
    ap.add_argument("--mapper", choices=["auto", "global", "incremental"], default="auto",
                    help="auto: fast global mapper, falls back to incremental if poses are jittery")
    ap.add_argument("--steps", type=int, default=15000, help="splat training steps (7000 = fast preview)")
    ap.add_argument("--viewer", action="store_true", help="watch splat training live")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    scene = a.out or (a.input.parent if a.input.is_file() else a.input.parent)
    images = scene / "images"
    work = scene / "colmap"
    db = work / "database.db"
    sparse = work / "sparse"
    dense = scene / "undistorted"
    splat_dir = scene / "splat"
    logfile = scene / "pipeline.log"
    if a.force:
        for p in (work, dense, scene / "undistorted_inc", splat_dir):  # images/ kept unless re-extracted
            shutil.rmtree(p, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    timings = {}

    def stage(name):
        timings[name] = time.time()
        log(name)

    stage("1/6 frames")
    if a.src == "images":
        images = a.input
        n = len(list(images.glob("*.[jJpP][pPnN]*[gG]")))
        log(f"  using {n} existing images in {images}")
    elif not images.exists() or a.force:
        shutil.rmtree(images, ignore_errors=True)
        n = extract_frames(a.input, images, a.fps, a.max_side)
    else:
        n = len(list(images.glob("*.jpg")))
        log(f"  reusing {n} frames")
    matcher = a.matcher or ("sequential" if a.src == "video" else "exhaustive")

    if not db.exists():
        stage("2/6 features (ALIKED)")
        run([COLMAP, "feature_extractor", "--database_path", db, "--image_path", images,
             "--ImageReader.single_camera", 1, "--ImageReader.camera_model", "OPENCV",
             "--FeatureExtraction.type", "ALIKED_N16ROT",
             "--AlikedExtraction.max_num_features", 4096], logfile)

        stage(f"3/6 match (LightGlue, {matcher})")
        cmd = [COLMAP, f"{matcher}_matcher", "--database_path", db,
               "--FeatureMatching.type", "ALIKED_LIGHTGLUE"]
        if matcher == "sequential":
            cmd += ["--SequentialMatching.overlap", 20, "--SequentialMatching.quadratic_overlap", 1]
        run(cmd, logfile)

    if not sparse.exists():
        stage("4/6 sfm (global mapper)")
        sparse.mkdir()
        run([COLMAP, "global_mapper", "--database_path", db, "--image_path", images,
             "--output_path", sparse], logfile)
    model = largest_model(sparse)
    mapper = a.mapper
    if mapper == "auto":
        jumps, turns = pose_jitter(model, logfile) if a.src == "video" else (0, 0)
        log(f"  pose jitter: {jumps:.0%} of steps are jumps, {turns} sudden turns")
        mapper = "incremental" if jumps > 0.08 or turns >= 3 else "global"
    if mapper == "incremental":
        sparse_inc = work / "sparse_inc"
        if not sparse_inc.exists():
            stage("4b/6 sfm (incremental mapper: global poses jittery, likely turning on the spot)")
            sparse_inc.mkdir()
            run([COLMAP, "mapper", "--database_path", db, "--image_path", images,
                 "--output_path", sparse_inc], logfile)
        inc = largest_model(sparse_inc)
        n_glob, n_inc = len(read_poses(model, logfile)), len(read_poses(inc, logfile))
        if a.mapper == "auto" and n_inc < 0.9 * n_glob:
            # incremental can fragment on weak captures; smooth poses aren't worth losing frames
            log(f"  incremental registered only {n_inc} vs {n_glob} frames; keeping global")
            mapper = "global"
        else:
            model = inc
    suffix = "_inc" if mapper == "incremental" else ""  # keep caches of the two mappers apart
    dense = scene / f"undistorted{suffix}"
    reg = subprocess.run([str(COLMAP), "model_analyzer", "--path", str(model)],
                         capture_output=True, text=True)
    stats = [l.split("]")[-1].strip() for l in (reg.stdout + reg.stderr).splitlines()
             if any(k in l for k in ("Registered images", "Points", "Mean reprojection"))]
    log("  " + " | ".join(stats) + f"  (of {n} frames)")
    if a.src == "video":
        clean = work / f"clean{suffix}"
        if not clean.exists():
            drop_pose_outliers(model, clean, logfile)
        model = clean

    if not dense.exists():
        stage("5/6 undistort")
        run([COLMAP, "image_undistorter", "--image_path", images, "--input_path", model,
             "--output_path", dense, "--output_type", "COLMAP"], logfile)

    stage(f"6/6 splat (Brush, {a.steps} steps)")
    splat_dir.mkdir(exist_ok=True)
    cmd = [BRUSH, dense, "--total-steps", a.steps, "--export-every", a.steps,
           "--export-path", splat_dir, "--export-name", "splat.ply"]
    if a.viewer:
        cmd.append("--with-viewer")
    run(cmd, logfile)
    export_cameras(model, splat_dir / "cameras.json", logfile)

    timings["end"] = time.time()
    names = list(timings)
    log("timing: " + ", ".join(f"{names[i].split()[0]} {timings[names[i+1]] - timings[names[i]]:.0f}s"
                               for i in range(len(names) - 1)) + f"  | total {time.time() - t0:.0f}s")
    log(f"done -> {splat_dir / 'splat.ply'}")


if __name__ == "__main__":
    main()

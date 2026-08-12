# img-3dpl

***img**age **-**to **3d** **p**ipe**l**ine*

Turn a folder of ordinary photos into a Gaussian Splatting model, on a computer
with no working graphics card.

[Vietnamese version of this document → README.vi.md](README.vi.md)

---

## No GPU required. Genuinely.

This entire pipeline was built and tested on a budget laptop whose dedicated
graphics card is **physically dead** — an Intel i3-1005G1, two cores, integrated
graphics. It still turned 320 photos into a 3D model.

Three things make that work:

- **Compiling CUDA code does not need a CUDA device.** It needs `nvcc`, which
  ships inside a Docker image. Your machine never executes a single GPU
  instruction to build the software.
- **The GPU-hungry stage runs on Colab's free T4.** Installation there takes
  seven seconds — no `apt install`, no `ldconfig`, no compiling.
- **The stage that stays on your machine does not want a GPU anyway.** Building
  camera positions is sequential work. CPU cores beat graphics cards at it, and
  free-tier Colab hands you only two of them.

If your laptop can run Docker and open a browser, it can do this. It will be
slower than a workstation. It will still finish.

---

The awkward part of photogrammetry on Colab is that the `colmap` package from
`apt` is **compiled without CUDA**. Exhaustive matching of 320 photos on CPU runs
for over three hours and does not finish. This repository fixes that, and splits
the work so each stage runs where it is fastest.

---

## How the work is split

| Stage | Runs on | Why there | 320 photos |
|---|---|---|---|
| **1. Match photos** | Colab T4 | Massively parallel, GPU wins by ~23x | **28 min** |
| **2. Build camera positions** | Your own machine | Sequential — a GPU cannot help, and your laptop likely has more CPU cores than free-tier Colab | ~105 min |
| **3. Train Gaussian Splatting** | Colab T4 | Needs CUDA | 45-70 min |

Stage 2 is the one people get wrong. Bundle adjustment can use a GPU, but
registering images is inherently sequential: each new photo depends on the ones
already placed. Free-tier Colab gives you 2 CPU cores. Most laptops have more.

```
   photos  ──►  [1] Colab GPU      ──►  database.db
                    match pairs

database.db ──►  [2] desktop app   ──►  images/ + sparse/0/
   + photos         camera positions

  sparse/0  ──►  [3] Colab GPU     ──►  .ply model
                    train 3DGS
```

---

## Measured results

320 JPEG photos at 1200x1600, one phone camera, orbiting a single object.

| Command | Machine | Pairs compared | Time | Seconds per pair |
|---|---|---|---|---|
| `sequential_matcher` | Intel i3-1005G1, CPU | ~3,000 | 38.6 min | 0.772 |
| `exhaustive_matcher` | Colab CPU (`apt` colmap) | 51,040 | over 3 h, **never finished** | — |
| `exhaustive_matcher` | **Colab T4, this pipeline** | **51,040** | **28 min** | **0.033** |

About **23x faster per unit of work**. The headline is not the ratio though — it
is that exhaustive matching moves from *impossible* to *routine*. Running those
51,040 pairs on the i3 would take roughly 11 hours.

Exhaustive matching also produces a **better** model than sequential. Sequential
only compares each photo with its 10 neighbours, so when you orbit an object it
never notices that the last photo overlaps the first. That missed loop closure
lets the reconstruction drift.

---

## Quick start

### Stage 1 — matching, on Colab

Open [`notebooks/1_match_images_colab.ipynb`](notebooks/1_match_images_colab.ipynb)
in Colab, set the runtime to **T4 GPU**, edit the configuration cell, run all
cells. You get a `database.db` on your Drive.

Installing COLMAP inside Colab takes **7 seconds** — it is a prebuilt,
self-contained tarball, no `apt install`, no `ldconfig`:

```python
!wget -q https://github.com/Ryanhuhut/colmap-cuda-colab/releases/latest/download/colmap-3.13.0-cuda12.2-ubuntu2204-sm75.tar.gz -O /tmp/colmap.tar.gz
!tar -xzf /tmp/colmap.tar.gz -C /opt
import os; os.environ["PATH"] = "/opt/colmap-cuda/bin:" + os.environ["PATH"]
!colmap -h | head -3
```

The last line must print `with CUDA`.

### Stage 2 — camera positions, on your machine

Download `database.db` from Drive, then run the desktop app. It needs Docker and
nothing else — COLMAP runs inside a container, so nothing is installed on your
system.

```bash
git clone https://github.com/Ryanhuhut/img-3dpl
cd img-3dpl/desktop-app

mkdir -p ~/.local/share/quet3d
wget https://github.com/Ryanhuhut/colmap-cuda-colab/releases/latest/download/colmap-3.13.0-cuda12.2-ubuntu2204-sm75.tar.gz \
     -O ~/.local/share/quet3d/colmap.tar.gz

chmod +x quet3d.py
./quet3d.py
```

Drop in the photo folder and the `.db` file, press start. It shows elapsed time,
a live progress bar driven by COLMAP's own output, a time estimate, and a large
warning so nobody shuts the machine down mid-run. It also blocks the system from
suspending while it works.

Requires GTK4 and libadwaita, which ship with any current GNOME desktop
(`python3-gobject gtk4 libadwaita`). The interface is in Vietnamese.

### Stage 3 — training, on Colab

Upload the output folder to Drive, open
[`notebooks/3_train_gaussian_splatting_colab.ipynb`](notebooks/3_train_gaussian_splatting_colab.ipynb),
edit the configuration cell, run all cells. Checkpoints are copied to Drive as
they appear, so a dropped session costs you nothing already finished.

---

## What is in here

```
notebooks/
  1_match_images_colab.ipynb              stage 1 — GPU matching
  3_train_gaussian_splatting_colab.ipynb  stage 3 — training
desktop-app/
  quet3d.py                               stage 2 — GTK4 app
  quet3d.desktop                          launcher entry
build-colmap/
  Dockerfile.builder                      CUDA 12.2 + Ubuntu 22.04 base
  01-build-ceres.sh                       Ceres 2.2.0, statically linked
  02-build-colmap.sh                      COLMAP 3.13.0, CUDA on
  03-package.sh                           bundle dependencies via ldd
  04-cho-va-dong-goi.sh                   wait for build, then package
```

The prebuilt COLMAP binary lives in a separate repository:
**[Ryanhuhut/colmap-cuda-colab](https://github.com/Ryanhuhut/colmap-cuda-colab)**.
Everything needed to rebuild it is in `build-colmap/` here as well.

---

## Traps worth knowing about

**COLMAP 3.13 renamed the GPU flags.** Nearly every tutorial online uses the old
names and will fail with `unrecognised option`:

| Old, broken | Correct for 3.13 |
|---|---|
| `--SiftExtraction.use_gpu` | `--FeatureExtraction.use_gpu` |
| `--SiftMatching.use_gpu` | `--FeatureMatching.use_gpu` |

Both already default to `1`. When in doubt, ask the binary rather than the
internet: `colmap feature_extractor -h | grep use_gpu`.

**The prebuilt binary is sm_75 only — Tesla T4.** On an L4 or A100 it fails with
a CUDA architecture error. Rebuild with `CUDA_ARCH="75;80;89"` in
`02-build-colmap.sh` if you have paid Colab.

**Do not build on Ubuntu 24.04.** Binaries compiled on an older system run on
newer ones, not the reverse. Building on 22.04 produces something that runs on
both — verified on clean 22.04 and 24.04 containers.

**Ubuntu 22.04's Ceres is too old.** COLMAP 3.13 uses `ceres::Manifold`, which
arrived in Ceres 2.1. Ubuntu 22.04 ships 2.0.0, so Ceres is built from source and
linked statically — that way Colab needs no Ceres at all.

**`libgomp` must be bundled.** Bare Ubuntu images do not have it. Miss it and the
package runs fine on your build machine, then dies on Colab with
`error while loading shared libraries: libgomp.so.1`.

**`libcuda.so.1` must never be bundled.** That is the driver, and it has to come
from the machine that actually owns the GPU.

**On SELinux systems, Docker mounts need the `:z` suffix.** Fedora, RHEL,
CentOS. Without it you get `Permission denied`.

---

## Requirements

**Stage 1 and 3:** a Google account. Free-tier Colab is enough.

**Stage 2:** Linux with Docker, about 3 GB of free disk, GTK4 and libadwaita.
No GPU needed — the app never touches one.

**Rebuilding COLMAP yourself:** Docker and roughly 20 GB of free disk. Still no
GPU required; compiling CUDA code only needs `nvcc`, which lives inside the base
image. The build takes about 30 minutes on 4 threads.

---

## Credits

- [COLMAP](https://github.com/colmap/colmap) — structure from motion
- [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) — Inria / MPII
- [Ceres Solver](https://github.com/ceres-solver/ceres-solver) — non-linear optimisation
- [SuperSplat](https://superspl.at/editor) — browser viewer

---

## License

MIT for the code in this repository. The tools it drives keep their own licenses:
COLMAP is BSD, and Gaussian Splatting is free for non-commercial research use —
check its terms before using it commercially.

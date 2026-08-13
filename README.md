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

### Stage 0 — shrink the photos to 1600px first

Do this before anything else. Matching 4000px photos is not 6x better than
matching 1600px photos, it is just slower — COLMAP downsizes to 3200px on its
own anyway, and Gaussian Splatting is happy at 1600px. Shrinking first also cuts
the upload from gigabytes to a few hundred megabytes.

Needs ImageMagick 7 (`magick`). Edit the three lines at the top, paste the rest:

```bash
# ==================== EDIT THIS ====================
SRC="/home/you/Downloads/Cap-GB"        # folder of original photos
DST="/home/you/quet3d/Cap-GB_1600"      # destination — name it after the scan
EXT="jpg"                               # extension: jpg / jpeg / png
# ===================================================

cd "$SRC" || { echo "NO SUCH FOLDER"; exit 1; }

echo "--- Photos: $(ls *.$EXT 2>/dev/null | wc -l)"
echo "--- EXIF focal length (must print ONE row only):"
magick identify -format "%[EXIF:FocalLengthIn35mmFilm] " *.$EXT 2>/dev/null | tr ' ' '\n' | sort | uniq -c

mkdir -p "$DST"
echo "--- Shrinking, this takes a few minutes..."
magick mogrify -path "$DST" -resize 1600x1600 -quality 93 *.$EXT

echo "--- DONE: $(ls "$DST" | wc -l) photos, $(du -sh "$DST" | cut -f1)"
```

The EXIF line is the important one. **It must print a single row.** Two rows
means two different focal lengths in the folder — a zoom that moved, or photos
from two cameras — and one camera model can no longer describe them all. Either
throw out the odd ones, or set `SINGLE_CAMERA = False` in stage 1.

Then pack the shrunk folder into a zip:

```bash
# ==================== EDIT THIS ====================
DST="/home/you/quet3d/Cap-GB_1600"    # same folder as above
# ===================================================

cd "$(dirname "$DST")" || exit 1
zip -r -0 "$(basename "$DST").zip" "$(basename "$DST")"
echo "--- DONE: $(du -sh "$(basename "$DST").zip" | cut -f1)"
```

`-0` means store, no compression. JPEG is already compressed; asking zip to
squeeze it again costs minutes and saves nothing. Upload that zip to Drive.

The photos may sit at the top of the zip or inside a folder — stage 1 finds them
either way, and ignores `__MACOSX/` and hidden files.

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

Stage 2 leaves you a folder like this:

```
Meo_3d/
├── images/                      ← needed: undistorted photos
├── sparse/0/
│   ├── cameras.bin              ← needed: lens parameters
│   ├── images.bin               ← needed: position and rotation of each photo
│   └── points3D.bin             ← needed: the sparse cloud
├── distorted/
│   ├── database.db              ← not needed, and by far the biggest file
│   └── sparse/0/*.bin           ← not needed: the model before undistortion
├── stereo/                      ← not needed: empty scaffolding for dense
└── run-colmap-*.sh              ← not needed
```

Zip the whole thing and upload one file:

```bash
# ==================== EDIT THIS ====================
SCENE="/home/you/quet3d/Meo_3d"    # the folder stage 2 produced
# ===================================================

cd "$(dirname "$SCENE")" || exit 1
NAME="$(basename "$SCENE")"
zip -r -0 "${NAME%_3d}.zip" "$NAME" -x "$NAME/distorted/*" "$NAME/stereo/*"
echo "--- DONE: $(du -sh "${NAME%_3d}.zip" | cut -f1)"
```

The `-x` flags leave out `distorted/` and `stereo/`. The notebook skips them by
itself if you include them, but `database.db` is usually several gigabytes and
uploading it is an hour you never get back.

Then open
[`notebooks/3_train_gaussian_splatting_colab.ipynb`](notebooks/3_train_gaussian_splatting_colab.ipynb),
paste the path to that zip into **cell 3 — the only line you have to edit** —
and run all cells. It finds `images/` and `sparse/0/` at whatever depth they
sit, extracts only those two, names the output after the zip (`Meo.zip` →
`Meo_30000.ply`), and trains. A `.tar.gz` or a plain folder on Drive works just
as well.

Checkpoints are copied to Drive as they appear, so a dropped session costs you
nothing already finished.

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

## The COLMAP binary

Stage 1 and stage 2 both use the same prebuilt COLMAP, so the database written on
Colab is read by the exact same version on your machine. Mixing versions is how
you end up staring at `SQL logic error`.

Always the newest build:

```
https://github.com/Ryanhuhut/colmap-cuda-colab/releases/latest/download/colmap-3.13.0-cuda12.2-ubuntu2204-sm75.tar.gz
```

Pinned to this exact version, for reproducible results later:

```
https://github.com/Ryanhuhut/colmap-cuda-colab/releases/download/v3.13.0-cuda12.2-sm75/colmap-3.13.0-cuda12.2-ubuntu2204-sm75.tar.gz
```

46 MB (47,654,853 bytes).

```
SHA256: 6c72e8535a780198ce5e56af01d9aac4447de32051cffa9a773733ebcd9ac317
```

Repository, build scripts and release notes:
**[Ryanhuhut/colmap-cuda-colab](https://github.com/Ryanhuhut/colmap-cuda-colab)**

---

## Everything this is built on

None of the hard parts here are mine. This repository is plumbing: it connects
existing tools, compiles one of them properly, and splits the work sensibly.

### The two that do the actual work

| Project | Role | License |
|---|---|---|
| [COLMAP](https://github.com/colmap/colmap) | Structure from motion — finds where each photo was taken from | BSD |
| [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) | The renderer, from Inria and MPII | Non-commercial research |

### Bundled inside the COLMAP package

| Library | Version | Role | License |
|---|---|---|---|
| [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) | 12.2.140 | GPU compute | NVIDIA EULA |
| [Ceres Solver](https://github.com/ceres-solver/ceres-solver) | 2.2.0 | Non-linear optimisation | BSD |
| [Eigen](https://eigen.tuxfamily.org) | 3.4.0 | Linear algebra | MPL2 |
| [Boost](https://www.boost.org) | 1.74.0 | Command line parsing, graphs | Boost |
| [PoseLib](https://github.com/PoseLib/PoseLib) | `f119951` | Camera pose solvers | BSD |
| [faiss](https://github.com/facebookresearch/faiss) | `36b7735` | Nearest-neighbour search | MIT |
| [SuiteSparse](https://people.engr.tamu.edu/davis/suitesparse.html) | 5.10.1 | Sparse matrix solvers | LGPL/GPL |
| [OpenBLAS](https://www.openblas.net) | 0.3.20 | Basic linear algebra | BSD |
| [LAPACK](https://www.netlib.org/lapack/) | 3.10.0 | Higher-level linear algebra | BSD |
| [METIS](https://github.com/KarypisLab/METIS) | 5.1.0 | Graph partitioning | Apache 2.0 |
| [CGAL](https://www.cgal.org) | 5.4 | Computational geometry | GPL/LGPL |
| [FreeImage](https://freeimage.sourceforge.io) | 3.18.0 | Image reading and writing | FIPL/GPL |
| [glog](https://github.com/google/glog) | 0.4.0 | Logging | BSD |
| [gflags](https://github.com/gflags/gflags) | 2.2.2 | Flag handling | BSD |
| [SQLite](https://www.sqlite.org) | 3.37.2 | The feature and match database | Public domain |

Plus the image codecs FreeImage pulls in — libjpeg, libpng, libtiff, libwebp,
libraw, OpenEXR, JPEG-XR — along with zlib, OpenSSL, libcurl and libgomp.
Seventy-nine shared libraries in total; the full list is in `BUILD_INFO.txt`
inside the tarball.

### The desktop app

| Project | Role | License |
|---|---|---|
| [GTK4](https://www.gtk.org) | Interface toolkit | LGPL |
| [libadwaita](https://gitlab.gnome.org/GNOME/libadwaita) | GNOME styling | LGPL |
| [PyGObject](https://pygobject.gnome.org) | Python bindings for both | LGPL |
| [Docker](https://www.docker.com) | Runs COLMAP without installing it | Apache 2.0 |

### Viewing the result

| Project | Role |
|---|---|
| [SuperSplat](https://superspl.at/editor) | Browser-based viewer and editor for `.ply` splats |

### Where it runs

[Google Colab](https://colab.research.google.com) free tier — a Tesla T4 with
15 GB of VRAM, and two CPU cores.

---

## License

MIT for the code in this repository. The tools it drives keep their own licenses:
COLMAP is BSD, and Gaussian Splatting is free for non-commercial research use —
check its terms before using it commercially.

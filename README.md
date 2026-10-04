# Generative AI Assignment 1: restoration autoencoders and face-to-sketch cGAN

Four models and one web application.

| Task | Model | Main test result |
|---|---|---|
| 1 | Universal denoising autoencoder (conv latent 8x8x32, no skips) | SSIM 0.68 to 0.77 per corruption type |
| 2 | Corruption classifier + three specialist autoencoders (hard routing) | Classifier 99.71% accurate; predicted routing within 0.002 SSIM of oracle |
| 3 | Jointly trained soft mixture-of-experts (gate + 3 experts + identity) | About equal to hard routing; better on low-severity occlusion |
| 4 | Style-conditioned U-Net / PatchGAN face-to-sketch cGAN | SSIM 0.49 on the official FS2K test set |

Report (IEEE, LaTeX): `report/main.tex`. Demo video: **TODO: YouTube link**. Weights & Biases projects: `genai-a1-task1` to `genai-a1-task4` (entity `seeratzahra500-fast-nuces`).

## 1. Run the application (Docker, one command)

Requirements: Docker with Compose v2, about 3 GB of free disk and an internet connection on the first start (about 435 MB of models are downloaded). No GPU is needed: inference uses ONNX Runtime on the CPU.

```bash
git clone https://github.com/Seeratzahra500/GenAI-Assignment1.git
cd GenAI-Assignment1
docker compose up --build
```

Open **http://localhost:3000**. The first start downloads the seven ONNX files into `./models/` (the health dot in the top right shows `7 of 7 models loaded` when ready; allow a few minutes). Later starts reuse the files. API documentation: http://localhost:8000/docs.

To stop: `Ctrl+C`, then `docker compose down`.

### Model files

The ONNX models are not stored in Git (they total about 435 MB). They are hosted in a public Hugging Face repository and downloaded by the backend on first start. The repository name is set in `.env`:

```
MODELS_REPO=<huggingface-username>/genai-a1-models
```

If you already have the files, copy them into `./models/` and no download happens:

| File | Size | Model |
|---|---|---|
| `task1_universal_conv.onnx` | 38 MB | Task 1 |
| `task2_classifier.onnx` | 1.2 MB | Task 2 classifier |
| `task2_spec_salt.onnx`, `task2_spec_blur.onnx`, `task2_spec_occlusion.onnx` | 38 MB each | Task 2 specialists |
| `task3_soft_moe.onnx` | 115 MB | Task 3 (gate, experts, blend; outputs image and weights) |
| `task4_generator.onnx` | 168 MB | Task 4 generator |

### Sample images

The "clean samples" strips in the app read from `app/backend/samples/pets` and `app/backend/samples/faces`. Create them once (needs the datasets, see section 3) and commit the small JPEG files:

```bash
python scripts/make_samples.py --pets data/oxford-iiit-pet --fs2k data/FS2K
```

The app also works without samples: upload any JPEG, PNG or WebP image (max 8 MB).

### Run without Docker (development)

```bash
# backend
cd app/backend
pip install -r requirements.txt
MODELS_DIR=../../models SAMPLES_DIR=./samples uvicorn main:app --port 8000
# frontend (second terminal)
cd app/frontend
npm install
npm run dev            # http://localhost:5173, proxies /api to port 8000
```

## 2. What the application does

| Workspace | Endpoint | Shows |
|---|---|---|
| Universal Restoration | `POST /api/universal` | clean target, model input, restored image, corruption settings, inference time, PSNR/SSIM |
| Hard-Routed Restoration | `POST /api/hard` | the above plus four classifier probabilities, predicted corruption, selected expert (clean uses an identity bypass), classifier and expert time |
| Soft Mixture-of-Experts Restoration | `POST /api/soft` | the above plus four routing weights and the contributing experts |
| Face-to-Sketch Generator | `POST /api/sketch` | photo and sketch side by side, style 1/2/3, webcam capture, download |

Also `GET /api/health`, `GET /api/samples`. The restoration endpoints take a multipart `file`, `corruption` (`none`, `salt`, `blur`, `occlusion`), `severity` (`low`, `medium`, `high`) and an optional `seed`. Use `corruption=none` for an image that is already corrupted. Example:

```bash
curl -F file=@cat.jpg -F corruption=salt -F severity=medium http://localhost:8000/api/hard
```

The webcam needs `http://localhost` or HTTPS. Uploads are validated (type, size, pixel count); errors return 400, 413, 415 or 422, and a missing model returns 503.

## 3. Reproduce the training (Kaggle or a GPU machine)

```bash
pip install -r requirements.txt
```

**Data**
- Oxford-IIIT Pet: `torchvision.datasets.OxfordIIITPet(root="data", split="trainval", download=True)` (the official source can be slow; the `test` split is in the same archive).
- FS2K: Google Drive id `1saIMhQ3dc5_ftkfGmBPbCluRn_zy7QQp` (`gdown <id> -O FS2K.zip`, then unzip to `data/FS2K`). Official split files `anno_train.json` and `anno_test.json` are inside.

**Code** (`src/`)

| Module | Purpose |
|---|---|
| `data.py` | Pet loader, seed-42 80/20 split, runtime corruptions, `build_manifests()` for the fixed validation and test manifests (`manifests/`) |
| `models.py` | `ConvAE` (dense baseline), `ConvAE2` (conv bottleneck), L1+SSIM loss |
| `train_ae.py`, `study2.py`, `study3.py` | Task 1 training and Optuna studies (`run_study`, `run_study_v2`, `run_study_v3`) |
| `eval_ae.py` | Test evaluation by type and severity, example grids, `export_onnx` with the PyTorch check |
| `task2_clf.py`, `task2_spec.py` | Task 2 classifier (balanced batches) and specialists, with their Optuna studies |
| `task3_moe.py` | Soft MoE: warm-up then joint training, `run_moe_study`, `train_moe` |
| `fs2k.py`, `gan.py` | Task 4 loader (stratified 15% validation, paired augmentation), U-Net generator, PatchGAN, `run_gan_study`, `train_gan` |

The exact cells that were run, in order, are in `notebooks/` (outputs cleared). Typical order: build manifests, Task 1 studies and final training, Task 2 classifier then specialists then routing evaluation, Task 3 study then final training (3 warm-up and 12 joint epochs) then evaluation, Task 4 study then 80-epoch training. Every notebook exports its ONNX file and compares it with PyTorch (`np.allclose`, atol 1e-4).

**Results and tracking**
- `results/`: per-sample CSVs, result tables, example and failure figures, routing heatmap.
- `studies/`: one CSV per Optuna study. The Task 1 conv study, the Task 2 studies and the Task 3 study were rebuilt from W&B run logs after a Kaggle session restore, so their `state` column is the W&B run state, not Optuna's. The Task 1 dense studies and the Task 4 study are the original exports.
- Weights & Biases holds the training curves, hyperparameters, checkpoints (artifacts) and the Task 4 sample grids.

## 4. Repository layout

```
app/backend/    FastAPI service, corruption code, model registry, download script, Dockerfile
app/frontend/   React + Tailwind (Vite), nginx config, Dockerfile
docker-compose.yml, .env
src/            training, evaluation and export code
manifests/      fixed validation and test corruption manifests
results/, studies/, notebooks/, report/, scripts/
```

## 5. Troubleshooting

- *Health dot shows fewer than 7 models*: check `docker compose logs backend`. A failed download usually means `MODELS_REPO` in `.env` is wrong or the Hugging Face repository is private.
- *Port 3000 or 8000 in use*: change the left side of the `ports` entries in `docker-compose.yml`.
- *Slow first request*: the first inference loads the model into memory; later ones take milliseconds.
- *Stale or damaged model file*: delete it from `./models/` and restart; it is downloaded again.

## 6. Limitations

The restorers make blurred and clean images slightly worse (a 24x compression cannot keep fine detail); the soft gate is nearly one-hot because it starts from an accurate classifier; Optuna budgets were short. Details and numbers are in the report.

## Credits

Oxford-IIIT Pet (Parkhi et al.), FS2K (Fan et al., "Facial-Sketch Synthesis: A New Challenge"). AI tools used: Claude (code, debugging, drafting) and Google Stitch (interface design); see the AI-use appendix in the report.

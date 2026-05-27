---
name: "docker-competition-submit"
description: "Docker packaging and submission for ML competitions. Invoke when building Docker images, verifying reproducibility, or preparing competition tar submissions."
---

# Docker Competition Submission

Guide for packaging the THU-BDC2026 project into a Docker image and verifying it for competition submission.

## Prerequisites Check

Before building Docker, verify:

1. Training completes successfully: `sh train.sh` produces `output/best_model.pth`
2. Prediction completes: `sh test.sh` produces `output/result.csv`
3. Self-scoring works: `python test/score_self.py` produces reasonable scores

## Dockerfile Requirements

The submission Docker must:

1. **Base image**: `nvidia/cuda:12.1.0-runtime-ubuntu22.04` (or compatible CUDA version)
2. **TA-Lib**: System-level installation before Python package
3. **Python dependencies**: Via `uv sync` or `pip install`
4. **Entrypoint**: Must run both `train.sh` and `test.sh` sequentially
5. **Output**: `result.csv` in the expected output directory

### Minimal Dockerfile Structure

```dockerfile
ARG IMAGE_NAME=nvidia/cuda
FROM ${IMAGE_NAME}:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y wget build-essential python3 python3-pip

# Install TA-Lib system library
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && cd ta-lib && \
    ./configure --prefix=/usr && make -j1 && make install && \
    cd .. && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

WORKDIR /app
COPY . .

RUN pip install uv && uv sync

CMD ["sh", "-c", "sh train.sh && sh test.sh"]
```

## Build Commands

### Build Image

```bash
docker buildx build --platform linux/amd64 \
  --build-arg IMAGE_NAME=nvidia/cuda \
  -t bdc2026 .
```

Key flags:
- `--platform linux/amd64`: Required for competition platform compatibility
- `--build-arg IMAGE_NAME`: Allows CUDA version override

### Export to Tar

```bash
docker save -o <team_name>.tar bdc2026:latest
```

**Naming**: Use team name as the tar filename.

### Check Image Size

```bash
docker images bdc2026
```

Target: < 10GB. If larger, use multi-stage builds and clean apt caches.

## Verification Pipeline

### Level 1: Local Run
```bash
sh train.sh && sh test.sh
# Verify: output/result.csv exists with 5 stocks
```

### Level 2: Docker Compose Run
```bash
docker compose up
# Verify: test/output/result.csv exists
```

This verifies the Docker image runs end-to-end. Check that:
- Training completes without OOM
- Model weights load correctly
- Prediction outputs correct format (stock_id, weight columns)

### Level 3: Batch Scoring Simulation
```bash
# 1. Export tar file
docker save -o <team_name>.tar bdc2026:latest

# 2. Place in test directory
cp <team_name>.tar test/tars/

# 3. Register in file list
echo "<team_name>.tar" >> test/tar_files_list.txt

# 4. Run batch test
python test/test.py       # Linux
python test/test_windows.py  # Windows

# 5. Check result
cat test/result.csv
# Expected: Team Name,Final Score
#           <team_name>,0.018...
```

## Common Issues & Solutions

### TA-Lib Installation Failure
```
Error: talib/common.c: No such file or directory
```
**Fix**: Install system TA-Lib BEFORE Python package. The build order matters.

### Docker OOM During Training
**Symptoms**: Container killed, exit code 137
**Solutions**:
- Reduce `max_stocks_per_sample` from 150 to 100
- Enable `use_gradient_checkpointing: True`
- Disable MultiScale if enabled
- Reduce `d_model` from 128 to 96

### Result Format Mismatch
**Expected format**:
```csv
stock_id,weight
000001,0.2
000002,0.2
000003,0.2
000004,0.2
000005,0.2
```
- Exactly 5 stocks (or fewer)
- Weights sum to exactly 1.0
- Stock IDs are 6-digit strings with leading zeros

### Platform Mismatch
```
WARNING: The requested image's platform (linux/amd64) does not match
```
**Fix**: Always build with `--platform linux/amd64` on ARM machines (Apple Silicon).

### Network Issues During Build
```bash
# Option 1: Use proxy
docker buildx build --build-arg HTTP_PROXY=http://proxy:port ...

# Option 2: Pre-download TA-Lib and COPY into image
# Download manually, add to project, use COPY instead of wget
```

## Pre-Submission Checklist

- [ ] `docker buildx build` succeeds without errors
- [ ] `docker compose up` produces `test/output/result.csv`
- [ ] `result.csv` has correct format (stock_id, weight)
- [ ] Weights sum to 1.0 (verified by `test/score_self.py`)
- [ ] Tar file size is reasonable (< 15GB)
- [ ] Tar file named as team name
- [ ] All model weights included in image
- [ ] No hardcoded absolute paths in code
- [ ] Config uses relative paths (`./data`, `./output`)
- [ ] `uv sync` or `pip install` completes in Docker build

## Image Size Optimization

If image is too large:

```dockerfile
# Clean apt cache
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Remove build dependencies
RUN apt-get remove -y wget build-essential && apt-get autoremove -y

# Use multi-stage build
FROM builder as build
# ... install TA-Lib from source ...

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
COPY --from=build /usr/lib/libta_lib* /usr/lib/
# ... smaller final image ...
```

## Competition-Specific Notes

- Competition uses **沪深300 (CSI 300)** constituent stocks
- Data source: Baostock (`get_stock_data.py`)
- Training data: historical daily bar data (OHLCV + turnover + change%)
- Prediction: top 5 stocks, weights sum to 1.0
- Scoring: weighted return of selected portfolio vs. CSI 300 universe
- Submission format: Docker tar file
- Online verification: `test/test.py` simulates the competition scoring pipeline
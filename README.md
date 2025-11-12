# cali

A Gui for Calcium Imaging Data Visualization, Segmentation and Analysis

[![CI](https://github.com/fdrgsp/cali/actions/workflows/ci.yml/badge.svg)](https://github.com/fdrgsp/fdrgsp/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/fdrgsp/fdrgsp/branch/main/graph/badge.svg)](https://codecov.io/gh/fdrgsp/cali)

[🚧 WIP 🚧]

Update of the analysis and vizualization code of [micromanager-gui](https://github.com/fdrgsp/micromanager-gui).

## To Run

`uvx git+https://github.com/fdrgsp/cali`

**Note:** Cellpose is an optional dependency. To use segmentation features, install with:

- `uvx git+https://github.com/fdrgsp/cali[cp4]` for Cellpose 4.x

- `uvx git+https://github.com/fdrgsp/cali[cp3]` for Cellpose 3.x

## To install

### Using (uv) pip

`(uv) pip install git+https://github.com/fdrgsp/cali`

To install with Cellpose support:

- `(uv) pip install git+https://github.com/fdrgsp/cali[cp4]` for Cellpose 4.x

- `(uv) pip install git+https://github.com/fdrgsp/cali[cp3]` for Cellpose 3.x


### NOTE

#### Building on macOS

If you encounter build errors with `oasis-deconv` (especially SDK-related errors), set these environment variables before installing:

```bash
export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk
export LDFLAGS="-L${SDKROOT}/usr/lib"
```

Then run your installation command.
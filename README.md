# dist-evd-solver

[中文](README_zh.md)

`dist-evd-solver` is a high-performance distributed eigenvalue decomposition (EVD) solver implemented using CUDA and the NVIDIA HPC SDK.

This project aims to explore and implement efficient solutions for large-scale dense symmetric eigenvalue problems, especially on modern GPU accelerators. Currently, it primarily implements the first stage of the two-stage method, which transforms a symmetric matrix into a band-diagonal matrix (`Symmetric to Band-diagonal`) using block Householder transformations. The core algorithm is `sy2sb`, which internally utilizes operations like TSQR (`Tall Skinny QR`) and `syr2k` (Symmetric Rank-2k Update).

## Dependencies

Before building this project, please ensure your system meets the following dependencies:

- **NVIDIA HPC SDK (nvhpc)**: Version `25.3` or higher. The project relies on several components from the SDK, including `CUDA`, `CUBLAS`, and `CUSOLVER`.
- **CMake**: Version `3.25` or higher.
- **C++ Compiler**: Must support the C++17 standard.

The following dependencies will be automatically downloaded and configured by CMake's `FetchContent` module at build time, so no manual installation is required:

- **fmt**: `11.2.0` - A modern C++ formatting library.
- **argh**: `v1.3.2` - A lightweight C++ command-line argument parsing library.
- **googletest**: `v1.17.0` - Google's C++ testing framework.

## How to Build

You can build the project using CMake with the following steps:

1.  **Clone the repository**
    ```bash
    git clone <repository-url>
    cd dist-evd-solver
    ```

2.  **Load the dependency environment**
    Before configuring the project with CMake, make sure to load the environment module for the NVIDIA HPC SDK.

3.  **Create a build directory and run CMake**
    ```bash
    mkdir build
    cd build
    cmake ..
    ```

4.  **Compile the project**
    ```bash
    make -j
    ```
    After successful compilation, the executable `dist-evd-solver` will be generated in the `build` directory.

## How to Run

This project is designed to be run in an HPC environment using a job scheduling system like Slurm. You can refer to the `.sbatch` scripts in the repository to configure and submit jobs.

### Slurm Job Script Example

Here is an example of a run script (`run-evd-h100.sbatch`):
```bash
#!/bin/zsh
#SBATCH --job-name=dist-evd-solver
#SBATCH --gres=gpu:h100_pcie:1
#SBATCH --output=log/job_%j.log
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --partition=gpu7
#SBATCH --mem-per-cpu=2048
#SBATCH --time=1-00:00:00

# Load environment (please configure this yourself)
module load nvhpc-hpcx

echo "--- Running ---"
# Run the executable
./build/dist-evd-solver --float -m=65536 -n=32 -t
echo "--- Done ---" 
```
Submit the job using the `sbatch run-evd-h100.sbatch` command. Log files will be saved in the `log/` directory.

### Command-Line Arguments

The executable `dist-evd-solver` supports the following command-line arguments:

| Argument           | Description                                                                  | Example     |
| ------------------ | ---------------------------------------------------------------------------- | ----------- |
| `--float`          | Use single-precision floating-point (float) for calculations (default).      | `--float`   |
| `--double`         | Use double-precision floating-point (double) for calculations.               | `--double`  |
| `-n, --size=<value>` | Specify the dimension of the input square matrix (N x N).                    | `-n=4096`   |
| `-m=<value>`       | Specify the row dimension of the input matrix (M x N), defaults to N.        | `-m=8192`   |
| `-t, --time`       | Run in timing mode, enabling performance timers for key operations.          | `-t`        |
| `--validate`       | Run validation checks on the results.                                        | `--validate`|
| `-v, --verbose`    | Enable verbose logging for detailed output.                                  | `-v`        |

**Note**: For the specific behavior of the arguments, please refer to the implementation in `src/main.cu`.

## Project Structure

```
.
├── CMakeLists.txt      # CMake configuration file
├── README.md           # Project description (this file)
├── README_zh.md        # Project description (Chinese)
├── build/              # Build output directory
├── log/                # Runtime log directory
├── src/                # Source code directory
│   ├── include/        # Header files
│   ├── matrix_ops/     # Matrix operations implementation (sy2sb, tsqr, syr2k, etc.)
│   ├── workflow/       # Core algorithm workflow
│   └── main.cu         # Main program entry point
└── *.sbatch            # Slurm job script examples
```

## Development Log & TODO

### 2025.7.21

- finished the version of single-gpu sy2sb

### 2025.7.14

- Added `sy2sb` workflow and its recursive implementation.
- Implemented the basic framework for `sy2sb` and the calling logic for `panelQR`.

### 2025.7.10

- Ported the single-GPU version of the `tsqr` operation.
- **TODO**:
  - Printing matrices with a leading dimension (lda) is not yet compatible.
  - Some data copies do not handle `lda` correctly; considering wrapping `cudaMemcpy2D` similar to `cublasSetMatrix`.
  - `sy2sb` is not yet fully compatible with all scenarios.
# dist-evd-solver

[English](README.md)

`dist-evd-solver` 是一个使用 CUDA 和 NVIDIA HPC SDK 实现的高性能分布式特征值分解 (EVD) 求解器。

该项目旨在探索和实现大规模稠密对称矩阵特征值问题的高效解法，特别是在现代 GPU 加速器上。当前主要实现了两阶段法中的第一阶段，即将对称矩阵通过分块豪斯霍尔德变换转化为带状矩阵 (`Symmetric to Band-diagonal`)。

## 依赖项

在构建本项目之前，请确保您的系统满足以下依赖：

- **NVIDIA HPC SDK (nvhpc)**: 版本 `25.3` 或更高。项目依赖于 SDK 中的 `CUDA`, `CUBLAS`, `CUSOLVER` 等多个组件。
- **CMake**: 版本 `3.25` 或更高。
- **C++ 编译器**: 支持 C++17 标准。

以下依赖项将由 CMake 的 `FetchContent` 模块在构建时自动下载和配置，无需手动安装：

- **fmt**: `11.2.0` - 一个现代化的 C++ 格式化库。
- **argh**: `v1.3.2` - 一个轻量级的 C++ 命令行参数解析库。
- **googletest**: `v1.17.0` - Google 的 C++ 测试框架。

## 如何构建

您可以按照以下步骤使用 CMake 构建项目：

1.  **克隆仓库**
    ```bash
    git clone <repository-url>
    cd dist-evd-solver
    ```

2.  **加载依赖环境**
    在使用 CMake 配置项目之前，请确保已加载 NVIDIA HPC SDK 的环境模块。

3.  **创建构建目录并运行 CMake**
    ```bash
    mkdir build
    cd build
    cmake ..
    ```

4.  **编译项目**
    ```bash
    make -j
    ```
    编译成功后，将在 `build` 目录下生成可执行文件 `dist-evd-solver`。

## 如何运行

本项目设计为在 HPC 环境下通过作业调度系统（如 Slurm）运行。您可以参考仓库中的 `.sbatch` 脚本来配置和提交作业。

### Slurm 作业脚本示例

以下是一个运行示例 (`run-evd-h100.sbatch`):
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

# 加载环境(请自行配置)
module load nvhpc-hpcx

echo "--- Running ---"
# 运行可执行文件
./build/dist-evd-solver --float -m=65536 -n=32 -t
echo "--- Done ---" 
```
使用 `sbatch run-evd-h100.sbatch` 命令提交作业。日志文件将保存在 `log/` 目录下。

### 命令行参数

可执行文件 `dist-evd-solver` 支持以下命令行参数：

| 参数          | 描述                                           | 示例        |
|---------------|------------------------------------------------|-------------|
| `--float`     | 使用单精度浮点数 (float) 进行计算，默认为双精度。 | `--float`   |
| `-m=<value>`  | 指定输入方阵的维度 (M x M)。                     | `-m=65536`  |
| `-n=<value>`  | 指定计算中使用的块大小 (Block Size)。            | `-n=32`     |
| `-t`, `--test`| 运行测试或计时模式。                              | `-t`        |

**注意**: 参数的具体行为请参考 `src/main.cu` 中的实现。

## 项目结构

```
.
├── CMakeLists.txt      # CMake 配置文件
├── README.md           # 项目说明 (英文)
├── README_zh.md        # 项目说明 (本文)
├── build/              # 编译输出目录
├── log/                # 运行时日志目录
├── src/                # 源代码目录
│   ├── include/        # 头文件
│   ├── matrix_ops/     # 矩阵操作相关实现
│   ├── workflow/       # 核心算法工作流
│   └── main.cu         # 程序主入口
└── *.sbatch            # Slurm 作业脚本示例
```

## 开发日志与 TODO

### 2025.7.14

- 新增 `sy2sb` 工作流，目前正在实现其递归部分。
- 已添加 `sy2sb` 的基本框架和对 `panelQR` 的调用逻辑。

### 2025.7.10

目前已经完成 `tsqr` 操作单卡版本移植，正在开发

- 未兼容带 lda 的矩阵打印
- 部分数据拷贝未兼容 lda，目前考虑包装 cudaMemcpy2D api 参照 cublasSetMatrix 之类的
- `sy2sb` 尚未完全兼容

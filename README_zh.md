![Torch MUSA_Logo](docs/pdf/images/torch_musa.png)
--------------------------------------------------------------------------------

> **语言**: [English](README.md) | **中文**

<!-- toc -->

- [概览](#概览)
- [使用方法](#使用方法)
- [安装](#安装)
  - [先决条件](#先决条件)
  - [Docker 镜像](#docker-镜像)
  - [通过 Python wheels 安装](#通过-python-wheels-安装)
  - [从源码构建](#从源码构建)
- [MUSA 支持的仓库](#musa-支持的仓库)
  - [torchvision](#torchvision)
  - [torchaudio](#torchaudio)
  - [其他仓库](#其他仓库)
- [许可证](#许可证)

<!-- tocstop -->

## 概览

**torch_musa** 是一个基于 PyTorch 扩展而来的 Python 包。配合 PyTorch 使用，用户可以通过 **torch_musa** 充分发挥摩尔线程（Moore Threads）显卡的算力。

**torch_musa** 的接口形式与 PyTorch 保持一致，这使得已经习惯 PyTorch 的用户可以平滑迁移到 **torch_musa**。因此在具体使用方式上，用户可以直接参考 [PyTorch 官方文档](https://docs.pytorch.org/docs/stable/index.html)，你只需要将设备字符串从 `cpu` 或 `cuda` 切换为 `musa` 即可。

**torch_musa** 还提供了一系列工具，帮助用户进行 CUDA 代码迁移、构建 MUSA 扩展以及调试。相关内容请参考 [README.md](torch_musa/utils/README.md)。

对于一些自研优化功能，例如 **动态双精度转换（Dynamic Double Casting）** 和 **统一内存管理（Unified Memory Management）**，请参考 [README.md](torch_musa/README.md)。

如果你希望使用 C/C++ 编写算子或网络层，我们提供了一个易用且高效的扩展 API，并尽可能减少样板代码，无需额外编写封装代码。可参考 [ResNet50 示例](torch_musa/examples/cpp/README.md)。

--------------------------------------------------------------------------------

## 使用方法

**我们推荐用户参考 [torchada](https://github.com/MooreThreads/torchada)，它可以让你将原本为 CUDA 编写的 torch 脚本直接在 MUSA 平台上运行。**

现在执行 `import torch` 会自动加载 torch_musa，在大多数情况下，你只需要将后端从 **cuda** 切换为 **musa**：

```python
import torch
torch.musa.is_available()  # 应该返回 True

# 创建张量：
a = torch.tensor([1.2, 2.3], dtype=torch.float32, device='musa')
b = torch.tensor([1.2, 2.3], dtype=torch.float32, device='cpu').to('musa')
c = torch.tensor([1.2, 2.3], dtype=torch.float32).musa()

# 以及部分 CUDA 模块或函数：
torch.backends.mudnn.allow_tf32 = True
event = torch.musa.Event()
stream = torch.musa.Stream()
```

对于分布式训练或推理，请使用 **mccl** 作为后端初始化进程组：

```python
import torch.distributed as dist

dist.init_process_group("mccl", rank=rank, world_size=world_size)
```

## 安装

### 先决条件

在安装 torch_musa 之前，请先在机器上安装以下软件包：

- （针对 Docker 用户）**[KUAE Cloud Native Toolkits](https://developer.mthreads.com/sdk/download/CloudNative?equipment=&os=&driverVersion=&version=)**，包括 Container-Toolkit、MTML、sGPU；
- **[MUSA-SDK](https://developer.mthreads.com/sdk/download/musa?equipment=&os=&driverVersion=&version=)**，包括 MUSA Driver、musa_toolkit、muDNN 和 MCCL（仅 *S4000* 支持）；
- **其他依赖库**，包括 [muThrust](https://github.com/MooreThreads/muThrust) 和 [muAlg](https://github.com/MooreThreads/muAlg)。

### Docker 镜像

我们在 [mcconline](https://mcconline.mthreads.com/repo) 上提供了若干已经发布的 Docker 镜像。

使用 `docker pull` 拉取镜像后，可以通过如下命令创建容器并进入：

```bash
# 例如，启动一个带 Python3.10 的 S80 Docker 容器
docker run -it --privileged \
  --pull always --network=host \
  --env MTHREADS_VISIBLE_DEVICES=all \
  registry.mthreads.com/mcconline/musa-pytorch-release-public:latest /bin/bash
```

容器在首次启动时会进行自检，你可以通过输出信息确认 MUSA 环境是否已经准备就绪。

### 通过 Python wheels 安装

从我们的 [Release 页面](https://github.com/MooreThreads/torch_musa/releases) 下载对应版本的 torch 与 torch_musa 的 wheel 包，
并确保所有先决条件已经正确安装。

### 从源码构建

首先克隆 torch_musa 仓库：

```bash
git clone https://github.com/MooreThreads/torch_musa.git
cd torch_musa
```

然后，为构建 torch_musa 设置 `MUSA_HOME`、`LD_LIBRARY_PATH` 和 `PYTORCH_REPO_PATH`：

```bash
export MUSA_HOME=path/to/musa_libraries(including mudnn and musa_toolkits) # 默认值为 /usr/local/musa/
export LD_LIBRARY_PATH=$MUSA_HOME/lib:$LD_LIBRARY_PATH
# 如果未设置 PYTORCH_REPO_PATH，在使用 build.sh 构建时将会在该目录外单独下载 PyTorch
export PYTORCH_REPO_PATH=/path/to/PyTorch
```

要构建 torch_musa，运行：

```bash
bash build.sh -c  # 清理缓存并从头开始构建 PyTorch 和 torch_musa
```

部分重要的构建参数说明如下：
- `--torch` / `-t`: 仅构建原生 PyTorch
- `--musa` / `-m`: 仅构建 torch_musa
- `--debug`: 使用调试模式构建
- `--asan`: 使用 ASan 模式构建
- `--clean` / `-c`: 清理所有已构建内容并重新构建
- `--wheel` / `-w`: 生成 wheel 包

例如，如果已经构建过 PyTorch，只需要重新构建 torch_musa 并生成 wheel 包，可以运行：

```bash
bash build.sh -m -w
```

对于 **S80/S3000** 用户，由于这些架构暂不提供 MCCL 库，在构建 torch_musa 时请添加 `USE_MCCL=0`：

```bash
USE_MCCL=0 bash build.sh -c
```

## MUSA 支持的仓库

### torchvision

对 torch_musa v2.7.0 及之后的版本，请从我们的 [vision 仓库](https://github.com/MooreThreads/vision) 安装 torchvision：

```shell
git clone https://github.com/MooreThreads/vision -b v0.22.1-musa --depth 1
cd vision && python setup.py install
```

否则，请从 [官方 torch 仓库](https://github.com/pytorch/vision) 安装 torchvision：

```shell
git clone https://github.com/pytorch/vision -b ${version} --depth 1
cd vision && python setup.py install
```

其中 `version` 取决于你使用的 torch 版本，例如当 torch 版本为 v2.5.0 时，`${version}` 应为 `0.20.0`。

如果在构建过程中出现无法找到 NVCC、CUDA 等相关错误，通常可以在 `setup.py` 顶部添加如下代码来解决：

```python
import torch_musa
```

### torchaudio

请从 [torch 官方 audio 仓库](https://github.com/pytorch/audio) 安装 torchaudio：

```shell
git clone https://github.com/pytorch/audio.git -b release/${version} --depth 1
cd audio && python setup.py install
```

其中 `version` 与 torch 版本保持一致。

与 torchvision 类似，某些情况下也可能需要在 `setup.py` 开头添加一行 ```import torch_musa``` 才能正常使用。

### 其他仓库

已经有许多仓库在上游支持了 MUSA 后端，
例如 [Transformers](https://github.com/huggingface/transformers.git)、[Accelerate](https://github.com/huggingface/accelerate.git)，
可以直接通过 `pip install [repo-name]` 从 PyPI 安装。

对于暂未支持 MUSA 的仓库，我们在 [MooreThreads GitHub](https://github.com/MooreThreads) 下提供了经过 MUSA 适配的版本，部分列表如下：

| 仓库 | 分支 | 链接 | 构建命令 |
| :-- | :-: | :-: | :-: |
| pytorch3d | musa-dev | https://github.com/MooreThreads/pytorch3d | python setup.py install |
| pytorch_sparse | master | https://github.com/MooreThreads/pytorch_sparse | python setup.py install |
| pytorch_scatter | master | https://github.com/MooreThreads/pytorch_scatter | python setup.py install |
| torchvision | v0.22.1-musa | https://github.com/MooreThreads/vision | python setup.py install |
| pytorch_lightning | musa-dev | https://github.com/MooreThreads/pytorch-lightning | python setup.py install |

如果用户在使用这些仓库时遇到任何问题，请在 torch_musa 仓库中提交 issue。
如果你自行完成了某个仓库的 MUSA 适配，也欢迎通过 Pull Request 的方式贡献，帮助我们完善与扩展这份列表。

## 许可证

torch_musa 使用 BSD 风格许可证，详情请参见 [LICENSE](LICENSE) 文件。


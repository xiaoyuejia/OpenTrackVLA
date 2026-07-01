#!/bin/bash
#
# TrackVLA 服务端启动脚本
# 用途：激活 conda 环境并启动推理服务器
#
# 使用方法：
#   ./start_trackvla_server.sh [--port PORT] [--device DEVICE]
#
# 示例：
#   ./start_trackvla_server.sh                    # 默认端口 12180, GPU
#   ./start_trackvla_server.sh --port 8080        # 指定端口
#   ./start_trackvla_server.sh --device cpu       # 使用 CPU
#   ./start_trackvla_server.sh --setup-ros        # 仅创建 ROS 工作空间
#

set -e

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_PATH="${HOME}/miniconda3"
CONDA_ENV="omtrack"
DEFAULT_PORT=12180
DEFAULT_DEVICE="cuda"

# ROS 工作空间配置
ROS_WS_DIR="$(dirname "${SCRIPT_DIR}")/trackvla_ros_ws"
TRACKVLA_ROS_PKG="${SCRIPT_DIR}/TrackVLA-ROS"

# 解析参数
PORT=$DEFAULT_PORT
DEVICE=$DEFAULT_DEVICE
MODEL_PATH="om-ai-lab/OpenTrackVLA"
SETUP_ROS_ONLY=false
USE_LOCAL_MODEL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --local)
            USE_LOCAL_MODEL=true
            shift
            ;;
        --setup-ros)
            SETUP_ROS_ONLY=true
            shift
            ;;
        --ros-ws)
            ROS_WS_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --port PORT      服务端口 (默认: 12180)"
            echo "  --device DEVICE  推理设备: cuda/cpu (默认: cuda)"
            echo "  --model PATH     模型路径或 HuggingFace ID"
            echo "  --local          使用本地 pretrained_model 目录"
            echo "  --setup-ros      仅创建 ROS 工作空间符号链接，不启动服务器"
            echo "  --ros-ws PATH    ROS 工作空间路径 (默认: ../trackvla_ros_ws)"
            echo "  -h, --help       显示帮助"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# ROS 工作空间设置 (仅创建目录和符号链接)
# =============================================================================
setup_ros_workspace() {
    echo ""
    echo "=========================================="
    echo "  设置 ROS 工作空间"
    echo "=========================================="
    echo "ROS 工作空间: ${ROS_WS_DIR}"
    echo "TrackVLA ROS 包: ${TRACKVLA_ROS_PKG}"
    echo ""

    # 检查 TrackVLA-ROS 是否存在
    if [ ! -d "${TRACKVLA_ROS_PKG}" ]; then
        echo "错误: TrackVLA-ROS 包不存在: ${TRACKVLA_ROS_PKG}"
        echo "请先运行: git submodule update --init --recursive"
        exit 1
    fi

    # 创建 ROS 工作空间目录结构
    mkdir -p "${ROS_WS_DIR}/src"

    # 创建符号链接
    LINK_TARGET="${ROS_WS_DIR}/src/trackvla_ros"
    if [ -L "${LINK_TARGET}" ]; then
        echo "符号链接已存在: ${LINK_TARGET}"
    elif [ -d "${LINK_TARGET}" ]; then
        echo "警告: ${LINK_TARGET} 是目录而非符号链接，跳过"
    else
        ln -s "${TRACKVLA_ROS_PKG}" "${LINK_TARGET}"
        echo "✓ 创建符号链接: ${LINK_TARGET} -> ${TRACKVLA_ROS_PKG}"
    fi

    echo ""
    echo "=========================================="
    echo "  ROS 工作空间设置完成!"
    echo "=========================================="
    echo ""
    echo "下一步: cd ${ROS_WS_DIR} && colcon build --symlink-install"
    echo "  2. source install/setup.bash"
    echo "  3. ros2 launch trackvla_ros trackvla.launch.py"
    echo ""
}

# 如果只需要设置 ROS，执行后退出
if [ "$SETUP_ROS_ONLY" = true ]; then
    setup_ros_workspace
    exit 0
fi

# =============================================================================
# 服务器启动流程
# =============================================================================

# 检查 conda
if [ ! -f "${CONDA_PATH}/etc/profile.d/conda.sh" ]; then
    # 尝试 anaconda3
    CONDA_PATH="${HOME}/anaconda3"
    if [ ! -f "${CONDA_PATH}/etc/profile.d/conda.sh" ]; then
        echo "错误: 未找到 conda 安装"
        exit 1
    fi
fi

echo "=========================================="
echo "  TrackVLA 推理服务器"
echo "=========================================="
echo "工作目录: ${SCRIPT_DIR}"
echo "Conda 环境: ${CONDA_ENV}"
echo "端口: ${PORT}"
echo "设备: ${DEVICE}"
echo "模型: ${MODEL_PATH}"
echo "=========================================="

# 设置环境变量，优先使用 ModelScope
export USE_MODELSCOPE=1
# 注意: 不设置 HF_HUB_OFFLINE，让 HuggingFace 作为 fallback

# 如果使用本地模型，更新 MODEL_PATH
if [ "$USE_LOCAL_MODEL" = true ]; then
    LOCAL_MODEL="${SCRIPT_DIR}/pretrained_model"
    if [ -d "${LOCAL_MODEL}" ] && [ -f "${LOCAL_MODEL}/config.json" ]; then
        MODEL_PATH="${LOCAL_MODEL}"
        echo "使用本地模型: ${MODEL_PATH}"
    else
        echo "警告: 本地模型目录不存在或无效: ${LOCAL_MODEL}"
        echo "请先下载模型到 pretrained_model/ 目录"
    fi
fi

# 自动设置 ROS 工作空间（如果不存在）
if [ ! -L "${ROS_WS_DIR}/src/trackvla_ros" ] && [ ! -d "${ROS_WS_DIR}/src/trackvla_ros" ]; then
    echo ""
    echo "[0/3] 首次运行，设置 ROS 工作空间..."
    setup_ros_workspace
fi

# 激活 conda 环境
echo "[1/2] 激活 conda 环境: ${CONDA_ENV}"
source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}

# 检查环境
python --version
echo "PyTorch CUDA 可用: $(python -c 'import torch; print(torch.cuda.is_available())')"

# 启动服务器
echo "[2/2] 启动服务器..."
cd "${SCRIPT_DIR}"
python TrackVLA-ROS/trackvla_ros/server/trackvla_server.py \
    --port ${PORT} \
    --device ${DEVICE} \
    --model "${MODEL_PATH}"

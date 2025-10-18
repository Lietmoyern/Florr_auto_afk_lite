#!/bin/bash


# 检查Python是否安装
if ! command -v python3 &> /dev/null
then
    echo "Python 3 未安装，请先安装Python 3"
    echo "您可以从App Store或官网(https://www.python.org/downloads/macos/)下载安装"
    exit 1
fi

# 检查pip是否安装
if ! command -v pip3 &> /dev/null
then
    echo "pip 未安装，尝试安装pip..."
    python3 -m ensurepip --upgrade
fi

# 检查Homebrew是否安装
if ! command -v brew &> /dev/null
then
    echo "Homebrew 未安装，正在安装Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # 将Homebrew添加到PATH
    if [ -f "/opt/homebrew/bin/brew" ]; then
        # Apple Silicon Macs
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [ -f "/usr/local/bin/brew" ]; then
        # Intel Macs
        echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/usr/local/bin/brew shellenv)"
    fi
fi

# 安装系统依赖
echo "正在安装系统依赖..."
brew install cmake pkg-config || echo "某些依赖安装失败，但继续执行..."

# 检查是否需要安装其他依赖
for dep in jpeg libpng libtiff openexr eigen tbb; do
    if ! brew list $dep &>/dev/null; then
        brew install $dep || echo "跳过 $dep 的安装"
    fi
done

# 升级pip和安装构建工具
echo "正在升级pip和构建工具..."
pip3 install --upgrade pip wheel setuptools

# 安装必要的依赖
echo "正在安装必要的Python依赖库..."

# 创建不包含opencv的临时requirements文件
if [ -f "requirements.txt" ]; then
    grep -v "opencv" requirements.txt > temp_requirements.txt 2>/dev/null || true
    
    # 先安装其他依赖
    if [ -s "temp_requirements.txt" ]; then
        echo "正在安装其他Python依赖..."
        pip3 install -r temp_requirements.txt
    fi
    rm -f temp_requirements.txt
fi

# 尝试安装OpenCV
echo "正在安装OpenCV..."
install_opencv() {
    # 方法1: 使用预编译的wheel
    echo "尝试方法1: 使用预编译wheel..."
    pip3 install --no-cache-dir opencv-contrib-python --only-binary=opencv-contrib-python && return 0
    
    # 方法2: 尝试不包含contrib的版本
    echo "方法1失败，尝试方法2: 安装基础版本..."
    pip3 install --no-cache-dir opencv-python --only-binary=opencv-python && return 0
    
    # 方法3: 不使用二进制，从源码编译
    echo "方法2失败，尝试方法3: 从源码编译..."
    pip3 install --no-cache-dir opencv-python || return 1
    
    return 0
}

if ! install_opencv; then
    echo "警告: OpenCV安装失败，程序可能无法正常运行"
    echo "您可以尝试手动安装: pip3 install opencv-python"
fi

# 验证OpenCV安装
if python3 -c "import cv2; print(f'OpenCV版本: {cv2.__version__}')" 2>/dev/null; then
    echo "✓ OpenCV安装成功"
else
    echo "✗ OpenCV安装失败，但继续启动程序..."
fi

# 运行主程序
echo "依赖安装完成，正在启动程序..."
python3 main.py
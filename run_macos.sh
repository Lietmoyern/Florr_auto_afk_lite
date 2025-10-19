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

# 安装必要的依赖
echo "正在安装必要的Python依赖库..."
pip3 install -r requirements.txt

# 运行主程序
echo "依赖安装完成，正在启动程序..."
python3 main.py

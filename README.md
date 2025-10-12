# florr_auto_afk_lite

## 项目简介
florr_auto_afk_lite 是Shiny-Ladybug制作的Florr_Auto_Afk项目为应用于MacOS的简化版，原项目https://github.com/Shiny-Ladybug/florr-auto-afk" ,本简化版只保留了其用于解决afk问题的功能，其他辅助功能都被移除了,本简化版由Lietmoyern制作，只做了简化和重组工程结构而没有增添新的功能。


## 使用方法

### 1. 配置参数

在运行前，可以根据需要修改`config.json`文件中的参数：

```json
{
  "afk_detection": {
    "interval": 5,              // 检测间隔(秒)
    "confidence_threshold": 0.7, // 置信度阈值
    "window_ratio_tolerance": 0.1, // 窗口比例容差
    "window_aspect_ratios": [1.0, 0.787] // 目标窗口宽高比
  },
  "mouse": {
    "speed": 100,              // 鼠标移动速度
    "failsafe": true           // 是否启用安全模式
  },
  "keyboard": {
    "key_press_delay": 0.15    // 按键延迟
  },
  "image_processing": {
    "rdp_epsilon": "0.2173913043478*width+0.4782608695652", // RDP简化参数
    "extend_length": 30,       // 路径延长长度
    "color_tolerance": 40      // 颜色容差
  },
  "timing": {
    "page_opacity_wait": 0.5,  // 页面透明度等待时间
    "post_action_wait": 1,     // 操作后等待时间
    "auto_exit_minutes": -1    // 运行时间(-1表示不自动退出)
  },
  "method": {
    "use_afk_bw": false        // 处理方法(false=分割模型, true=黑白处理)
  }
}
```

### 2. 运行程序

确保已安装所需的Python库：
```bash
pip3 install opencv-python numpy scipy pyautogui torch ultralytics scikit-image fastapi uvicorn websockets rdp
```

```bash
python main.py
```

程序启动后，将自动进行以下操作：
- 加载配置和模型
- 启动WebSocket服务器（如果配置为使用afk_bw方法）
- 开始周期性检测屏幕上的AFK窗口
- 检测到AFK窗口时，自动规划路径并执行操作
- 操作完成后，继续监控

## 工作原理

### 1. AFK窗口检测流程

1. 定时截取屏幕图像
2. 使用YOLO模型检测图像中的窗口对象
3. 验证检测到的窗口宽高比是否符合预期
4. 如果符合条件，则认为是AFK窗口

### 2. 路径规划与执行

根据配置的方法，程序会使用以下两种方式之一处理AFK窗口：

#### 黑白图像处理方法 (`use_afk_bw: true`)
- 将网页半透明转不透明(必须安装`set_opaque.js`脚本)
- 识别起点（彩色圆点）和终点（黑色圆点）
- 使用Dijkstra算法计算从起点到终点的最优路径
- 使用RDP算法简化路径
- 控制鼠标沿路径移动

#### 图像分割方法 (`use_afk_bw: false`)
- 使用分割模型提取路径区域
- 构建路径骨架
- 基于骨架生成平滑路径
- 控制鼠标沿路径移动

### 3. 脚本控制机制

程序启动时会创建一个WebSocket服务器，默认监听`localhost:8765`。油猴脚本`set_opaque.js`通过WebSocket客户端连接到本地服务器，并发送命令控制是否将网页转为黑白
<div class="warning-box">
  ⚠️ **警告：** M28正在查封油猴脚本，不建议安装`set_opaque.js·！
</div>

## 项目结构

```
florr_auto_afk_lite/
├── main.py            # 主程序入口，协调各模块工作
├── detector.py        # 包含YOLO模型检测功能
├── helpful_functions.py # 包含路径规划和执行的核心逻辑
├── connection.py      # WebSocket服务器实现
├── config.json        # 配置文件
├── set_opaque.js      # 浏览器脚本，用于控制网页透明度
├── models/            # 存储YOLO模型文件
│   ├── afk-det.pt     # AFK窗口检测模型
│   └── afk-seg.pt     # 图像分割模型
├── logs/              # 存储截图日志
├── requirements.txt   # 项目依赖列表
└── README.md          # 项目说明文档
```
## 许可证

该项目使用和源项目相同的GPLv3许可证，详见项目根目录下的LICENSE文件。

## 免责声明


此工具仅用于学习和研究目的。使用此工具可能违反马修斯28的服务条款。请在使用前了解并遵守相关规定。作者不对使用此工具可能导致的任何后果负责。




#!/usr/bin/env python3
import os, sys, re, subprocess, time, webbrowser, signal, atexit, threading

# ==================== 配置区域 ====================
DEFAULT_DIR = os.path.dirname(os.path.abspath(__file__))
JUPYTER_TIMEOUT = 20
PIP_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"

# ANSI 颜色定义
C_GREEN = "\033[32m"
C_CYAN = "\033[36m"
C_YELLOW = "\033[33m"
C_RED = "\033[31m"
C_GRAY = "\033[90m"
C_BOLD = "\033[1m"
C_END = "\033[0m"
# =================================================

def show_status(text, stop_event):
    """显示动态旋转的加载指示器"""
    spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    i = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\r {C_CYAN}{spinner[i % len(spinner)]}{C_END} {text}...")
        sys.stdout.flush()
        time.sleep(0.08)
        i += 1
    sys.stdout.write("\r" + " " * 60 + "\r") # 清除行

def get_python_env(target_dir):
    """确保虚拟环境存在并返回 Python 路径"""
    venv_dir = os.path.join(target_dir, 'venv')
    py_name = 'python.exe' if os.name == 'nt' else 'python3'
    py_path = os.path.join(venv_dir, 'Scripts' if os.name == 'nt' else 'bin', py_name)

    if not os.path.exists(py_path):
        stop_event = threading.Event()
        t = threading.Thread(target=show_status, args=("正在初始化虚拟环境", stop_event))
        t.start()
        subprocess.run([sys.executable, '-m', 'venv', venv_dir], check=True)
        stop_event.set()
        t.join()
        print(f" {C_GREEN}✔{C_END} 虚拟环境创建成功")
    
    return py_path

def install_deps(py_path):
    """静默检查并安装依赖"""
    try:
        subprocess.run([py_path, '-m', 'pip', 'show', 'notebook'], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        stop_event = threading.Event()
        t = threading.Thread(target=show_status, args=("正在安装 Jupyter Notebook", stop_event))
        t.start()
        subprocess.run([py_path, '-m', 'pip', 'install', '-i', PIP_MIRROR, 'notebook'], check=True, capture_output=True)
        stop_event.set()
        t.join()
        print(f" {C_GREEN}✔{C_END} 依赖库安装完成")

def run_and_manage_jupyter(python_path, target_dir):
    """启动逻辑：解决加载阻塞、匹配失败与美化输出"""
    # 强制 UTF-8 环境，解决 Python 3.13+ 的 Locale 问题
    env = {**os.environ, 'PYTHONUTF8': '1', 'LANG': 'en_US.UTF-8', 'LC_ALL': 'en_US.UTF-8'}
    cmd = [python_path, '-m', 'notebook', '--no-browser']
    
    stop_event = threading.Event()
    loader = threading.Thread(target=show_status, args=("正在唤醒 Jupyter 引擎", stop_event))
    loader.start()

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='replace', cwd=target_dir, bufsize=1
    )

    # 自动清理机制
    atexit.register(lambda: process.terminate() if process.poll() is None else None)
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

    url = None
    start_time = time.time()
    
    try:
        while time.time() - start_time < JUPYTER_TIMEOUT:
            line = process.stdout.readline()
            if not line and process.poll() is not None: break
            
            # 增强匹配模式：适配 tree/lab 以及各种 token 格式
            match = re.search(r'(http://(?:localhost|127\.0\.0\.1):\d+/\S*token=[a-f0-9]+)', line)
            if match:
                url = match.group(1)
                break
            
            # 备选匹配：如果发现文件路径也视为服务已启动
            if "jpserver-" in line and ".html" in line:
                url = "http://127.0.0.1:8888/tree"
                break
    finally:
        stop_event.set()
        loader.join()

    if url:
        print(f" {C_GREEN}●{C_END} {C_BOLD}环境就绪!{C_END}")
        print(f" {C_CYAN}╰─▶{C_END} {url}\n")
        
        # 异步打开浏览器，防止主线程阻塞
        threading.Thread(target=lambda: (time.sleep(0.5), webbrowser.open(url)), daemon=True).start()
        
        print(f"{C_GRAY}{'─'*60}{C_END}")
        print(f"{C_GRAY}实时日志流 (按 Ctrl+C 安全退出):{C_END}")
        
        # 接管输出，防止缓冲区满导致进程挂起
        try:
            for log_line in process.stdout:
                sys.stdout.write(f"{C_GRAY}{log_line}{C_END}")
                sys.stdout.flush()
        except KeyboardInterrupt:
            print(f"\n{C_YELLOW}[!] 收到退出指令，正在关闭服务...{C_END}")
    else:
        process.terminate()
        print(f" {C_RED}✘ 错误：无法捕获启动链接。{C_END}")
        print(f" {C_YELLOW}建议：手动运行 '{os.path.relpath(python_path)} -m notebook' 查看具体报错。{C_END}")

def main():
    # 路径解析
    path_arg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    target_dir = os.path.abspath(path_arg)
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 打印欢迎语
    print(f"\n{C_BOLD}Jupyter 自动化环境管理器{C_END}")
    print(f"{C_GRAY}工作目录: {target_dir}{C_END}\n")

    try:
        py_path = get_python_env(target_dir)
        install_deps(py_path)
        run_and_manage_jupyter(py_path, target_dir)
    except Exception as e:
        print(f"\n{C_RED}[!] 运行脚本时发生崩溃: {e}{C_END}")
        sys.exit(1)

if __name__ == '__main__':
    main()
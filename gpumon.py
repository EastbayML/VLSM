import curses
import psutil
import pynvml

def initialize_nvml():
    pynvml.nvmlInit()

def shutdown_nvml():
    pynvml.nvmlShutdown()

def get_gpu_processes():
    gpu_processes = []
    device_count = pynvml.nvmlDeviceGetCount()
    
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        gpu_index = pynvml.nvmlDeviceGetIndex(handle)
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        
        for process in processes:
            gpu_processes.append({
                'gpu_index': gpu_index,
                'pid': process.pid,
                'used_memory': process.usedGpuMemory / (1024 * 1024)  # Convert to MB
            })
    
    return gpu_processes

def get_gpu_utilization():
    gpu_utilizations = {}
    device_count = pynvml.nvmlDeviceGetCount()
    
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_utilizations[i]=utilization.gpu
    
    return gpu_utilizations

def monitor_resources(stdscr, interval=1):
    curses.curs_set(0)  # Hide cursor
    stdscr.nodelay(1)   # Non-blocking input
    initialize_nvml()
    
    try:
        while True:
            stdscr.clear()
            
            # Get CPU usage
            cpu_usage = psutil.cpu_percent(interval=interval)
            
            # Get memory usage
            memory_info = psutil.virtual_memory()
            memory_usage = memory_info.percent
            
            # Get GPU processes
            gpu_processes = get_gpu_processes()
            
            gpu_util = get_gpu_utilization()

            # Add column titles
            stdscr.addstr(0, 0, f"{'PID':<8}{'Command Line':<30}{'CPU%':>6}{'Mem':>6}M   {'GPU':>4}{'Mem':>6}M{'util':>6}" )
            line = 1
            
            process_gpu_map = {}
            for process in gpu_processes:
                if process['pid'] not in process_gpu_map:
                    process_gpu_map[process['pid']] = []
                process_gpu_map[process['pid']].append({
                    'gpu_index': process['gpu_index'],
                    'used_memory': process['used_memory']
                })
            
            for pid, gpu_info in process_gpu_map.items():
                try:
                    p = psutil.Process(pid)
                    cmdline = ' '.join(p.cmdline())
                    cmd=cmdline.rsplit('/',1)[-1]
                    cpu_percent = p.cpu_percent(interval=0.1)
                    mem_info = p.memory_info()
                    mem_percent = mem_info.rss / psutil.virtual_memory().total * 100
                    
                    for gpu in gpu_info:
                        stdscr.addstr(line, 0, f"{pid:<8}{cmd:<30}{cpu_percent:>6.1f} {mem_percent:>6.0f}   {gpu['gpu_index']:>4} {gpu['used_memory']:>6.0f} {gpu_util[gpu['gpu_index']]:>4.0f}%")
                        line += 1
                except psutil.NoSuchProcess:
                    continue
            
            stdscr.refresh()
            curses.napms(interval * 1000)
    finally:
        shutdown_nvml()

if __name__ == "__main__":
    curses.wrapper(monitor_resources)
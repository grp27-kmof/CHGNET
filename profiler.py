import time
import psutil
import torch

try:
    from pynvml import NVMLError, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
    from pynvml import nvmlDeviceGetUtilizationRates, nvmlInit
except ImportError:
    NVMLError = None
    nvmlDeviceGetHandleByIndex = None
    nvmlDeviceGetMemoryInfo = None
    nvmlDeviceGetUtilizationRates = None
    nvmlInit = None


def _get_nvml_handle():
    if nvmlInit is None or nvmlDeviceGetHandleByIndex is None:
        return None
    try:
        nvmlInit()
        return nvmlDeviceGetHandleByIndex(0)
    except NVMLError:
        return None


handle = _get_nvml_handle()


def _gpu_stats():
    if handle is None:
        return None, None
    return nvmlDeviceGetUtilizationRates(handle), nvmlDeviceGetMemoryInfo(handle)


class StageProfiler:
    def __init__(self):
        self.start = None
        self.stage = ""

    def begin(self, stage):
        self.stage = stage
        self.start = time.perf_counter()

        # Reset peak GPU memory for this stage
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        cpu = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory()

        util, mem = _gpu_stats()

        pytorch_mem = 0
        if torch.cuda.is_available():
            pytorch_mem = torch.cuda.memory_allocated() / (1024 ** 2)

        print("\n" + "=" * 70)
        print(f" STARTING : {stage}")
        print("=" * 70)
        print(f"CPU Usage            : {cpu:.1f}%")
        print(f"RAM Used             : {ram.used / (1024 ** 3):.2f} GB")
        print(f"RAM Available        : {ram.available / (1024 ** 3):.2f} GB")
        if util is not None and mem is not None:
            print(f"GPU Utilization      : {util.gpu}%")
            print(f"GPU Memory Used      : {mem.used / (1024 ** 2):.1f} MB")
            print(f"GPU Memory Total     : {mem.total / (1024 ** 2):.1f} MB")
        else:
            print("GPU Telemetry        : unavailable (non-NVIDIA device)")
        print(f"PyTorch GPU Memory   : {pytorch_mem:.1f} MB")
        print("=" * 70)

    def end(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - self.start

        cpu = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory()

        util, mem = _gpu_stats()

        pytorch_mem = 0
        peak_mem = 0

        if torch.cuda.is_available():
            pytorch_mem = torch.cuda.memory_allocated() / (1024 ** 2)
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

        print("\n" + "-" * 70)
        print(f" COMPLETED : {self.stage}")
        print("-" * 70)
        print(f"Elapsed Time         : {elapsed:.2f} sec")
        print(f"CPU Usage            : {cpu:.1f}%")
        print(f"RAM Used             : {ram.used / (1024 ** 3):.2f} GB")
        if util is not None and mem is not None:
            print(f"GPU Utilization      : {util.gpu}%")
            print(f"GPU Memory Used      : {mem.used / (1024 ** 2):.1f} MB")
            print(f"GPU Memory Total     : {mem.total / (1024 ** 2):.1f} MB")
        else:
            print("GPU Telemetry        : unavailable (non-NVIDIA device)")
        print(f"PyTorch GPU Memory   : {pytorch_mem:.1f} MB")
        print(f"PyTorch Peak Memory  : {peak_mem:.1f} MB")
        print("-" * 70 + "\n")


# Optional test
if __name__ == "__main__":
    profiler = StageProfiler()

    profiler.begin("Test Stage")

    # Dummy workload
    if torch.cuda.is_available():
        x = torch.randn(5000, 5000, device="cuda")
        y = torch.matmul(x, x)
        del x, y

    time.sleep(2)

    profiler.end()
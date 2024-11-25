import time
from collections import deque

class TimeRecorder:
    def __init__(self, name: str, maxlen: int, log_after_end_record: bool = True):
        self.name = name
        self.log_after_end_record = log_after_end_record

        self.time_queue = deque(maxlen=maxlen)

        self.time_start_record = None
    
    def __enter__(self):
        """进入上下文时，自动开始计时"""
        self.start_record()
        return self  # 返回自己，方便在with语句内访问

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时，自动结束计时并记录"""
        self.end_record()

    def start_record(self):
        """开始计时，记录当前时间"""
        self.time_start_record = time.perf_counter()  # 使用perf_counter获取高精度时间

    def end_record(self):
        """结束计时，计算时间并记录"""
        if self.time_start_record is None:
            raise ValueError("Timing has not started, cannot end the recording.")
        else:
            # 结束计时，计算时间差并记录
            time_end_record = time.perf_counter()
            elapsed_time = time_end_record - self.time_start_record
            self.time_queue.append(elapsed_time)
            self.time_start_record = None  # 计时结束后重置开始时间
        
        if self.log_after_end_record:
            self.log()
    
    def log(self):
        """打印 time_queue 中所有时间的均值"""
        if len(self.time_queue) == 0:
            raise ValueError("No recorded times to calculate the average.")
        else:
            # 计算并打印平均时间
            avg_time = sum(self.time_queue) / len(self.time_queue)
            print(f"{self.name} average time: {avg_time:.6f} seconds")
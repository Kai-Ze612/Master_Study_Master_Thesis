#!/usr/bin/env python3
"""
Comprehensive System Monitor
Monitors: CPU, Memory, Disk, Network, Thermal, GPU (if available)
"""

import psutil
import time
import os
import glob
from datetime import datetime

class SystemMonitor:
    
    def __init__(self):
        self.gpu_available = self._check_gpu_availability()
    
    def _check_gpu_availability(self):
        """Check if GPU monitoring is available"""
        nvidia_smi = os.path.exists('/usr/bin/nvidia-smi')
        amd_sysfs = os.path.exists('/sys/class/drm/card0/device/hwmon')
        return {'nvidia': nvidia_smi, 'amd': amd_sysfs}
    
    def get_cpu_info(self):
        """Get CPU usage, frequency, and per-core statistics"""
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        cpu_freq = psutil.cpu_freq(percpu=True)
        cpu_count = psutil.cpu_count(logical=True)
        
        print("\n" + "="*60)
        print("CPU INFORMATION")
        print("="*60)
        print(f"Physical Cores: {psutil.cpu_count(logical=False)}")
        print(f"Logical Cores: {cpu_count}")
        print(f"Overall Usage: {psutil.cpu_percent(interval=1):.1f}%")
        
        print("\nPer-Core Statistics:")
        for i, (percent, freq) in enumerate(zip(cpu_percent, cpu_freq if cpu_freq else [None]*len(cpu_percent))):
            freq_str = f"{freq.current:.0f} MHz" if freq else "N/A"
            print(f"  Core {i}: {percent:5.1f}% | Frequency: {freq_str}")
    
    def get_thermal_info(self):
        """Get thermal zone temperatures"""
        print("\n" + "="*60)
        print("THERMAL INFORMATION")
        print("="*60)
        
        zones = glob.glob('/sys/class/thermal/thermal_zone*')
        
        for zone_path in sorted(zones):
            zone_name = os.path.basename(zone_path)
            
            try:
                with open(f'{zone_path}/type', 'r') as f:
                    zone_type = f.read().strip()
                
                with open(f'{zone_path}/temp', 'r') as f:
                    temp_milliC = int(f.read().strip())
                    temp_C = temp_milliC / 1000.0
                
                # Try to read trip points
                trip_points = []
                trip_files = glob.glob(f'{zone_path}/trip_point_*_temp')
                for trip_file in sorted(trip_files)[:2]:  # Show first 2 trip points
                    try:
                        with open(trip_file, 'r') as f:
                            trip_temp = int(f.read().strip()) / 1000.0
                            trip_points.append(f"{trip_temp:.0f}°C")
                    except:
                        pass
                
                trip_str = f" | Trips: {', '.join(trip_points)}" if trip_points else ""
                print(f"{zone_type:20s}: {temp_C:5.1f}°C{trip_str}")
                
            except Exception as e:
                print(f"Error reading {zone_name}: {e}")
    
    def get_memory_info(self):
        """Get memory usage statistics"""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        print("\n" + "="*60)
        print("MEMORY INFORMATION")
        print("="*60)
        print(f"Total RAM:      {mem.total / (1024**3):6.2f} GB")
        print(f"Available:      {mem.available / (1024**3):6.2f} GB")
        print(f"Used:           {mem.used / (1024**3):6.2f} GB ({mem.percent}%)")
        print(f"Free:           {mem.free / (1024**3):6.2f} GB")
        print(f"Buffers/Cache:  {(mem.buffers + mem.cached) / (1024**3):6.2f} GB")
        
        print(f"\nSwap Total:     {swap.total / (1024**3):6.2f} GB")
        print(f"Swap Used:      {swap.used / (1024**3):6.2f} GB ({swap.percent}%)")
        print(f"Swap Free:      {swap.free / (1024**3):6.2f} GB")
    
    def get_disk_info(self):
        """Get disk usage statistics"""
        print("\n" + "="*60)
        print("DISK INFORMATION")
        print("="*60)
        
        partitions = psutil.disk_partitions()
        for partition in partitions:
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                print(f"\nDevice: {partition.device}")
                print(f"  Mountpoint:  {partition.mountpoint}")
                print(f"  Filesystem:  {partition.fstype}")
                print(f"  Total:       {usage.total / (1024**3):6.2f} GB")
                print(f"  Used:        {usage.used / (1024**3):6.2f} GB ({usage.percent}%)")
                print(f"  Free:        {usage.free / (1024**3):6.2f} GB")
            except PermissionError:
                continue
        
        # Disk I/O statistics
        disk_io = psutil.disk_io_counters()
        if disk_io:
            print(f"\nDisk I/O Statistics:")
            print(f"  Read:        {disk_io.read_bytes / (1024**3):6.2f} GB")
            print(f"  Write:       {disk_io.write_bytes / (1024**3):6.2f} GB")
    
    def get_network_info(self):
        """Get network statistics"""
        print("\n" + "="*60)
        print("NETWORK INFORMATION")
        print("="*60)
        
        net_io = psutil.net_io_counters()
        print(f"Bytes Sent:     {net_io.bytes_sent / (1024**3):6.2f} GB")
        print(f"Bytes Received: {net_io.bytes_recv / (1024**3):6.2f} GB")
        print(f"Packets Sent:   {net_io.packets_sent:,}")
        print(f"Packets Recv:   {net_io.packets_recv:,}")
    
    def get_gpu_info_nvidia(self):
        """Get NVIDIA GPU information using nvidia-smi"""
        import subprocess
        
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit', 
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                print("\n" + "="*60)
                print("NVIDIA GPU INFORMATION")
                print("="*60)
                
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = [p.strip() for p in line.split(',')]
                        idx, name, temp, util_gpu, util_mem, mem_used, mem_total, power_draw, power_limit = parts
                        
                        print(f"\nGPU {idx}: {name}")
                        print(f"  Temperature:     {temp}°C")
                        print(f"  GPU Utilization: {util_gpu}%")
                        print(f"  Mem Utilization: {util_mem}%")
                        print(f"  Memory Usage:    {mem_used} MB / {mem_total} MB")
                        print(f"  Power Draw:      {power_draw} W / {power_limit} W")
        except Exception as e:
            print(f"\nNVIDIA GPU monitoring unavailable: {e}")
    
    def get_gpu_info_amd(self):
        """Get AMD GPU information from sysfs"""
        try:
            hwmon_paths = glob.glob('/sys/class/drm/card*/device/hwmon/hwmon*')
            
            if hwmon_paths:
                print("\n" + "="*60)
                print("AMD GPU INFORMATION")
                print("="*60)
                
                for hwmon_path in hwmon_paths:
                    try:
                        # Temperature
                        temp_files = glob.glob(f'{hwmon_path}/temp*_input')
                        for temp_file in temp_files:
                            with open(temp_file, 'r') as f:
                                temp = int(f.read().strip()) / 1000.0
                                print(f"  Temperature: {temp:.1f}°C")
                        
                        # Power
                        power_files = glob.glob(f'{hwmon_path}/power*_average')
                        for power_file in power_files:
                            with open(power_file, 'r') as f:
                                power = int(f.read().strip()) / 1000000.0
                                print(f"  Power Draw:  {power:.1f} W")
                    except:
                        continue
        except Exception as e:
            print(f"\nAMD GPU monitoring unavailable: {e}")
    
    def get_power_info(self):
        """Get power/battery information if available"""
        battery = psutil.sensors_battery()
        
        if battery:
            print("\n" + "="*60)
            print("BATTERY INFORMATION")
            print("="*60)
            print(f"Percentage:     {battery.percent}%")
            print(f"Power Plugged:  {battery.power_plugged}")
            if battery.secsleft != psutil.POWER_TIME_UNLIMITED:
                hours = battery.secsleft // 3600
                minutes = (battery.secsleft % 3600) // 60
                print(f"Time Remaining: {hours}h {minutes}m")
    
    def monitor_once(self):
        """Run all monitoring functions once"""
        print("\n" + "#"*60)
        print(f"System Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("#"*60)
        
        self.get_cpu_info()
        self.get_thermal_info()
        self.get_memory_info()
        self.get_disk_info()
        self.get_network_info()
        
        # GPU monitoring
        if self.gpu_available['nvidia']:
            self.get_gpu_info_nvidia()
        if self.gpu_available['amd']:
            self.get_gpu_info_amd()
        
        self.get_power_info()
    
    def monitor_continuous(self, interval=2):
        """Continuously monitor system"""
        try:
            while True:
                os.system('clear')  # Clear screen
                self.monitor_once()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    import sys
    
    monitor = SystemMonitor()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--continuous':
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        monitor.monitor_continuous(interval)
    else:
        monitor.monitor_once()
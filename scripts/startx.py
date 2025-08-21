#!/usr/bin/env python

import subprocess
import shlex
import re
import platform
import tempfile
import os
import sys

# pci_records 和 generate_xorg_conf 函数保持不变
def pci_records():
    # ... (此函数无需修改)
    records = []
    command = shlex.split('lspci -vmm')
    output = subprocess.check_output(command).decode()

    for devices in output.strip().split("\n\n"):
        record = {}
        records.append(record)
        for row in devices.split("\n"):
            key, value = row.split("\t")
            record[key.split(':')[0]] = value

    return records

def generate_xorg_conf(devices):
    # ... (此函数无需修改)
    xorg_conf = []

    device_section = """
Section "Device"
    Identifier     "Device{device_id}"
    Driver         "nvidia"
    VendorName     "NVIDIA Corporation"
    BusID          "{bus_id}"
EndSection
"""
    server_layout_section = """
Section "ServerLayout"
    Identifier     "Layout0"
    {screen_records}
EndSection
"""
    screen_section = """
Section "Screen"
    Identifier     "Screen{screen_id}"
    Device         "Device{device_id}"
    DefaultDepth    24
    Option         "AllowEmptyInitialConfiguration" "True"
    SubSection     "Display"
        Depth       24
        Virtual 1024 768
    EndSubSection
EndSection
"""
    screen_records = []
    for i, bus_id in enumerate(devices):
        xorg_conf.append(device_section.format(device_id=i, bus_id=bus_id))
        xorg_conf.append(screen_section.format(device_id=i, screen_id=i))
        screen_records.append('Screen {screen_id} "Screen{screen_id}" 0 0'.format(screen_id=i))

    xorg_conf.append(server_layout_section.format(screen_records="\n    ".join(screen_records)))

    output =  "\n".join(xorg_conf)
    output += """
Section "ServerFlags"
    Option "AutoAddGpu" "off"
    Option "AllowMouseOpenFail" "true"
    Option "ProbeAllGpus" "false"
EndSection
    """
    print('-'*40)
    print(output) # 建议在调试时才取消注释
    print('-'*40)
    return output

# --- 主要修改在这里 ---

def startx(display, gpu_index=0): # 增加一个 gpu_index 参数，默认为0
    if platform.system() != 'Linux':
        raise Exception("Can only run startx on linux")

    all_gpus = [] # 先收集所有找到的GPU
    for r in pci_records():
        if r.get('Vendor', '') == 'NVIDIA Corporation' \
                and r['Class'] in ['VGA compatible controller', '3D controller']:
            bus_id = 'PCI:' + ':'.join(map(lambda x: str(int(x, 16)), re.split(r'[:\.]', r['Slot'])))
            all_gpus.append(bus_id)

    if not all_gpus:
        raise Exception("错误：未找到任何NVIDIA显卡")

    print(f"系统中找到 {len(all_gpus)} 个NVIDIA GPU: {all_gpus}")
    
    # 检查用户指定的GPU索引是否有效
    if gpu_index >= len(all_gpus):
        raise IndexError(f"错误：GPU索引 {gpu_index} 超出范围。有效的索引为 0 到 {len(all_gpus) - 1}。")

    # 只选择用户指定的那个GPU
    target_gpu_bus_id = all_gpus[gpu_index]
    print(f"将在 GPU {gpu_index} (BusID: {target_gpu_bus_id}) 上启动Xorg...")
    
    # generate_xorg_conf 需要一个列表，所以把单个BusID放进列表里
    devices_to_configure = [target_gpu_bus_id]

    try:
        fd, path = tempfile.mkstemp()
        # with open(path, "w") as f:
        #     f.write(generate_xorg_conf(devices_to_configure))
        with open("./xorg.conf", "w") as f:
            f.write(generate_xorg_conf(devices_to_configure))
        # print('-'*40)
        # print(generate_xorg_conf(devices_to_configure))
        # print('-'*40)
        # command = shlex.split("Xorg -noreset +extension GLX +extension RANDR +extension RENDER -config %s :%s" % (path, display))
        # subprocess.call(command)
    finally:
        os.close(fd)
        # 确保在脚本退出时能删除临时文件
        if os.path.exists(path):
            os.unlink(path)


if __name__ == '__main__':
    # 允许传入两个参数：display 和 gpu_index
    display = 0
    gpu_index = 0

    if len(sys.argv) > 1:
        display = int(sys.argv[1])
    if len(sys.argv) > 2:
        gpu_index = int(sys.argv[2])
    
    print(f"准备在 DISPLAY=:{display} 上为 GPU {gpu_index} 启动X Server")
    startx(display, gpu_index)
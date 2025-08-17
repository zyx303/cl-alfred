#!/bin/bash

# CL-ALFRED 自动重启下载脚本
# 使用方法：bash download_with_restart.sh

export ALFRED_ROOT=~/work/cl-alfred

echo "======================================"
echo "CL-ALFRED 自动重启下载"
echo "======================================"

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到 Python"
    exit 1
fi

# 使用自动重启下载器运行下载任务
python auto_restart_download.py \
    --repo-id "byeonghwikim/abp_dataset" \
    --local-dir "data/json_feat_2.1.0" \
    --folder-pattern "train/look_at_obj_in_light*/**" \
    --repo-type "dataset" \
    --max-retries 10 \
    --retry-delay 60 \
    --exponential-backoff

echo ""
echo "下载任务完成！"

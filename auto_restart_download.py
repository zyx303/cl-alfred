#!/usr/bin/env python3
"""
通用自动重启下载脚本
支持各种下载任务的自动重试机制
"""

import os
import sys
import time
import signal
import logging
import argparse
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__)))

from utils.download_feat import download_hf_patterns

class AutoRestartDownloader:
    def __init__(self, max_retries=5, retry_delay=30, exponential_backoff=False):
        """
        自动重启下载器
        
        Args:
            max_retries: 最大重试次数
            retry_delay: 初始重试延迟时间（秒）
            exponential_backoff: 是否使用指数退避策略
        """
        self.max_retries = max_retries
        self.initial_retry_delay = retry_delay
        self.exponential_backoff = exponential_backoff
        self.retry_count = 0
        self.start_time = datetime.now()
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self.should_stop = False
        
        # 设置日志
        log_file = f"download_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _signal_handler(self, signum, frame):
        """处理中断信号"""
        self.logger.info(f"接收到信号 {signum}，正在停止下载...")
        self.should_stop = True
        
    def _calculate_retry_delay(self):
        """计算重试延迟时间"""
        if self.exponential_backoff:
            # 指数退避: 30, 60, 120, 240, 480秒
            return self.initial_retry_delay * (2 ** (self.retry_count - 1))
        else:
            # 固定延迟
            return self.initial_retry_delay
            
    def _check_existing_files(self, local_dir, folder_pattern):
        """检查已存在的文件"""
        if not os.path.exists(local_dir):
            return False
            
        # 简单检查目录是否非空
        try:
            files = list(Path(local_dir).rglob("*"))
            if files:
                self.logger.info(f"发现已存在 {len(files)} 个文件在 {local_dir}")
                return True
        except Exception:
            pass
            
        return False
        
    def download_huggingface(self, repo_id, local_dir, folder_pattern, repo_type="dataset", token=None, force_download=False):
        """
        HuggingFace 下载任务
        """
        self.logger.info("=" * 60)
        self.logger.info("HuggingFace 自动重启下载器")
        self.logger.info("=" * 60)
        self.logger.info(f"最大重试次数: {self.max_retries}")
        self.logger.info(f"重试延迟策略: {'指数退避' if self.exponential_backoff else '固定延迟'}")
        self.logger.info(f"仓库ID: {repo_id}")
        self.logger.info(f"本地目录: {local_dir}")
        self.logger.info(f"文件夹模式: {folder_pattern}")
        self.logger.info(f"仓库类型: {repo_type}")
        self.logger.info("=" * 60)
        
        # 检查是否已存在文件
        if not force_download and self._check_existing_files(local_dir, folder_pattern):
            response = input("发现已存在文件，是否继续下载？(y/N): ")
            if response.lower() not in ['y', 'yes']:
                self.logger.info("用户取消下载")
                return True
                
        while self.retry_count <= self.max_retries and not self.should_stop:
            try:
                self.logger.info(f"开始下载 (第 {self.retry_count + 1} 次尝试)")
                
                download_hf_patterns(
                    repo_id=repo_id,
                    local_dir=local_dir,
                    folder_pattern=folder_pattern,
                    repo_type=repo_type,
                    token=token
                )
                
                self.logger.info("✅ 下载成功完成！")
                total_time = datetime.now() - self.start_time
                self.logger.info(f"总耗时: {total_time}")
                return True
                
            except KeyboardInterrupt:
                self.logger.info("下载被用户中断")
                return False
                
            except Exception as e:
                self.retry_count += 1
                self.logger.error(f"❌ 下载失败: {str(e)}")
                
                if self.retry_count <= self.max_retries and not self.should_stop:
                    retry_delay = self._calculate_retry_delay()
                    self.logger.info(f"将在 {retry_delay} 秒后重试 (第 {self.retry_count}/{self.max_retries} 次重试)")
                    
                    # 等待，支持中断
                    for i in range(retry_delay):
                        if self.should_stop:
                            break
                        time.sleep(1)
                        if (i + 1) % 10 == 0:  # 每10秒显示一次倒计时
                            remaining = retry_delay - (i + 1)
                            self.logger.info(f"还有 {remaining} 秒后重试...")
                else:
                    self.logger.error(f"已达到最大重试次数 ({self.max_retries})，停止重试")
                    break
        
        total_time = datetime.now() - self.start_time
        self.logger.info(f"下载结束，总耗时: {total_time}")
        self.logger.info(f"总重试次数: {self.retry_count}")
        return False

def main():
    parser = argparse.ArgumentParser(description="自动重启下载脚本")
    
    # HuggingFace 参数
    parser.add_argument('--repo-id', required=True, help='HuggingFace 仓库ID')
    parser.add_argument('--local-dir', required=True, help='本地保存目录')
    parser.add_argument('--folder-pattern', required=True, help='文件夹匹配模式')
    parser.add_argument('--repo-type', default='dataset', choices=['model', 'dataset', 'space'], help='仓库类型')
    parser.add_argument('--token', help='HuggingFace 访问令牌')
    
    # 重试参数
    parser.add_argument('--max-retries', type=int, default=5, help='最大重试次数')
    parser.add_argument('--retry-delay', type=int, default=30, help='重试延迟时间（秒）')
    parser.add_argument('--exponential-backoff', action='store_true', help='使用指数退避策略')
    parser.add_argument('--force', action='store_true', help='强制下载，即使文件已存在')
    
    args = parser.parse_args()
    
    # 创建下载器
    downloader = AutoRestartDownloader(
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        exponential_backoff=args.exponential_backoff
    )
    
    # 执行下载
    success = downloader.download_huggingface(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        folder_pattern=args.folder_pattern,
        repo_type=args.repo_type,
        token=args.token,
        force_download=args.force
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

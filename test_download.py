import time
import logging
import sys
from datetime import datetime
from utils.download_feat import download_hf_patterns

class AutoRestartDownloader:
    def __init__(self, max_retries=5, retry_delay=30):
        """
        自动重启下载器
        
        Args:
            max_retries: 最大重试次数
            retry_delay: 重试前等待时间（秒）
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_count = 0
        
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
        
    def download_with_retry(self, repo_id, local_dir, folder_pattern, repo_type="dataset", token=None):
        """
        带重试机制的下载函数
        """
        self.logger.info(f"开始自动重启下载，最大重试次数: {self.max_retries}")
        self.logger.info(f"下载参数:")
        self.logger.info(f"  - 仓库ID: {repo_id}")
        self.logger.info(f"  - 本地目录: {local_dir}")
        self.logger.info(f"  - 文件夹模式: {folder_pattern}")
        self.logger.info(f"  - 仓库类型: {repo_type}")
        
        while self.retry_count <= self.max_retries:
            try:
                self.logger.info(f"开始下载 (第 {self.retry_count + 1} 次尝试)")
                
                download_hf_patterns(
                    repo_id=repo_id,
                    local_dir=local_dir,
                    folder_pattern=folder_pattern,
                    repo_type=repo_type,
                    token=token
                )
                
                self.logger.info("下载成功完成！")
                return True
                
            except KeyboardInterrupt:
                self.logger.info("下载被用户中断")
                return False
                
            except Exception as e:
                self.retry_count += 1
                self.logger.error(f"下载失败: {str(e)}")
                
                if self.retry_count <= self.max_retries:
                    self.logger.info(f"将在 {self.retry_delay} 秒后重试 (第 {self.retry_count}/{self.max_retries} 次重试)")
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"已达到最大重试次数 ({self.max_retries})，停止重试")
                    return False
        
        return False

if __name__ == "__main__":
    import os
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

    # 创建自动重启下载器
    downloader = AutoRestartDownloader(
        max_retries=10,      # 最大重试5次
        retry_delay=5      # 重试前等待30秒
    )
    
    # 执行下载
    success = downloader.download_with_retry(
        repo_id="byeonghwikim/abp_images",
        local_dir='data/json_feat_2.1.0',
        folder_pattern='train/**/feat_conv_panoramic.pt',
        repo_type="dataset"
    )
    
    if success:
        print("\n✅ 下载任务成功完成！")
        sys.exit(0)
    else:
        print("\n❌ 下载任务失败！")
        sys.exit(1)
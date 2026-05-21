# 远程服务器开发使用说明

通过 sshfs 把远程服务器的项目目录挂载到本地，实现"像编辑本地文件一样编辑远程文件"。

## 一、环境信息

| 项 | 值 |
|---|---|
| 远程主机 | `163.61.150.40` |
| SSH 端口 | `60024` |
| 用户 | `root` |
| 远程项目路径 | `/root/DriveVLA-W0` |
| **本地挂载点** | **`/home/wang/remote/DriveVLA-W0`** |
| 远程硬件 | 8× NVIDIA A100-SXM4-80GB，877G 磁盘 |
| 主机名 | `xt-p-A100-013` |

已配置：
- 本地 `~/.ssh/id_ed25519` 公钥已加入远程 `/root/.ssh/authorized_keys`，**SSH 免密登录**
- 本地已安装 `sshfs`（apt 包 `sshfs 3.7.3`）

## 二、日常使用

### 1. 编辑远程文件
挂载后，远程文件就是 `/home/wang/remote/DriveVLA-W0/` 下的本地路径，任何文本编辑器、IDE、`Read`/`Edit`/`Write` 工具都能直接操作。所有改动会**实时写回远程服务器**。

示例：
```bash
# 用 VSCode 打开远程项目
code /home/wang/remote/DriveVLA-W0

# 命令行查看 / 编辑
ls /home/wang/remote/DriveVLA-W0/utils/
vim /home/wang/remote/DriveVLA-W0/utils/train_grpo_stage2b.py
```

### 2. 在远程执行命令
SSH 已配置免密，直接：
```bash
ssh -p 60024 root@163.61.150.40 "cd /root/DriveVLA-W0 && nvidia-smi"
```

或登入交互式 shell：
```bash
ssh -p 60024 root@163.61.150.40
```

### 3. 检查挂载状态
```bash
mount | grep DriveVLA
# 期望输出: root@163.61.150.40:/root/DriveVLA-W0 on /home/wang/remote/DriveVLA-W0 type fuse.sshfs (...)
```

## 三、挂载 / 卸载

### 重新挂载（断网或重启后）
```bash
sshfs root@163.61.150.40:/root/DriveVLA-W0 /home/wang/remote/DriveVLA-W0 \
  -p 60024 \
  -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3
```

参数说明：
- `reconnect`：网络抖动后自动重连
- `ServerAliveInterval=15`：每 15 秒发心跳包
- `ServerAliveCountMax=3`：3 次无响应才判定断连

### 卸载
```bash
fusermount3 -u /home/wang/remote/DriveVLA-W0
```

如果提示 `device is busy`，先关掉所有访问该目录的进程（包括打开它的编辑器、终端），再卸载。强制卸载（最后手段）：
```bash
fusermount3 -uz /home/wang/remote/DriveVLA-W0
```

## 四、本地 ↔ 远程 同步

挂载下 sshfs 是**实时**的（每次读写都走 SSH），但有些场景需要批量同步。

### 把本地代码批量推到远程
```bash
rsync -avz --progress -e "ssh -p 60024" \
  --exclude='logs/' \
  --exclude='pretrained_models/' \
  --exclude='cache/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache/' \
  --exclude='.venv/' \
  /home/wang/Project/DriveVLA-W0/ \
  root@163.61.150.40:/root/DriveVLA-W0/
```

### 把远程结果拉回本地
```bash
# 拉日志
rsync -avz --progress -e "ssh -p 60024" \
  root@163.61.150.40:/root/DriveVLA-W0/logs/ \
  /home/wang/Project/DriveVLA-W0/logs/

# 拉某个 checkpoint
rsync -avz --progress -e "ssh -p 60024" \
  root@163.61.150.40:/root/DriveVLA-W0/logs/run-xxx/best.pt \
  /home/wang/Downloads/
```

## 五、未同步内容（远程需手动准备）

为节省带宽，以下大目录**没有**同步到远程，需要在远程单独准备：

| 目录 | 本地大小 | 处理方式 |
|---|---|---|
| `pretrained_models/` | 82G | 远程重新下载（HuggingFace / 内部源） |
| `logs/` | 147G | 训练时自动生成，无需同步 |
| `cache/` | <1M | 训练时自动生成 |

还需要在远程：
- 装 conda 环境（参考本地 `environment.yaml`、`requirements.txt`、`Training.md`）
- 配置 git 用户信息（如果需要在远程提交）
- 准备数据集（远程 `data/` 路径）

## 六、性能与注意事项

### 性能
- **小文件读写**：基本无感（编辑代码、跑 grep）
- **大文件读写**：明显比本地慢，走网络带宽
- **建议**：训练数据、模型权重放远程；不要在挂载目录里跑大量 IO 密集任务，直接 SSH 到远程跑

### 注意事项
1. **不要在挂载目录直接运行训练脚本**。Python 解释器、数据加载、checkpoint 写盘全部会走 sshfs，慢得不可接受。训练用 SSH 在远程跑。
2. **git 操作**：在挂载目录里做 `git status` / `git log` 是查询远程仓库；想看本地仓库去 `/home/wang/Project/DriveVLA-W0`。两个仓库各自独立。
3. **断网后**：挂载会"卡死"，命令悬挂。先 `fusermount3 -uz` 强制卸载，再重新挂载。
4. **VSCode 打开挂载目录可能卡**：因为 VSCode 会扫描所有文件建索引。建议在 `.vscode/settings.json` 排除大目录，或者用 VSCode Remote-SSH 扩展直接连远程（体验更好，不需要 sshfs）。

## 七、备选方案：VSCode Remote-SSH

如果你主要用 VSCode 写代码，**Remote-SSH 扩展**比 sshfs 更顺手：
- 安装 VSCode 扩展 `Remote - SSH`
- 配置 `~/.ssh/config`：
  ```
  Host drivevla-remote
      HostName 163.61.150.40
      Port 60024
      User root
      IdentityFile ~/.ssh/id_ed25519
  ```
- VSCode 中 `F1 → Remote-SSH: Connect to Host → drivevla-remote`
- 远程跑 VSCode Server，本地只是窗口；终端、调试器、语言服务全在远程，没有 sshfs 的 IO 延迟

两种方式可以并存：日常在 VSCode Remote-SSH 里开发，命令行批量操作用 sshfs。

## 八、常用快捷命令汇总

```bash
# 挂载
sshfs root@163.61.150.40:/root/DriveVLA-W0 /home/wang/remote/DriveVLA-W0 -p 60024 -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3

# 卸载
fusermount3 -u /home/wang/remote/DriveVLA-W0

# 检查挂载
mount | grep DriveVLA

# 登录远程
ssh -p 60024 root@163.61.150.40

# 远程跑命令
ssh -p 60024 root@163.61.150.40 "<command>"

# 推代码到远程
rsync -avz --progress -e "ssh -p 60024" --exclude='logs/' --exclude='pretrained_models/' --exclude='cache/' --exclude='__pycache__/' /home/wang/Project/DriveVLA-W0/ root@163.61.150.40:/root/DriveVLA-W0/

# 拉日志回本地
rsync -avz --progress -e "ssh -p 60024" root@163.61.150.40:/root/DriveVLA-W0/logs/ /home/wang/Project/DriveVLA-W0/logs/
```

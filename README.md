# 局部可信度引导的边缘条件图像生成

《生成式人工智能导论》课程项目。基于 SD1.5 和 T2I-Adapter 实际微调原适配器，研究局部条件出错时，输入预掩码、多尺度门控和配对训练各自能提供什么改善。当前输入使用从照片提取的 Canny 边缘作为草图代理，真实自由手绘尚未验证。

[匿名实验报告：7 页正文＋参考文献](docs/submission/anonymous_report.pdf) · [课堂汇报：11 页](docs/submission/presentation.pdf) · [可编辑报告](docs/REPORT.md) · [逐种子结果](docs/results/course_results.csv) · [讲解稿](docs/PRESENTATION_NOTES.md) · [答辩问答](docs/DEFENSE_GUIDE.md)

## 主论文和代码来源

- Chong Mou 等，T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability for Text-to-Image Diffusion Models，AAAI 2024 主轨，38(5): 4296–4304，DOI [10.1609/aaai.v38i5.28226](https://doi.org/10.1609/aaai.v38i5.28226)。[AAAI 主轨清单](https://aaai.org/wp-content/uploads/2024/02/AAAI-24_Main_2024-02-01.pdf) · [CCF 官方目录](https://www.ccf.org.cn/Academic_Evaluation/AI/)。
- 作者参考实现为 [SD 分支 train.py 固定版本](https://github.com/TencentARC/T2I-Adapter/blob/16bba674b472121d5a86e3ed6b935f91d516bc74/train.py)，提交 16bba674b472121d5a86e3ed6b935f91d516bc74。
- 实际运行实现为本项目基于 Diffusers 0.30.3 编写的训练循环，不是作者完整训练流水线的直接执行。使用作者发布的 sketch SD1.5 v2 权重，共同初始化所有配置，实际更新约 77M Adapter 参数。完整方法额外训练 56,516 个门控参数。
- SD1.5、Adapter 和 CLIP 的精确仓库 revision、大小和 SHA-256 保存在 docs/resources/models。实际数据划分保存在 docs/resources/splits，图像校验值在 docs/resources/images_sha256.json。下载脚本默认使用这些快照，避免以后模型仓库或 Python 随机实现变化造成漂移。

## 方法和复现边界

完整方法首先计算 R 与边缘条件的逐点乘积；在 Adapter 的四个尺度上，用可靠性图和条件特征预测有界的空间权重；训练时对同一目标照片构造两份局部干预，要求可信区域内的噪声预测一致。VAE、文本编码器和 U-Net 冻结，U-Net 前向保留对输入残差的梯度。

空间控制强度已经有 SmartControl 等先行工作。本项目不声称首次提出空间门控，而是检验显式局部可靠性输入下的增量设计。输入预掩码是必须比较的强基线。当前 R 来自已知合成干预区域，是控制系数，不是模型预测或经过校准的正确概率。

预处理为 RGB 中心裁剪、Lanczos 缩放和 Canny 100/200 白色边缘，与作者完整 sketch 预处理不同。有限数据微调不能直接对标原论文完整训练的 FID。基线和改进均实际训练；生成样例和指标由本组检查点产生。

## 目录

- aigc：数据干预、可训练适配器与门控、训练、生成及离线评价。
- scripts：资源恢复、冻结编码缓存、实验队列、统计分析和图表。
- configs：本地验收、预试验及四卡正式配置。
- docs/resources：轻量资源快照、图像校验值与上游版本记录。
- docs/results 和 docs/figures：已完成的正式结果、配置、日志、逐图诊断与展示图。
- assets、data、runs、offline、work、outputs：资源与运行文件，不进入 Git。

## 环境

校内实测为 Linux x86_64，Python 3.10.14，PyTorch 2.2.2，CUDA runtime 12.1，RTX 4090 24GB。项目 env 单独安装固定依赖，基础镜像不变。离线包已包含 29 个匹配 wheels；在项目根目录执行：

```bash
bash scripts/bootstrap_linux.sh
```

联网机器可使用独立环境安装。4090 的对应版本示例：

```bash
python3.10 -m venv env
source env/bin/activate
python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-linux-cp310.txt
```

本机 5070 验收对应 Python 3.12、PyTorch 2.11.0+cu128。不同硬件和软件构建不保证逐位一致。Linux 离线包要求与已验收基础 Torch 匹配。

## 模型 数据和缓存

在可联网的准备机器执行。模型通道默认 hf-mirror，使用固定 revision 和逐文件校验；镜像的版本与哈希不等同于作者数字签名。

```bash
python scripts/prepare_assets.py models
python scripts/prepare_assets.py data
python scripts/prepare_metrics.py
python scripts/cache_latents.py --resolution 512
```

固定划分为 8,000 训练、400 开发及 1,200 测试候选图，image ID 无交叠。正式首轮只使用冻结测试清单的前 400 张，剩余 800 张保留。图像 caption 固定为按标注 ID 排序的第一条。缓存保存 VAE 后验 mean/logvar、文本特征和边缘图，训练仍采样后验噪声。

只做工程短跑时，可先对 smoke 清单的 16 张图生成 512 缓存：

```bash
python scripts/cache_latents.py --manifest data/manifests/smoke.jsonl --resolution 512
python -m aigc.train --config configs/local_512_acceptance.json --variant full --seed 42 --output runs/check_full
python -m aigc.train --config configs/local_512_acceptance.json --variant full --seed 42 --output runs/check_full --max-steps 4 --resume
```

384 分辨率的 configs/smoke.json 需要另行创建对应分辨率缓存。工程短跑不是方法效果的验证。

## 正式训练和消融

| variant | 实际训练配置 |
|---|---|
| baseline | 干净条件，更新原 Adapter；两路相同条件控制处理次数 |
| augmented | 相同局部干预增强，更新原 Adapter |
| premask | 增强加输入可信度预掩码 |
| full | 预掩码、多尺度学习门控与配对损失 |
| no_pair | 完整方法去除配对损失 |
| fixed_gate | 完整方法的学习门控替换为固定可靠性权重 |

正式配置 configs/course_4gpu.json：512、全局 batch 8、micro batch 4、1,500 优化步、AdamW 学习率 1e-5、权重衰减 0.01、梯度裁剪 1、BF16、配对权重 0.1、每 500 步保存。全局 batch 以独立目标照片数计，每图有两路条件。配对项按图像归一化后取批量均值。

基线与完整方法单卡命令：

```bash
CUDA_VISIBLE_DEVICES=0 env/bin/python -m aigc.train --config configs/course_4gpu.json --variant baseline --seed 42 --output runs/course/baseline_s42
CUDA_VISIBLE_DEVICES=1 env/bin/python -m aigc.train --config configs/course_4gpu.json --variant full --seed 42 --output runs/course/full_s42
```

四卡队列默认四主配置各两个训练种子 42、43，再跑两个种子 42 消融。每卡一个实验，累积时限跨阶段共用 ledger：

```bash
env/bin/python scripts/run_queue.py --gpus 0,1,2,3 --config configs/course_4gpu.json --phase train --max-hours 6
```

同一实验两卡 DDP 的已验收入口：

```bash
CUDA_VISIBLE_DEVICES=0,1 env/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 --module aigc.train --config configs/local_512_acceptance.json --variant full --seed 42 --output runs/check_ddp
```

训练保存 last.pt、config.json、train.jsonl、status.json。检查点含 Adapter、门控、AdamW 状态、精度 scaler、seed、全局 step 与配置。恢复须显式加 --resume；已有输出不自动覆盖。数据采样和干预由全局样例位置控制。

## 生成 评价与统计

```bash
env/bin/python scripts/run_queue.py --gpus 0,1,2,3 --config configs/course_4gpu.json --phase generate --sample-count 400 --max-hours 6
env/bin/python scripts/run_queue.py --gpus 0,1,2,3 --config configs/course_4gpu.json --phase evaluate --sample-count 400 --max-hours 6
env/bin/python scripts/summarize.py --output outputs/course_results
env/bin/python scripts/analyze_course.py --output outputs/course_analysis.json
```

逐模型入口：

```bash
env/bin/python -m aigc.generate --checkpoint runs/course/full_s42/last.pt --manifest data/manifests/test.jsonl --output runs/course/full_s42/generated --count 400 --steps 25
env/bin/python -m aigc.evaluate --samples runs/course/full_s42/generated --distribution
```

每图生成 clean、0、1 三个视图，相同文本、可信图和初始噪声。DDIM 25 步、CFG 7.5、生成基种子 2026 加 image ID。生成中断后加 --resume，配置或检查点改变时使用新的输出目录。

主指标为可信区域边缘 F1 和新增错误笔画的相对跟随提升量。辅助指标为 CLIP cosine、Clean-FID 和 KID。分布指标只取每图第一个干预视图，明确标注 FID-400；KID 原始值不自动乘 1000。安全过滤输出保留，另做各方法共同排除过滤图像的敏感性分析。主表给出两个训练种子均值及标准差；图像 bootstrap 不等同于训练种子不确定性。

新增假笔画样例不足时不机械报告错误跟随率；擦除样例按类别诊断。两个消融只与完整方法种子 42 直接配对，避免单种子与双种子平均混比。所有对照与失败结果均保留。

## 已验收的资源和工程链路

512/BF16 单卡真实训练、保存恢复、两卡项目 DDP、参数更新、离线 CLIP 和 Inception 均已通过。20 步四路实测的稳态中位步时与峰值显存为：micro1 1.81 秒和 4.73 GiB，micro2 0.96 秒和 6.32 GiB，micro4 0.63 秒和 9.69 GiB，micro8 0.62 秒和 16.14 GiB。正式训练实际峰值可能略高，日志逐步记录。

正式 10 次训练平均每次 16.35 分钟，最大已分配显存 9.97 GiB；四卡并行训练阶段约 49.9 分钟，训练、生成与评测完整流程约 108.7 分钟，不包括先期下载、部署、预试验与备份。队列 ledger 统计子进程活跃期间墙钟时间，不能代替平台计费时间或关闭资源分配。

## 实际结果与结论

400 张测试图，两个干预视图，主配置各两个训练种子。下表为均值 ± 训练种子标准差；错误跟随越低越好，F1 与 CLIP 越高越好。全部逐种子指标、FID-400/KID 与配对统计见 docs/results/course_analysis.json。

| 方法 | 可信边缘 F1 | 错误跟随提升量 | CLIP cosine |
|---|---:|---:|---:|
| 原适配器微调 | 0.7081 ± 0.0006 | 0.4522 ± 0.0065 | 0.3082 ± 0.0005 |
| 同增强基线 | 0.6822 ± 0.0058 | 0.2669 ± 0.0117 | 0.3085 ± 0.0001 |
| 输入预掩码 | 0.6850 ± 0.0011 | 0.0739 ± 0.0033 | 0.3087 ± 0.0004 |
| 完整方法 | 0.6857 ± 0.0014 | 0.0325 ± 0.0009 | 0.3098 ± 0.0002 |

完整方法减预掩码的错误跟随差值为 -0.04137，图像簇 bootstrap 95% 区间 [-0.05358, -0.02827]。该指标只覆盖 260 张至少一个干预视图产生新增笔画的图像，不是错误概率。F1 差值 +0.00069 的区间跨零；相对原基线 F1 下降约 0.02245。输入预掩码已经解释大部分收益，错误抑制存在布局保持代价。

种子 42 的完整方法、no_pair 与 fixed_gate 的错误跟随分别为 0.03312、0.03174、0.03187；消融没有证明两个学习组件各自提供稳定收益。共同排除任意配置被安全过滤的 15 张图后，完整方法相对预掩码的错误抑制仍存在，但 CLIP 微小优势的区间跨零。所有负面结果与标记保留。

训练与指标源码的 SHA-256 与读取测试结果前冻结的协议逐项一致，见 docs/results/core_source_audit.json。后续图表、资源恢复脚本和文档有整理更新，不冒充正式训练前已运行的版本。

## 报告与结果复核

匿名报告正文 7 页，另 1 页参考文献；课堂汇报为 11 页横向 PDF。源码包保留报告文本、图表与生成脚本。实际贡献说明、课堂讲解稿和问答分别见 docs/CONTRIBUTIONS.md、docs/PRESENTATION_NOTES.md、docs/DEFENSE_GUIDE.md；课程条目对应见 docs/COURSE_COMPLIANCE.md。

无需 GPU、模型和 COCO 原图即可用保存的轻量记录重新计算表格与图表：

```bash
python scripts/analyze_course.py --runs docs/results/runs --output work/recomputed_analysis.json
python scripts/make_figures.py --runs docs/results/runs --output docs/figures
```

PDF 复生成另需 requirements-report.txt 的绘图和排版依赖。Windows 默认使用系统宋体、黑体；其他系统须用 --font 指定可嵌入中文字体。字体文件不随源码分发。报告 JSON 和图表可直接编辑：

```bash
python scripts/build_report.py --output outputs/匿名实验报告.pdf --slides outputs/课堂汇报.pdf
python scripts/export_report.py --output docs/REPORT.md
```

完整检查点约 9.24 GB，另有完整生成 TAR 约 5.19 GB，保存在本地实验备份中。源码仓库只收录轻量记录及检查点校验清单，不包含第三方权重、原始数据、SSH 密钥或登录配置。仓库地址为 https://github.com/qingqi567703/aigc-trusted-adapter 。本仓库的匿名报告与汇报 PDF 位于 docs/submission；完整本地备份未随源码公开分发。

## 相关工作与研究结论边界

[SmartControl](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06412.pdf) 已研究空间控制强度与粗糙条件冲突；[SketchingReality](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ea34f9c3acb3de6fa29582ff469bcd6b-Abstract-Conference.html) 明确区分边缘图与真实自由手绘。本项目的外部 R、合成干预和配对训练需要用简单强基线核查，不能包装成全新空间控制范式。

当前证据限定于一个生成骨干、合成条件错误与有限训练种子。即使指标正向，也不直接证明真实手绘泛化、置信度估计能力或顶会级独立贡献。

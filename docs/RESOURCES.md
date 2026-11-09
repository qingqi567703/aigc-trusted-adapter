# 论文、模型、数据与评测资源

以下为实际采用或用于界定贡献的公开来源。模型与数据已下载并参与运行；逐文件校验值在 resources。链接公开可检索，不等同于任何时间和网络下都能访问。

| 类型 | 来源 | 本项目用途与核查 |
|---|---|---|
| 主论文 | [AAAI DOI](https://doi.org/10.1609/aaai.v38i5.28226)、[正式主轨清单](https://aaai.org/wp-content/uploads/2024/02/AAAI-24_Main_2024-02-01.pdf) | AAAI 2024，38(5):4296–4304。发行元数据已保存；期刊网站部分请求触发机器人验证，以主轨清单和出版元数据交叉核实。 |
| CCF 资格 | [CCF 人工智能目录](https://www.ccf.org.cn/Academic_Evaluation/AI/) | AAAI 位于 A 类；主论文不是仅 arXiv 或 Workshop。 |
| 原代码 | [作者 SD 分支固定提交](https://github.com/TencentARC/T2I-Adapter/tree/16bba674b472121d5a86e3ed6b935f91d516bc74) | 实际核对 train.py 中预训练加载及训练入口。本项目实际运行的是自写 Diffusers 循环。 |
| 生成骨干 | [Stable Diffusion 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/451f4fe16113bff5a5d2269ed5ad43b0592e9a14) | 固定 revision，使用本地 fp16 生成权重；训练冻结骨干。 |
| 适配器 | [作者 sketch SD1.5 v2](https://huggingface.co/TencentARC/t2iadapter_sketch_sd15v2/tree/a73a564e35b0956deffd164e1ee463f967c3f952) | 下载到本机并真实加载；各对照共同初始化、各自微调。 |
| 文本图像指标 | [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32/tree/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268) | 仅用于离线评价，不是本项目新增可训练模型。 |
| 图像与文字 | [COCO 2017 数据](https://cocodataset.org/#download) | 8,000 训练、400 开发、1,200 测试候选；测试只使用前 400。固定清单、caption 和 JPEG SHA-256 已保存。 |
| 分布指标 | [Clean-FID](https://github.com/GaParmar/clean-fid)、[NVlabs Inception 权重](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt) | clean-fid 0.1.35；离线 Inception 提取和 2048 维形状已验收，清洗缩放与特征路径已核对。 |
| 直接近邻 | [SmartControl，ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06412.pdf) | 空间控制尺度与冲突处理已有工作，本项目不能声称首次提出空间门控。未实际训练该方法作直接对照。 |
| 自由手绘边界 | [SketchingReality，ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ea34f9c3acb3de6fa29582ff469bcd6b-Abstract-Conference.html) | 已核对正式页面和方法描述；边缘代理不代表真实自由手绘。未作本项目直接实验对照。 |

`scripts/prepare_assets.py` 默认恢复固定 revision 与相同划分，检查每个模型文件和 JPEG。`scripts/prepare_metrics.py` 准备固定 Inception 权重。模型、数据和第三方依赖沿用各自的使用条款，源码包只包含获取说明与资源清单，不把第三方权重和 COCO 原图上传 GitHub。

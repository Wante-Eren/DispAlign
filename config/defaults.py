from yacs.config import CfgNode as CN

_C = CN()

# ===================== 模型配置 =====================
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.DEVICE_ID = '0'
_C.MODEL.NAME = 'IDEA'
_C.MODEL.PRETRAIN_PATH_T = '/root/cloud-ssd/IDEA-main/pretrained/pytorch_model.bin'
_C.MODEL.NECK = 'bnneck'
_C.MODEL.IF_WITH_CENTER = 'no'
_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
_C.MODEL.ID_LOSS_WEIGHT = 0.25
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
_C.MODEL.DIST_TRAIN = False
_C.MODEL.IF_LABELSMOOTH = 'on'

# 借鉴自MambaPro的配置
_C.MODEL.PROMPT = False
_C.MODEL.ADAPTER = False
_C.MODEL.FROZEN = False

# Transformer设置
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.TRANSFORMER_TYPE = 'vit_base_patch16_224'
_C.MODEL.STRIDE_SIZE = [16, 16]
_C.MODEL.FORWARD = 'old'
_C.MODEL.DIRECT = 1  # 0=分模态分类，1=拼接特征分类

# SIE参数
_C.MODEL.SIE_COE = 1.0
_C.MODEL.SIE_CAMERA = True
_C.MODEL.SIE_VIEW = False

# IDEA特有的配置（适配新CDA模块）
_C.MODEL.PREFIX = True
_C.MODEL.TEXT_PROMPT = 2
_C.MODEL.INVERSE = True
_C.MODEL.DA = True  # 控制是否启用新CDA模块（保留开关，兼容新模块）

# ===================== 新增：新CDA（Wave语义路由器）核心配置 =====================
_C.MODEL.CDA = CN()
# 基础CDA结构参数
_C.MODEL.CDA.N_HEADS = 1  # 注意力头数
_C.MODEL.CDA.N_HEAD_CHANNELS = 512  # 每个注意力头的通道数
_C.MODEL.CDA.N_GROUPS = 1  # 分组数（需满足 N_HEAD_CHANNELS*N_HEADS % N_GROUPS == 0）
_C.MODEL.CDA.STRIDE = 2  # 采样步长
_C.MODEL.CDA.WINDOW_SIZE = (5, 5)  # CDA窗口大小
_C.MODEL.CDA.Q_SIZE = (16, 8)  # 查询特征尺寸
_C.MODEL.CDA.STRIDE_BLOCK = (4, 4)  # 分块步长
_C.MODEL.CDA.KSIZE = 5  # 卷积核大小（Offset生成用）
_C.MODEL.CDA.OFFSET_RANGE_FACTOR = 5  # 偏移量范围因子（替代旧OFF_FAC，功能一致）

# WaveSemanticRouter 核心参数（阻尼波动方程）
_C.MODEL.CDA.WAVE_NUM_GROUPS = 4  # 波动算子分组数（需满足 N_HEAD_CHANNELS*N_HEADS/N_GROUPS % WAVE_NUM_GROUPS == 0）
_C.MODEL.CDA.WAVE_ALPHA = 0.1  # 单模态WPO的阻尼系数（控制衰减速度）
_C.MODEL.CDA.WAVE_TIME_STEP = 1.0  # 单模态WPO的时间步（控制波动演化程度）
# 全局二次共振WPO参数（独立配置，避免影响单模态）
_C.MODEL.CDA.WAVE_GLOBAL_ALPHA = 0.05  # 全局WPO阻尼系数（建议比单模态小，保证平滑）
_C.MODEL.CDA.WAVE_GLOBAL_TIME_STEP = 0.8  # 全局WPO时间步（建议比单模态短）

# 旧CDA参数（已废弃，保留注释便于回退）
# _C.MODEL.DA_SHARE = False  # 旧CDA的跨模态共享偏移量参数（新模块通过WaveRouter实现语义共享）
# _C.MODEL.OFF_FAC = 5.0     # 旧CDA的偏移因子（新模块对应 CDA.OFFSET_RANGE_FACTOR）

# ===================== 输入配置 =====================
_C.INPUT = CN()
_C.INPUT.SIZE_TRAIN = [256, 128]
_C.INPUT.SIZE_TEST = [256, 128]
_C.INPUT.PROB = 0.5
_C.INPUT.RE_PROB = 0.5
_C.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
_C.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]
_C.INPUT.PADDING = 10

# ===================== 数据集配置 =====================
_C.DATASETS = CN()
_C.DATASETS.NAMES = ('RGBNT201')
_C.DATASETS.ROOT_DIR = './data'

# ===================== 数据加载器配置 =====================
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 14
_C.DATALOADER.SAMPLER = 'softmax_triplet'
_C.DATALOADER.NUM_INSTANCE = 8

# ===================== 求解器配置 =====================
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "Adam"
_C.SOLVER.MAX_EPOCHS = 50
_C.SOLVER.BASE_LR = 0.00035
_C.SOLVER.LARGE_FC_LR = False
_C.SOLVER.BIAS_LR_FACTOR = 2
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.MARGIN = 0.3
_C.SOLVER.CLUSTER_MARGIN = 0.3
_C.SOLVER.CENTER_LR = 0.5
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005
_C.SOLVER.RANGE_K = 2
_C.SOLVER.RANGE_MARGIN = 0.3
_C.SOLVER.RANGE_ALPHA = 0
_C.SOLVER.RANGE_BETA = 1
_C.SOLVER.RANGE_LOSS_WEIGHT = 1
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0001
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (40, 70)
_C.SOLVER.WARMUP_FACTOR = 0.01
_C.SOLVER.WARMUP_ITERS = 10
_C.SOLVER.WARMUP_METHOD = "linear"
_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30
_C.SOLVER.SEED = 1111
_C.MODEL.NO_MARGIN = True
_C.SOLVER.CHECKPOINT_PERIOD = 50
_C.SOLVER.LOG_PERIOD = 10
_C.SOLVER.EVAL_PERIOD = 1
_C.SOLVER.IMS_PER_BATCH = 64

# ===================== 测试配置 =====================
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 128
_C.TEST.RE_RANKING = 'no'
_C.TEST.WEIGHT = ""
_C.TEST.NECK_FEAT = 'before'
_C.TEST.FEAT_NORM = 'yes'
_C.TEST.MISS = 'None'

# ===================== 杂项选项 =====================
_C.OUTPUT_DIR = "./IDEA"

# 新增：快速开关（调试用）
_C.MODEL.CDA.ENABLE_GLOBAL_WPO = True  # 是否启用全局二次共振WPO
_C.MODEL.CDA.VISUALIZE_WEIGHTS = True  # 是否可视化模态语义权重（训练时输出TensorBoard）
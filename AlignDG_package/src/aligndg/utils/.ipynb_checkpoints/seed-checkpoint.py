import os, random
import numpy as np

def set_global_seed(seed: int, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        # 严格确定性（部分算子不支持会报warning/异常）
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except ImportError:
        pass  # 没装 torch 就略过

def seed_worker(worker_id: int):
    import torch
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

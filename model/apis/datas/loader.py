import torch
from torch.utils.data import DataLoader as DL
from torch.utils.data import Subset, random_split, default_collate

from ..tools.config import Configuration
from ..datas.dataset_base import Dataset_Base

# 加载数据集
def DatasetLoader(cfg: Configuration, dataset_origin: Dataset_Base, batch_size=None):
    if dataset_origin.mode == 'normal':
        num_all = len(dataset_origin)
        num_train = int(num_all * cfg.train['proportion'])
        num_test = num_all - num_train
        indices_train = torch.linspace(0, num_train - 1, num_train)
        indices_test = torch.linspace(num_train, num_all - 1, num_test)

        dataset_train = Subset(dataset_origin, indices_train.long())
        dataset_test = Subset(dataset_origin, indices_test.long())
        dataset_valid, _ = random_split(dataset_test, [cfg.valid['proportion'], 1 - cfg.valid['proportion']])

        # 加载数据集形成batch
        bs_train = batch_size or cfg.train['batch'] // cfg.device_num
        bs_valid = batch_size or cfg.valid['batch'] // cfg.device_num
        bs_test = batch_size or cfg.test['batch'] // cfg.device_num
        dataset_train = DL(dataset_train, bs_train, shuffle=True, pin_memory=False, num_workers=0, drop_last=False)
        dataset_valid = DL(dataset_valid, bs_valid, shuffle=False, pin_memory=False, num_workers=0)
        dataset_test = DL(dataset_test, bs_test, shuffle=False, pin_memory=False, num_workers=0)
        dataset_free = None
    else:
        dataset_train = None
        dataset_valid = None
        dataset_test = None
        dataset_free = DL(dataset_origin, 1, shuffle=False, pin_memory=True, num_workers=0, collate_fn=custom_collate)

    dataset = {
        'train': dataset_train,
        'valid': dataset_valid,
        'test': dataset_test,
        'free': dataset_free
    }

    return dataset

def custom_collate(batch):
    assert len(batch) == 1  # 只为处理单个 batch
    out = {}
    batch_ = batch[0]
    for key, value in batch_.items():
        if value is None:
            out[key] = value  # 保留 None
        else:
            try:
                out[key] = default_collate([value])
            except Exception:
                out[key] = [value]  # 如果不能 stack，就保持 list 格式
    return out
import numpy as np
import torch
from scipy import stats

# 对字典内的数据进行规整处理
class SumUpHandle:
    def __init__(self):
        self.datas = None
    
    def __call__(self, datas):
        if self.datas is None:
            self.datas = {}
            for k, v in datas.items():
                self.datas[k] = [self._v_item(v)]
        else:
            # 合并key
            k1 = set(self.datas.keys())
            k2 = set(datas.keys())
            ks = k1 | k2
            l = len(self)

            for k in ks:
                v = datas.get(k, None)
                if k in self.datas:
                    self.datas[k].append(self._v_item(v))
                else:
                    # 第一次见，补全长度，之前都是None
                    self.datas[k] = [None] * l + [self._v_item(v)]
        assert self.check_datas(), "Data length not consistent"
    
    def __len__(self):
        self.check_datas()
        if self.datas is None:
            return 0
        else:
            return len(self.datas[list(self.datas.keys())[0]])

    def __getitem__(self, k):
        if self.datas is None:
            raise ValueError("No data found")
        else:
            return self.datas.get(k, None)
    
    def __setitem__(self, k, v):
        if self.datas is None:
            self.datas = {k: [self._v_item(v)]}
        else:
            if k in self.datas:
                self.datas[k] = self._v_item(v)
            else:
                raise ValueError("Key not found in previous data")

    def _v_item(self, v):
        if isinstance(v, np.ndarray):
            return v.item()
        elif isinstance(v, torch.Tensor):
            return v.item()
        elif isinstance(v, bool):
            return int(v)
        elif isinstance(v, float) or isinstance(v, int):
            return v
        elif v is None:
            return v
        else:
            raise ValueError("Unsupported data type")

    def check_datas(self):
        return self.datas is not None and len({len(v) for v in self.datas.values()}) == 1

    def get_sumup_result(self, method='all', has_postfix=True):
        assert method in ['all', 'sum', 'mean', 'trim_mean'], "Unsupported method"

        # 检查所有key对应的value长度一样
        assert self.check_datas(), "Data length not consistent"

        result = {}
        for k, v in self.datas.items():
            v_ = [v__ for v__ in v if v__ is not None]  # 去除None值
            v_np = np.array(v_)  # 转换为numpy数组
            if method == 'all' or method =='mean':
                _v = np.mean(v_np)
                if has_postfix:
                    result[f'{k}_mean'] = _v
                else:
                    result[k] = _v
            if method == 'all' or method == 'trim_mean':
                _v = stats.trim_mean(v_np, 0.1)
                if has_postfix:
                    result[f'{k}_trim_mean'] = _v
                else:
                    result[k] = _v
            if method == 'all' or method =='sum':
                _v = np.sum(v_np)
                if has_postfix:
                    result[f'{k}_sum'] = _v
                else:
                    result[k] = _v
        return result

    def reset(self):
        self.datas = None

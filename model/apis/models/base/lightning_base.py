import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import lightning as L
import matplotlib.pyplot as plt
import queue
import pprint
from pytorch_lightning.utilities import rank_zero_only

from ...tools.config import Configuration
from ...criterions.loss import Loss
from ...criterions.metric import MetricBatch
from ...tools.sumup_handle import SumUpHandle
from ...tools.util import init_logger

class LightningBase(L.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        # 定义模型的各个部分
        self.cfg: Configuration = kwargs.get('cfg', None)
        self.dtype_torch = self.cfg.dtype_model_torch
        self.logger_self = None

        # 记录每次valid的结果，在最后绘制效果最差的结果
        self.valid_results = queue.PriorityQueue()

        # 定义评估函数
        self.criterions = {
            'loss': Loss(self.cfg),  # 损失函数
            'metric_batch': MetricBatch(self.cfg),  # 评估函数，每个batch都计算一次
        }
        self.sumup_handle = {
            'train_loss': SumUpHandle(),
            'train_metric': SumUpHandle(),
            'train_grad': SumUpHandle(),
            'validtest_loss': SumUpHandle(),
            'validtest_metric': SumUpHandle()
        }

        # 保存一些我关注的量
        self.save_hyperparameters()
        self.params_log = {
            'is_rsu': self.cfg.is_rsu,
            'is_obu': not self.cfg.is_rsu,
            'lr': self.cfg.train['lr'],
            'lr_pre_trained': self.cfg.train['lr_pre_trained'],
            'batch_size': self.cfg.train['batch'],
            'path_planning_method': self.cfg.path_planning_head_params['method']
        }

    # 模型推理
    def forward(self, batch):
        raise NotImplementedError

    # 整理输出
    def collate_outps_tgt(self, batch):
        raise NotImplementedError
    
    # 可视化每个batch
    def show_batch_result(self, outputs, name, global_step, save_to_disk=False, show_on_tensorboard=True):
        raise NotImplementedError

    def run_base(self, batch: dict, log_flag=True):
        # 模型推理
        oups = self.forward(**batch)

        # 整理输出
        oups_tgt = self.collate_outps_tgt(batch)

        # 进行评估
        losses = self.criterions['loss'](oups, oups_tgt)
        metrics_batch = self.criterions['metric_batch'](oups, oups_tgt)

        # 记录
        if log_flag:
            self.set_log(losses, batch_size=len(next(iter(batch))))
        if self.training:
            self.sumup_handle['train_loss'](losses)
            self.sumup_handle['train_metric'](metrics_batch)
        else:
            self.sumup_handle['validtest_loss'](losses)
            self.sumup_handle['validtest_metric'](metrics_batch)

        return oups, oups_tgt, losses, metrics_batch

    def set_log(self, loss_dict, batch_size):
        prefix = 'train' if self.training else 'validtest'
        for k, v in loss_dict.items():
            self.log(f'{prefix}/{k}', v, batch_size=batch_size, prog_bar=False, logger=False, on_epoch=True, on_step=False, sync_dist=True)

    def get_optimizer(self, params, lrs, names, weight_decays):
        assert len(params) == len(lrs) == len(names) == len(weight_decays)
        return optim.Adam([
            {'params': param, 'lr': lr, 'name': name, 'weight_decay': weight_decay} for param, lr, name, weight_decay in zip(params, lrs, names, weight_decays)
        ])
    
    def get_scheduler(self, optimizer, T_max, eta_min=1e-6):
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    def configure_optimizers(self):
        optimizer = self.get_optimizer([self.parameters()], [self.cfg.train['lr']], ['main'], [1e-4])
        scheduler = self.get_scheduler(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }

    def training_step(self, batch, batch_idx):
        _, _, losses, _ = self.run_base(batch)
        return losses['all']

    def validation_step(self, batch, batch_idx):
        return self.run_base(batch)

    def test_step(self, batch, batch_idx):
        return self.run_base(batch, log_flag=False)

    def check_nan_inf(self, data):
        if isinstance(data, torch.Tensor):
            assert not torch.isnan(data).any(), f"NAN!"
            assert not torch.isinf(data).any(), f"INF!"
        elif isinstance(data, dict):
            for k in data:
                self.check_nan_inf(data[k])
        elif isinstance(data, (list, tuple)):
            for d in data:
                self.check_nan_inf(d)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def on_fit_start(self):
        self.logger.log_hyperparams(
            params=self.params_log
        )
        self.logger_self = init_logger(self.cfg, f'{self.__class__.__name__}-v{self.logger.version}')
        self.logger_self.info(f'cfg: \n{str(self.cfg)}')
        formatted_params = self.format_dict_for_print(self.params_log, keys_per_line=2)
        self.logger_self.info(f'Params_log:\n{formatted_params}')
        return super().on_fit_start()

    def get_single_from_oups(self, oups, idx):
        oups_ = {}
        for k in oups:
            if isinstance(oups[k], list):
                oups_[k] = [o[idx].unsqueeze(0) for o in oups[k]]
            elif isinstance(oups[k], torch.Tensor):
                oups_[k] = oups[k][idx].unsqueeze(0)
            else:
                raise TypeError(f"Unsupported data type: {type(oups[k])}")
        return oups_

    def get_single_from_oups_tgt(self, oups_tgt, idx):
        oups_tgt_ = {}
        for k in oups_tgt:
            if isinstance(oups_tgt[k], torch.Tensor):
                oups_tgt_[k] = oups_tgt[k][idx].unsqueeze(0)
            else:
                raise TypeError(f"Unsupported data type: {type(oups_tgt[k])}")
        return oups_tgt_

    def reset_sumup_handle(self):
        for k in self.sumup_handle:
            self.sumup_handle[k].reset()

    def on_validation_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx=0):
        _, _, losses, _ = outputs
        self.valid_results.put((losses['all'].item(), self.valid_results.qsize(), outputs))

    def get_abs_grad(self):
        max_abs_grad = 0.0
        for name, param in self.named_parameters():
            if param.grad is not None:
                max_abs_grad = max(max_abs_grad, param.grad.abs().max().item())
        return max_abs_grad

    def on_before_optimizer_step(self, optimizer):
        max_grad = self.get_abs_grad()
        self.sumup_handle['train_grad']({'max_grad': max_grad})

    def on_validation_end(self):
        # queue转换成list
        valid_results = []
        while not self.valid_results.empty():
            valid_results.append(self.valid_results.get()[-1])

        # 找出最好和最差的部分。为了提高效率，我就粗略地认为，最好的和最坏的分别在最好和最坏的batch内
        best_outputs_batch = valid_results[0]
        worest_outputs_batch = valid_results[-1]

        # 遍历batch_size，找出最差中的最差，最好中的最好
        def get_best_worst(outputs):
            oups, oups_tgt, _, _ = outputs
            _queue = queue.PriorityQueue()
            for i in range(len(oups['segmentation'])):
                oups_ = self.get_single_from_oups(oups, i)
                oups_tgt_ = self.get_single_from_oups_tgt(oups_tgt, i)
                losses_ = self.criterions['loss'](oups_, oups_tgt_)
                metrics_ = self.criterions['metric_batch'](oups_, oups_tgt_)
                _queue.put((losses_['all'], _queue.qsize(), [oups_, oups_tgt_, losses_, metrics_]))
            _list = []
            while not _queue.empty():
                _list.append(_queue.get()[-1])
            return _list[0], _list[-1]
        best_outputs, _ = get_best_worst(best_outputs_batch)
        _, worest_outputs = get_best_worst(worest_outputs_batch)

        # 绘制最差的部分
        self.show_batch_result(best_outputs, 'valid/best', self.global_step, save_to_disk=False, show_on_tensorboard=True)
        self.show_batch_result(worest_outputs, 'valid/worest', self.global_step, save_to_disk=False, show_on_tensorboard=True)
        self.valid_results = queue.PriorityQueue()

        # 记录
        losses_metrics = {
            **self.sumup_handle['validtest_loss'].get_sumup_result(method='mean', has_postfix=False),
            **self.sumup_handle['validtest_metric'].get_sumup_result(method='mean', has_postfix=False)
        }
        for k, v in losses_metrics.items():
            self.logger.experiment.add_scalar(f'valid/{k}', v, self.current_epoch)

    def on_train_epoch_end(self):
        infos = {
            **self.sumup_handle['train_loss'].get_sumup_result(method='mean', has_postfix=False),
            **self.sumup_handle['train_metric'].get_sumup_result(method='mean', has_postfix=False),
            **self.sumup_handle['train_grad'].get_sumup_result(method='mean', has_postfix=False)
        }
        for k, v in infos.items():
            self.logger.experiment.add_scalar(f'train/{k}', v, self.current_epoch)

        # 将infos打印出来
        formatted_infos = self.format_dict_for_print(infos, keys_per_line=2)
        self.logger_self.info(
            f'Epoch: {self.current_epoch} / {self.trainer.max_epochs} complete. \nInfos:\n{formatted_infos}'
        )

        self.reset_sumup_handle()
    
    def format_dict_for_print(self, infos_dict, keys_per_line=2):
        # 计算每个键的最大长度
        max_key_length = max(len(k) for k in infos_dict.keys())
        max_value_length = max(len(str(v)) for v in infos_dict.values())
        
        lines = []
        keys = list(infos_dict.keys())
        num_keys = len(keys)
        
        for i in range(0, num_keys, keys_per_line):
            line_parts = []
            for j in range(i, min(i + keys_per_line, num_keys)):
                key = keys[j]
                value = 'None' if infos_dict[key] is None else infos_dict[key]
                line_parts.append(f"'{key:<{max_key_length}}': {value:<{max_value_length}}")
            lines.append(', '.join(line_parts))
        
        return '\n'.join(lines)


    # def on_test_batch_end(self, outputs, batch, batch_idx):
    #     # 绘制每一个
    #     oups, oups_tgt, _, _ = outputs
    #     for i in range(len(oups['segmentation'])):
    #         oups_ = self.get_single_from_oups(oups, i)
    #         oups_tgt_ = self.get_single_from_oups_tgt(oups_tgt, i)
    #         losses_ = self.criterions['loss'](oups_, oups_tgt_)
    #         metrics_ = self.criterions['metric_batch'](oups_, oups_tgt_)
    #         outputs = [oups_, oups_tgt_, losses_, metrics_]
    #         self.show_batch_result(outputs, 'test', batch_idx * len(oups['segmentation']) + i, save_to_disk=True, show_on_tensorboard=False)

    @rank_zero_only
    def on_test_end(self):
        self.logger.log_hyperparams(
            params=self.params_log, 
            metrics=self.sumup_handle['validtest_metric'].get_sumup_result(method='mean', has_postfix=False),  # 获取规整后的结果
        )
        self.reset_sumup_handle()

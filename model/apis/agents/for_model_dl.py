import os
import torch
import pickle
import queue
import numpy as np
import matplotlib.pyplot as plt
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Timer, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from tqdm import tqdm
from matplotlib.gridspec import GridSpec

from .for_base import AgentBase
from .for_plot_in_use import AgentPlotInUse
from ..tools.config import Configuration
from ..tools.util import extract_data_for_path_planning_from_datas, save_datas_to_disk
from ..models.model_perception import ModelPerception
from ..models.model_pathplanning import ModelPathPlanning
from ..datas.loader import DatasetLoader
from ..datas.dataset_perception import Dataset_Perception
from ..datas.dataset_path_planning import Dataset_PathPlanning
from ..tools.sumup_handle import SumUpHandle

class AgentModelDL(AgentBase):
    def __init__(self, cfg: Configuration,*args, **kwargs):
        super().__init__(cfg)

        self.agent_plot = None
        self.dataset = None
        self.model_lightning = None
        self.current_step = None
        self.ckpt_path = None
        self.trainer = None
        self.steps = ['step_1', 'step_2']
        self.sumup = {
            'losses': SumUpHandle(),
            'metrics': SumUpHandle()
        }
        self.Modelxx_step = {
            'step_1': ModelPerception,
            'step_2': ModelPathPlanning
        }
        self.Datasetxx_setp = {
            'step_1': Dataset_Perception,
            'step_2': Dataset_PathPlanning
        }
        self.flag_train = kwargs.get('flag_train')

        self.init_all()

    def init_all(self):
        self.logger.info(f'cfg: \n{str(self.cfg)}')

        self.init_model()
        self.logger.info('Finish to Initialize Model_Lightning')

        self.init_dataset()
        self.logger.info('Finish to Initialize Dataset')

        self.init_trainer()
        self.logger.info('Finish to Initialize Loggerxx_step')

        self.agent_plot = AgentPlotInUse(self.cfg)
        self.logger.info('Finish to Initialize Plot')
        
        super().init_all()

    def init_model(self):
        # 直接从默认的cfg中获取设定
        ckpt_paths = self.cfg.ckpts
        self.current_step = f'step_{self.cfg.current_step}'  # 当前的step，只对当前的step进行训练和测试
        assert self.current_step in self.steps, f'{self.current_step} should be in {self.steps}'

        if self.flag_train:
            # 训练模式，初始化当前step的模型
            last_step = f'step_{self.cfg.current_step - 1}'
            if last_step in self.steps and ckpt_paths[last_step] is not None:
                # current_step之前存在预训练模型
                ckpt_path = ckpt_paths[last_step]
                self.logger.info(f'Loading parameters from {ckpt_path}')

                # 将last_step对应的ckpt加载进当前模型架构中，并使用默认cfg初始化模型
                current_model = self.Modelxx_step[self.current_step].load_from_checkpoint(
                    ckpt_path,
                    cfg=self.cfg,  # 用默认的cfg进行初始化
                    strict=False
                )

                # 如果预训练学习率为0，即为冻结
                if self.cfg.train['lr_pre_trained'] == 0.0:
                    self.logger.info(f'Freezing parameters of last step. {ckpt_path}')
                    # 将step_last对应的ckpt加载进上一个step所对应的模型架构中，并使用ckpt内自带的cfg
                    last_model = self.Modelxx_step[last_step].load_from_checkpoint(
                        ckpt_path,
                        strict=True
                    )
                    # 冻结部分模型参数
                    for name, param in current_model.named_parameters():
                        if name in last_model.state_dict():
                            param.requires_grad = False
            else:
                self.logger.info('Initializing a new model')
                # current_step之前不存在预训练模型，则初始化一个新的模型
                current_model = self.Modelxx_step[self.current_step](cfg=self.cfg)
            self.model_lightning = current_model
        else:
            self.ckpt_path = ckpt_paths[self.current_step]
            # 测试模式，从默认的cfg中获取ckpt的路径，但加载模型的时候还是用ckpt中自带的cfg
            self.model_lightning = self.Modelxx_step[self.current_step].load_from_checkpoint(
                self.ckpt_path,
                strict=True
            )

    def init_dataset(self):
        # 只加载当前step对应的数据集
        self.logger.info(f'Loading Dataset for {self.current_step}')
        dataset_ = DatasetLoader(
            self.cfg,
            self.Datasetxx_setp[self.current_step](self.cfg)
        )
        self.dataset = dataset_
        self.logger.info(
            f"Size: [{len(dataset_['train']) * self.cfg.train['batch']}, {len(dataset_['valid']) * self.cfg.valid['batch']}, {len(dataset_['test']) * self.cfg.test['batch']}]"
        )

    def init_trainer(self):
        if self.flag_train:
            # 训练模式，要关注多GPU并行训练
            if self.current_step == 'step_1':
                if self.cfg.is_rsu:
                    name = self.current_step + '-rsu'
                else:
                    name = self.current_step + '-obu'
            elif self.current_step == 'step_2':
                name = self.current_step + f"-{self.cfg.path_planning_head_params['method']}"
            else:
                raise ValueError(f'Invalid step: {self.current_step}')
            logger = TensorBoardLogger(
                save_dir=os.path.join(self.cfg.path_logs, 'lightning_logs'), 
                name=name, default_hp_metric=True
            )
            callbacks = []
            callbacks.append(Timer(duration='00:24:00:00', verbose=True))
            if self.current_step == 'step_1':
                if self.cfg.is_rsu:
                    dirpath = os.path.join(self.cfg.path_ckpts, self.current_step, 'rsu')
                else:
                    dirpath = os.path.join(self.cfg.path_ckpts, self.current_step, 'obu')
            elif self.current_step == 'step_2':
                method = self.cfg.path_planning_head_params['method']
                if self.cfg.ckpts['step_1'] is not None:
                    dirpath = os.path.join(self.cfg.path_ckpts, self.current_step, method, f'{self.cfg.train["lr_pre_trained"]:.5f}')
                else:
                    assert self.cfg.train["lr_pre_trained"] is None, f'lr_pre_trained should be None when step_1 ckpt is None'
                    dirpath = os.path.join(self.cfg.path_ckpts, self.current_step, method, 'new')
            else:
                raise ValueError(f'Invalid step: {self.current_step}')
            callbacks.append(ModelCheckpoint(
                dirpath=dirpath, 
                filename='{epoch}',
                monitor='validtest/all', save_last=True, save_top_k=-1, mode='min', verbose=False
            ))
            callbacks.append(LearningRateMonitor(logging_interval='epoch'))
            if self.cfg.train['patience'] is not None:
                callbacks.append(EarlyStopping(
                    monitor='validtest/all', min_delta=self.cfg.train['min_delta'], patience=self.cfg.train['patience'], 
                    verbose=False, mode='min', check_finite=True
                ))
            self.trainer = L.Trainer(
                log_every_n_steps=1, max_epochs=self.cfg.train['epoch'], check_val_every_n_epoch=1,
                accelerator='gpu', 
                devices=self.cfg.device_num,
                strategy=DDPStrategy(find_unused_parameters=True) if self.cfg.device_num > 1 else 'auto',
                default_root_dir=self.cfg.path_ckpts,
                logger=logger,
                callbacks=callbacks,
                # precision=32,
                # gradient_clip_val=1.0,
                # gradient_clip_algorithm='value',
                enable_progress_bar=False  # 在正常有可视化的情况下，设定为 True ，当使用sbatch这种没有可视化的情况，设定为 False
            )
        else:
            # 测试模式，单GPU测试即可
            self.trainer = L.Trainer(
                log_every_n_steps=1, check_val_every_n_epoch=1,
                accelerator='gpu', devices=1, strategy='auto',
                logger=TensorBoardLogger(
                    save_dir=os.path.join(self.cfg.path_ckpts, 'lightning_logs'), 
                    name=self.current_step, default_hp_metric=False
                ),
                default_root_dir=self.cfg.path_ckpts
            )

    def run(self):
        if self.flag_train:
            # 模型训练
            self.trainer.fit(
                model=self.model_lightning,
                train_dataloaders=self.dataset['train'], 
                val_dataloaders=self.dataset['valid'],
            )
            if self.trainer.is_global_zero:
                # 获取最佳模型路径和模型
                # self.ckpt_path = self.trainer.checkpoint_callback.best_model_path
                # 获取最终模型路径和模型
                self.ckpt_path = self.trainer.checkpoint_callback.last_model_path
                self.logger.info(f'Step: {self.current_step}, Best Model: {self.ckpt_path}')
                # # 创建快捷方式
                # shortcut_path = os.path.join(self.cfg.path_ckpts, self.current_step, "last.ckpt")
                # if os.path.exists(shortcut_path):
                #     os.remove(shortcut_path)
                # os.symlink(self.ckpt_path, shortcut_path)
                # self.ckpt_path = shortcut_path
            else:
                return

        # 进行模型测试
        if self.flag_train:
            # 说明刚刚经历过模型训练，需要重新生成trainer
            self.flag_train = False
            self.init_trainer()
        self.trainer.test(
            model=self.model_lightning, 
            dataloaders=self.dataset['test'],
            ckpt_path=self.ckpt_path
        )

    def get_ckpt_path_of_current_step(self):
        return self.ckpt_path

    def get_model_final(self):
        return self.model_lightning[self.steps[-1]]

    # def run_for_visualize(self, dataset_id=0):
    #     # 将模型移动至device并设定为eval
    #     model_lightning = self.get_model_final()
    #     model_lightning.to(self.cfg.device)
    #     model_lightning.eval()

    #     visual_records = {
    #         'batch': [],
    #         'oups':[],
    #         'oups_tgt':[],
    #         'losses': [],
    #         'metrics': []
    #     }

    #     # 从pkl中选取最后一份数据进行可视化
    #     with open(self.cfg.path_dataset, 'rb') as pkl:
    #         datas_origin = pickle.load(pkl)
    #         # 对于所选数据的每一个sequence，提取数据
    #         dataset = None
    #         for datas in tqdm(
    #             datas_origin[dataset_id]['sequence'],
    #             desc='Sequence', unit='frame', 
    #             leave=False
    #         ):
    #             # 将提取的数据记入dataset
    #             dataset = extract_data_for_path_planning_from_datas(datas, self.cfg, dataset)

    #     # 按时序推理
    #     for i in tqdm(
    #         range(len(dataset['stamp'])), 
    #         desc='Inference', 
    #         leave=False
    #     ):
    #         with torch.no_grad():
    #             # 整理成batch
    #             batch = {}
    #             for k, v in dataset.items():
    #                 if isinstance(v[i], np.ndarray):
    #                     batch[k] = torch.from_numpy(v[i]).to(self.cfg.device).unsqueeze(0)
    #                 elif isinstance(v[i], float):
    #                     batch[k] = torch.tensor(v[i]).to(self.cfg.device).unsqueeze(0)
    #                 elif isinstance(v[i], int):
    #                     batch[k] = torch.tensor(v[i]).to(self.cfg.device).unsqueeze(0)
    #                 elif v[i] is None:
    #                     # 当前规划的内容为空
    #                     batch[k] = None
    #                 else:
    #                     raise ValueError(f'Unsupported type: {type(v[i])}')

    #             # 推理
    #             oups, oups_tgt, losses, metrics = model_lightning.run_base(batch.copy(), log_flag=False)

    #             # 记录结果
    #             visual_records['batch'].append(batch)
    #             visual_records['oups'].append(oups)
    #             visual_records['oups_tgt'].append(oups_tgt)
    #             visual_records['losses'].append(losses)
    #             visual_records['metrics'].append(metrics)
    #             self.sumup['losses'](losses)
    #             self.sumup['metrics'](metrics)

    #     # 为结果可视化创建文件夹
    #     path_folder = os.path.join(self.cfg.path_results, 'model', f'{dataset_id}')
    #     os.makedirs(path_folder, exist_ok=True)

    #     # 可视化结果
    #     paths_fig = []
    #     for i, batch, oups, oups_tgt, losses, metrics in tqdm(
    #         zip(range(len(visual_records['oups'])), visual_records['batch'], visual_records['oups'], visual_records['oups_tgt'], visual_records['losses'], visual_records['metrics']),
    #         total=len(visual_records['oups']),
    #         desc='Visualize',
    #         leave=False
    #     ):
    #         # 对数据进行处理
    #         processed_datas = self.agent_plot.process_datas(batch, oups, oups_tgt, losses, metrics)

    #         # 将展示后的结果进行渲染
    #         fig = self.agent_plot.render_datas(*processed_datas)

    #         # 将渲染后的fig保存在本地
    #         _path_fig = os.path.join(path_folder, f'{i}.png')
    #         fig.savefig(_path_fig)
    #         paths_fig.append(_path_fig)

    #     # 将可视化的内容保存成gif
    #     self.agent_plot.save_gif(path_folder, paths_fig)

    # def run_for_visualize_last_x(self, x=10):
    #     path_folder_model_result = os.path.join(self.cfg.path_results, 'model')
    #     paths_folder = [os.path.join(path_folder_model_result, str(i)) for i in range(self.cfg.collect['num_folder_max']-x, self.cfg.collect['num_folder_max'])]
    #     paths_fig = []
    #     for path_folder in paths_folder:
    #         assert os.path.exists(path_folder), f'Folder not exists: {path_folder}'
    #         _paths_fig = [p for p in os.listdir(path_folder) if p.endswith('.png')]
    #         _paths_fig = sorted(_paths_fig, key=lambda x: int(x.split('.')[0]))
    #         paths_fig.extend([os.path.join(path_folder, p) for p in _paths_fig])
    #     self.agent_plot.save_gif(path_folder_model_result, paths_fig)

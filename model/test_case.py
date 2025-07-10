import os
import torch
torch.set_float32_matmul_precision('medium')
import lightning as L
L.seed_everything(2025)
from tqdm import tqdm

from apis.tools.config import Configuration
from apis.agents.for_test_case import AgentTestCase

if __name__ == '__main__':
    path_cfg = '/'.join(os.path.abspath(__file__).split('/')[:-1] + ['configs', 'config.yaml'])
    cfg = Configuration(path_cfg)
    agent_test_case = AgentTestCase(cfg)
    modes = ['E2E']

    # 跑多个测试用例
    tbar = {
        'A*': tqdm(range(cfg.test_case['num']), desc='A*', leave=False),
        'E2E': tqdm(range(cfg.test_case['num']), desc='E2E', leave=False)
    }
    for i in agent_test_case.tbar:
        agent_test_case.reset()
        agent_test_case.record_from_snapshot()

        for mode in modes:
            agent_test_case.restore_to_snapshot()
            agent_test_case.run(mode)
            tbar[mode].set_postfix(agent_test_case.get_sumup_result(mode, method='mean', has_postfix=False))
            tbar[mode].update()

    # 展示结果
    for mode in modes:
        agent_test_case.show_result(mode)
        agent_test_case.plot_result(mode)

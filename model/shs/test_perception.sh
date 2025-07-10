# !/bin/bash
python "3_Codes/1216-E2E/test_model_open_loop.py" --model_type perception --path_ckpt "3_Codes/1216-E2E/ckpts/step_2/distribution/new/last.ckpt" "--device" "cuda:0"
python "3_Codes/1216-E2E/test_model_open_loop.py" --model_type perception --path_ckpt "3_Codes/1216-E2E/ckpts/step_1/obu/last.ckpt" "--device" "cuda:0"
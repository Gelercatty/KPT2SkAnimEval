# Setup

`python=3.11`

`uv sync`

1. 准备好smpl模型
2. 将humanml3d 和 taichi 数据集准备好，在`configs/base.yaml` 中配置好数据集和smpl模型路径


# Smplify3D

# 使用smplify3d 拟合kpt数据形成smpl参数 h3d 数据集
python -m methods.Smplify3D.fit --dir C:\Users\PC\Desktop\workingSpace\KPT2SkAnimEval\dataset\humanML3D --save_folder C:\Users\PC\Desktop\workingSpace\KPT2SkAnimEval\dataset\out\smplify3d_bates --cuda 0

# 使用smplify3d 拟合kpt数据形成smpl参数 taichi 数据集
python -m methods.Smplify3D.fit --dir C:\Users\PC\Desktop\workingSpace\KPT2SkAnimEval\dataset\taichi --save_folder C:\Users\PC\Desktop\workingSpace\KPT2SkAnimEval\dataset\out\smplify3d_taichi_bates --cuda 0


# Hyberik

## 使用hyberik 拟合KPT数据形成smpl参数  h3d数据集
python -m methods.hyberIK.fit --dir C:\Users\PC\Desktop\workingSpace\KPT2SkAnimEval\dataset\humanML3D --save_folder C:\Users\PC\Desktop\workingSpace\KPT2SkAnimEval\dataset\out\hyberik_humanml3d_bates
## 使用hyberik 拟合KPT数据形成smpl参数  taichi数据集
python -m methods.hyberIK.fit --dir C:\Users\PC\Desktop\workingSpace\KPT2SkAnimEval\dataset\taichi --save_folder C:\Users\PC\Desktop\workingSpace\KPT2SkAnimEval\dataset\out\hyberik_taichi_bates


# 将smpl模型参数转换为重建的kpt数据，用于评估, h3d, hyberik
python utools/smpl_param2kpt.py --dir C:\Users\PC\Desktop\workingSpace\KPT2SkAnimEval\dataset\out\hyberik_humanml3d_bates  --out C:\Users\PC\Desktop\workingSpace\KPT2SkAnimEval\dataset\out\hyberik_humanml3d_reKpt --model-dir C:\Users\PC\Desktop\workingSpace\KPT2SkAnimEval\deps\smpl_models --cuda 0 --joints 24 --humanml22

# 将smpl模型参数转换为重建的kpt数据，用于评估  h3d, smplify3d
python utools/smpl_param2kpt.py --dir C:\Users\PC\Desktop\workingSpace\KPT2SkAnimEval\dataset\out\smplify3d_humanml3d_bates\result  --out C:\Users\PC\Desktop\workingSpace\KPT2SkAnimEval\dataset\out\smplify3d_humanml3d_reKpt --model-dir C:\Users\PC\Desktop\workingSpace\KPT2SkAnimEval\deps\smpl_models --cuda 0 --joints 24 --humanml22

# 将smpl模型参数转换为重建的kpt数据，用于评估 taichi, hyberik
python utools/smpl_param2kpt.py --dir C:\Users\PC\Desktop\workingSpace\KPT2SkAnimEval\dataset\out\hyberik_taichi_bates   --out C:\Users\PC\Desktop\workingSpace\KPT2SkAnimEval\dataset\out\hyberik_taichi_reKpt --model-dir C:\Users\PC\Desktop\workingSpace\KPT2SkAnimEval\deps\smpl_models --cuda 0 --joints 24 --humanml22
# 将smpl模型参数转换为重建的kpt数据，用于评估 taichi, smplify3d
python utools/smpl_param2kpt.py --dir C:\Users\PC\Desktop\workingSpace\KPT2SkAnimEval\dataset\out\smplify3d_taichi_bates\result --out C:\Users\PC\Desktop\workingSpace\KPT2SkAnimEval\dataset\out\smplify3d_taichi_reKpt --model-dir C:\Users\PC\Desktop\workingSpace\KPT2SkAnimEval\deps\smpl_models --cuda 0 --joints 24 --humanml22


# Run

```python main.py```  
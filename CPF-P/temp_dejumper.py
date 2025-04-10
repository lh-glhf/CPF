import numpy as np

# 加载.npy文件
true_data = np.load('/home/LAB/lilanhao/workpath/gspc-torch-seq/w.o.dejumper/results/PatchTST_Close_ftMS_sl10_ll0_pl10_dm512_nh8_el2_dl1_df2048_atprob_fc1_ebtimeF_dtTrue_mxTrue_benchmark_0/true.npy')
pred_data = np.load('/home/LAB/lilanhao/workpath/gspc-torch-seq/w.o.dejumper/results/PatchTST_Close_ftMS_sl10_ll0_pl10_dm512_nh8_el2_dl1_df2048_atprob_fc1_ebtimeF_dtTrue_mxTrue_benchmark_0/pred.npy')

# 确保两个数组的形状相同
assert true_data.shape == pred_data.shape, "true_data 和 pred_data 的形状不匹配"

# 计算均方误差（MSE）
mse = np.mean((true_data - pred_data) ** 2)

print(f"均方误差（MSE）: {mse}")
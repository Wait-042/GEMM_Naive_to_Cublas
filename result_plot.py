#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026/9/8 20:29
# @Author :
# @File : result_plot.py
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # 读取基准测试结果（CSV 格式）
    df = pd.read_csv(r'.\gemm_benchmark.txt')
    kernel_order = df['kernel'].unique()

    # kernel_sel里设置了kernel name的话会只统计kernel_sel里面的kernel结果
    # kernel_sel = ['gemm_naive', 'gemm_coalescing', 'gemm_smem', 'gemm_tile1d', 'gemm_tile2d', 'gemm_register',
    #               'gemm_float4', 'gemm_without_bankconflict', 'gemm_double_buffer', 'gemm_async', 'gemm_async_opt', 'cublasSgemm',]
    kernel_sel = []
    # 查看数据前几行，确认列名和格式
    print(df.head())

    # 因为 M=N=K，取 M 列作为矩阵尺寸
    df['size'] = df['M']
    all_M = sorted(df['M'].unique())

    # 创建一个图形
    plt.figure(figsize=(12, 8))

    # 按照原始顺序绘制每个 kernel 的折线
    for kernel_name in kernel_order:
        sub_df = df[df['kernel'] == kernel_name]
        # 筛选当前 kernel 的数据
        if kernel_name == 'transpose':
            continue
        else:
            if kernel_sel and kernel_name in kernel_sel:
                # 绘制折线，X 轴为矩阵尺寸，Y 轴为 GFLOPS
                plt.plot(sub_df['M'], sub_df['gflops'], marker='o', label=kernel_name)
            else:
                plt.plot(sub_df['M'], sub_df['gflops'], marker='o', label=kernel_name)

    # 设置坐标轴标签和标题
    plt.xlabel('Matrix Size (M=N=K)')
    plt.xticks(all_M, rotation=45)
    plt.ylabel('GFLOPS')
    plt.title('GEMM Performance vs Matrix Size')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例放在外侧，避免遮挡
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 保存图像
    # plt.savefig('gemm_performance.png', dpi=300, bbox_inches='tight')
    plt.show(block=True)

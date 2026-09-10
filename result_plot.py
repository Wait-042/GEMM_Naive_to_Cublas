#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026/9/8 20:29
# @Author :
# @File : result_plot.py
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    df = pd.read_csv(r'.\gemm_benchmark.txt')
    kernel_order = df['kernel'].unique()

    kernel_sel = []
    # kernel_sel = ['gemm_naive', 'gemm_coalescing', 'gemm_smem', 'gemm_tile1d', 'gemm_tile2d',
    #               'gemm_register', 'gemm_float4', 'gemm_without_bankconflict', 'gemm_double_buffer',
    #               'gemm_async', 'gemm_async_opt', 'cublasSgemm']

    # 只统计 M >= limit_min_M 的尺寸；设为 0 表示不限制
    limit_min_M = 2048

    print(df.head())

    # 剔除 transpose（已合并进 gemm_async_opt）
    df = df[df['kernel'] != 'transpose'].copy()

    # 应用尺寸下限过滤
    if limit_min_M > 0:
        df = df[df['M'] >= limit_min_M].copy()

    if df.empty:
        raise SystemExit(f"No data with M >= {limit_min_M}. Check limit_min_M or input file.")

    df['size'] = df['M']
    all_M = sorted(df['M'].unique())

    # ------------------------------------------------------------------
    # 计算相对 cuBLAS 的百分比（逐尺寸，再取均值）
    # ------------------------------------------------------------------
    cublas_df = (
        df[df['kernel'] == 'cublasSgemm']
        .set_index(['M', 'N', 'K'])['gflops']
    )
    df['cublas_gflops'] = df.set_index(['M', 'N', 'K']).index.map(cublas_df)
    df['pct_of_cublas'] = df['gflops'] / df['cublas_gflops'] * 100.0

    pct_summary = (
        df.groupby('kernel')['pct_of_cublas']
        .agg(['mean', 'min', 'max', 'count'])
        .sort_values('mean', ascending=False)
    )

    print(f"\n===== Relative to cuBLAS (GFLOPS %), M >= {limit_min_M} =====")
    print(f"{'Kernel':<32}{'Mean %':>10}{'Min %':>10}{'Max %':>10}{'#Sizes':>8}")
    print("-" * 70)
    for kernel_name, row in pct_summary.iterrows():
        if kernel_name == 'cublasSgemm':
            print(f"{kernel_name:<32}{100.0:>10.2f}{100.0:>10.2f}{100.0:>10.2f}{int(row['count']):>8}")
        else:
            print(f"{kernel_name:<32}{row['mean']:>10.2f}{row['min']:>10.2f}{row['max']:>10.2f}{int(row['count']):>8}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 绘图
    # ------------------------------------------------------------------
    plt.figure(figsize=(12, 8))

    for kernel_name in kernel_order:
        if kernel_name == 'transpose':
            continue
        if kernel_sel and kernel_name not in kernel_sel:
            continue

        sub_df = df[df['kernel'] == kernel_name]
        if sub_df.empty:
            continue
        plt.plot(sub_df['M'], sub_df['gflops'], marker='o', label=kernel_name)

    plt.xlabel('Matrix Size (M=N=K)')
    plt.xticks(all_M, rotation=45)
    plt.ylabel('GFLOPS')
    plt.title(f'GEMM Performance vs Matrix Size (M >= {limit_min_M})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # plt.savefig('gemm_performance.png', dpi=300, bbox_inches='tight')
    plt.show(block=True)

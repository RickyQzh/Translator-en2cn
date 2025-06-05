import os
import json
import time
import shutil
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from train import train_model, calculate_warmup_steps
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import queue
import threading
import torch

def init_multiprocessing():
    """初始化多进程设置，解决CUDA多进程问题"""
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # 如果已经设置过，忽略错误
        pass

def create_experiment_dir(base_dir="./ablation"):
    """创建实验目录，使用时间戳命名子目录"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"experiment_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

def get_experiment_name(config, base_name=""):
    """根据配置生成实验名称"""
    # 提取关键参数生成名称
    name_parts = []
    
    if base_name:
        name_parts.append(base_name)
    
    # 添加关键参数名字
    for key in config.get('ablation_keys', []):
        if key in config:
            name_parts.append(f"{key}_{config[key]}")
    
    return "_".join(name_parts)

def run_single_experiment(experiment_data):
    """运行单个实验的函数，用于多进程执行"""
    experiment, experiment_dir = experiment_data
    config = experiment['config']
    name = experiment['name'] 
    description = experiment['description']
    
    # 生成实验名称和路径
    exp_name = get_experiment_name(config, name)
    exp_output_dir = os.path.join(experiment_dir, exp_name)
    
    print(f"[PID {os.getpid()}] 开始实验: {exp_name}")
    
    # 将配置保存到该实验目录
    os.makedirs(exp_output_dir, exist_ok=True)
    
    # 训练模型
    try:
        # 确保在进程中设置CUDA设备
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # 使用第一个GPU
            
        exp_results = train_model(config, output_dir=exp_output_dir, silent=True)
        
        # 收集结果
        result_entry = {
            'experiment': exp_name,
            'description': description,
            **config,
            **exp_results
        }
        
        print(f"[PID {os.getpid()}] 实验 {exp_name} 完成")
        return result_entry
        
    except Exception as e:
        print(f"[PID {os.getpid()}] 实验 {exp_name} 失败: {str(e)}")
        return None
    finally:
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def run_experiments_in_batches(experiments, experiment_dir, max_concurrent=2):
    """批量运行实验，控制并发数量"""
    results = []
    completed_count = 0
    total_experiments = len(experiments)
    
    # 准备实验数据
    experiment_data = [(exp, experiment_dir) for exp in experiments]
    
    print(f"{'='*80}")
    print(f"开始并行执行实验")
    print(f"总实验数: {total_experiments}")
    print(f"最大并发数: {max_concurrent}")
    print(f"{'='*80}")
    
    # 使用ProcessPoolExecutor来管理并发执行
    with ProcessPoolExecutor(max_workers=max_concurrent) as executor:
        # 提交所有任务
        future_to_exp = {
            executor.submit(run_single_experiment, exp_data): exp_data[0]['name'] 
            for exp_data in experiment_data
        }
        
        # 收集结果
        for future in as_completed(future_to_exp):
            exp_name = future_to_exp[future]
            completed_count += 1
            
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                    print(f"✓ [{completed_count}/{total_experiments}] 实验 {exp_name} 成功完成")
                    
                    # 显示实验结果摘要
                    if 'final_test_bleu' in result:
                        print(f"   Test BLEU: {result['final_test_bleu']:.4f}")
                    if 'final_test_loss' in result:
                        print(f"   Test Loss: {result['final_test_loss']:.4f}")
                else:
                    print(f"✗ [{completed_count}/{total_experiments}] 实验 {exp_name} 失败")
            except Exception as exc:
                print(f"✗ [{completed_count}/{total_experiments}] 实验 {exp_name} 产生异常: {exc}")
            
            # 显示进度
            progress = (completed_count / total_experiments) * 100
            print(f"进度: {progress:.1f}% ({completed_count}/{total_experiments})")
            print("-" * 40)
    
    success_rate = len(results) / total_experiments * 100 if total_experiments > 0 else 0
    print(f"实验完成! 成功率: {success_rate:.1f}% ({len(results)}/{total_experiments})")
    
    return results

def run_ablation_study(max_concurrent=2):
    """执行消融实验"""
    # 创建主实验目录
    base_dir = "./ablation"
    os.makedirs(base_dir, exist_ok=True)
    experiment_dir = create_experiment_dir(base_dir)
    
    print(f"创建消融实验，结果将保存在: {experiment_dir}")
    print(f"最大并发实验数: {max_concurrent}")
    
    # 设置基准配置
    base_config = {
        'batch_size': 256,
        'epochs': 30,
        'learning_rate': 1e-3,
        'd_model': 256,
        'num_heads': 8,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'd_ff': 1024,
        'dropout': 0.2,
        'label_smoothing': 0.1,
        'patience': 5,
        'use_positional_encoding': True,
        'use_multihead_attention': True,
    }
    
    # 定义消融实验配置
    ablation_experiments = [
        # 1. 基准模型
        {
            'name': 'baseline',
            'config': base_config.copy(),
            'description': '基准模型，使用默认参数'
        },
        
        # 2. 变化Transformer尺寸
        {
            'name': 'small_model',
            'config': {**base_config, 'd_model': 128, 'num_heads': 4, 'd_ff': 512, 'ablation_keys': ['d_model', 'num_heads', 'd_ff']},
            'description': '较小的模型尺寸'
        },
        {
            'name': 'large_model',
            'config': {**base_config, 'd_model': 512, 'num_heads': 16, 'd_ff': 2048, 'ablation_keys': ['d_model', 'num_heads', 'd_ff']},
            'description': '较大的模型尺寸'
        },
        
        # 3. 变化层数
        {
            'name': 'layers_3',
            'config': {**base_config, 'num_encoder_layers': 3, 'num_decoder_layers': 3, 'ablation_keys': ['num_encoder_layers', 'num_decoder_layers']},
            'description': '减少编码器和解码器层数'
        },
        {
            'name': 'layers_9',
            'config': {**base_config, 'num_encoder_layers': 9, 'num_decoder_layers': 9, 'ablation_keys': ['num_encoder_layers', 'num_decoder_layers']},
            'description': '增加编码器和解码器层数'
        },
        
        # 4. 变化Dropout率
        {
            'name': 'dropout_0.1',
            'config': {**base_config, 'dropout': 0.1, 'ablation_keys': ['dropout']},
            'description': '较小的Dropout率'
        },
        {
            'name': 'dropout_0.3',
            'config': {**base_config, 'dropout': 0.3, 'ablation_keys': ['dropout']},
            'description': '较大的Dropout率'
        },
        
        # 5. 变化学习率
        {
            'name': 'lr_2e-4',
            'config': {**base_config, 'learning_rate': 2e-4, 'ablation_keys': ['learning_rate']},
            'description': '较小的学习率'
        },
        {
            'name': 'lr_5e-3',
            'config': {**base_config, 'learning_rate': 5e-3, 'ablation_keys': ['learning_rate']},
            'description': '较大的学习率'
        },
        
        # 6. 变化标签平滑系数
        {
            'name': 'label_smoothing_0.0',
            'config': {**base_config, 'label_smoothing': 0.0, 'ablation_keys': ['label_smoothing']},
            'description': '无标签平滑'
        },
        {
            'name': 'label_smoothing_0.2',
            'config': {**base_config, 'label_smoothing': 0.2, 'ablation_keys': ['label_smoothing']},
            'description': '较大的标签平滑系数'
        },
        
        # 7. 变化Warmup比例
        {
            'name': 'warmup_ratio_0.02',
            'config': {**base_config, 'warmup_ratio': 0.02, 'ablation_keys': ['warmup_ratio']},
            'description': '较小的预热比例 (2%)'
        },
        {
            'name': 'warmup_ratio_0.2',
            'config': {**base_config, 'warmup_ratio': 0.2, 'ablation_keys': ['warmup_ratio']},
            'description': '较大的预热比例 (20%)'
        },
        
        # 8. 消融位置编码（Positional Encoding）
        {
            'name': 'no_positional_encoding',
            'config': {**base_config, 'use_positional_encoding': False, 'ablation_keys': ['use_positional_encoding']},
            'description': '不使用位置编码'
        },
        
        # 9. 消融多头注意力（使用单头注意力）
        {
            'name': 'no_multihead_attention',
            'config': {**base_config, 'use_multihead_attention': False, 'ablation_keys': ['use_multihead_attention']},
            'description': '使用单头注意力替代多头注意力'
        },
        
        # 10. 变化注意力头数
        {
            'name': 'attention_heads_4',
            'config': {**base_config, 'num_heads': 4, 'ablation_keys': ['num_heads']},
            'description': '减少注意力头数 (4)'
        },
        {
            'name': 'attention_heads_16',
            'config': {**base_config, 'num_heads': 16, 'ablation_keys': ['num_heads']},
            'description': '增加注意力头数 (16)'
        },
        
        # 11. 不同优化器对比
        {
            'name': 'optimizer_adagrad',
            'config': {**base_config, 'optimizer': 'adagrad', 'ablation_keys': ['optimizer']},
            'description': '使用Adagrad优化器'
        },
        {
            'name': 'optimizer_rmsprop',
            'config': {**base_config, 'optimizer': 'rmsprop', 'ablation_keys': ['optimizer']},
            'description': '使用RMSprop优化器'
        },
        
        # 12. 不同激活函数对比
        {
            'name': 'activation_gelu',
            'config': {**base_config, 'activation': 'gelu', 'ablation_keys': ['activation']},
            'description': '使用GELU激活函数'
        },
        {
            'name': 'activation_swish',
            'config': {**base_config, 'activation': 'swish', 'ablation_keys': ['activation']},
            'description': '使用Swish激活函数'
        },
        
        # 13. 不同隐藏维度对比
        {
            'name': 'd_model_128',
            'config': {**base_config, 'd_model': 128, 'ablation_keys': ['d_model']},
            'description': '较小的隐藏维度 (128)'
        },
        
        # 14. 位置编码方式对比
        {
            'name': 'position_learned',
            'config': {**base_config, 'position_encoding_type': 'learned', 'ablation_keys': ['position_encoding_type']},
            'description': '使用学习型位置编码'
        },
        {
            'name': 'position_relative',
            'config': {**base_config, 'position_encoding_type': 'relative', 'ablation_keys': ['position_encoding_type']},
            'description': '使用相对位置编码'
        },
        
        # 15. 层归一化位置对比
        {
            'name': 'norm_pre',
            'config': {**base_config, 'norm_position': 'pre', 'ablation_keys': ['norm_position']},
            'description': '前置层归一化'
        },
        {
            'name': 'no_layer_norm',
            'config': {**base_config, 'use_layer_norm': False, 'ablation_keys': ['use_layer_norm']},
            'description': '移除层归一化'
        }
    ]
    
    print(f"\n总共 {len(ablation_experiments)} 个实验将并行执行")
    
    # 使用批量并行执行实验
    results = run_experiments_in_batches(ablation_experiments, experiment_dir, max_concurrent)
    
    # 保存所有实验结果到CSV文件
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(experiment_dir, 'all_results.csv'), index=False)
        
        # 生成结果比较图
        plot_ablation_results(results_df, experiment_dir)
        
        print(f"\n所有消融实验完成! 成功完成 {len(results)}/{len(ablation_experiments)} 个实验")
        print(f"结果已保存到: {experiment_dir}")
    else:
        print("\n警告: 没有成功完成的实验!")
    
    return results_df if results else pd.DataFrame()

def plot_ablation_results(results_df, output_dir):
    """绘制消融实验结果比较图"""
    try:
        # 检查输入DataFrame是否为空
        if results_df.empty:
            print("警告: 结果DataFrame为空，无法生成比较图")
            return
            
        # 检查必要的列是否存在
        required_columns = ['experiment', 'best_val_bleu', 'final_test_bleu', 'final_test_loss']
        missing_columns = [col for col in required_columns if col not in results_df.columns]
        if missing_columns:
            print(f"警告: 结果DataFrame缺少必要的列: {missing_columns}")
            # 如果缺少实验名列，则无法继续
            if 'experiment' in missing_columns:
                print("错误: 缺少experiment列，无法绘制图表")
                return
        
        # 设置更好的样式
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 根据实验名称排序
        results_df = results_df.sort_values('experiment')
        
        # 创建一对子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # 绘制验证BLEU比较图（如果存在相应列）
        if 'best_val_bleu' in results_df.columns:
            bars1 = ax1.bar(results_df['experiment'], results_df['best_val_bleu'], color='skyblue', width=0.6)
            ax1.set_xlabel('Experiment', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Validation BLEU Score', fontsize=12, fontweight='bold')
            ax1.set_title('Ablation Study: Validation BLEU Comparison', fontsize=14, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45, labelsize=10)
            ax1.tick_params(axis='y', labelsize=10)
            ax1.grid(axis='y', linestyle='--', alpha=0.7)
        else:
            ax1.text(0.5, 0.5, 'No validation BLEU data available', 
                    ha='center', va='center', fontsize=12, color='gray')
            ax1.set_title('Missing Data', fontsize=14, fontweight='bold')
        
        # 绘制测试损失比较图（如果存在相应列）
        if 'final_test_loss' in results_df.columns:
            bars2 = ax2.bar(results_df['experiment'], results_df['final_test_loss'], color='salmon', width=0.6)
            ax2.set_xlabel('Experiment', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Test Loss', fontsize=12, fontweight='bold')
            ax2.set_title('Ablation Study: Test Loss Comparison', fontsize=14, fontweight='bold')
            ax2.tick_params(axis='x', rotation=45, labelsize=10)
            ax2.tick_params(axis='y', labelsize=10)
            ax2.grid(axis='y', linestyle='--', alpha=0.7)
        else:
            ax2.text(0.5, 0.5, 'No test loss data available', 
                    ha='center', va='center', fontsize=12, color='gray')
            ax2.set_title('Missing Data', fontsize=14, fontweight='bold')
        
        plt.tight_layout(pad=3.0)
        plt.savefig(os.path.join(output_dir, 'ablation_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 额外生成测试BLEU对比图（如果存在相应列）
        if 'final_test_bleu' in results_df.columns:
            plt.figure(figsize=(12, 10))
            bars3 = plt.bar(results_df['experiment'], results_df['final_test_bleu'], color='lightgreen', width=0.6)
            plt.xlabel('Experiment', fontsize=12, fontweight='bold')
            plt.ylabel('Test BLEU Score', fontsize=12, fontweight='bold')
            plt.title('Ablation Study: Test BLEU Comparison', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'ablation_test_bleu.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 生成n-gram BLEU对比图（如果存在相应列）
        ngram_columns = ['final_test_bleu_1', 'final_test_bleu_2', 'final_test_bleu_3', 'final_test_bleu_4']
        available_ngram_columns = [col for col in ngram_columns if col in results_df.columns]
        
        if available_ngram_columns:
            plt.figure(figsize=(16, 10))
            
            x_pos = range(len(results_df))
            width = 0.2
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            
            for i, col in enumerate(available_ngram_columns):
                n_gram = int(col.split('_')[-1])
                plt.bar([p + width * i for p in x_pos], results_df[col], 
                       width=width, label=f'BLEU-{n_gram}', color=colors[i], alpha=0.8)
            
            plt.xlabel('Experiment', fontsize=12, fontweight='bold')
            plt.ylabel('BLEU Score', fontsize=12, fontweight='bold')
            plt.title('Ablation Study: N-gram BLEU Comparison', fontsize=14, fontweight='bold')
            plt.xticks([p + width * 1.5 for p in x_pos], results_df['experiment'], rotation=45, ha='right')
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'ablation_ngram_bleu.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"比较图表已生成在: {output_dir}")
        
        # 创建一个易读的HTML报告
        create_html_report(results_df, output_dir)
    except Exception as e:
        print(f"生成比较图表时出错: {str(e)}")

def create_html_report(results_df, output_dir):
    """创建HTML格式的实验结果报告"""
    try:
        # 检查输入DataFrame是否为空
        if results_df.empty:
            print("警告: 结果DataFrame为空，无法生成HTML报告")
            return
            
        # 检查关键列是否存在
        required_columns = ['experiment', 'best_val_bleu', 'final_test_bleu']
        missing_columns = [col for col in required_columns if col not in results_df.columns]
        if missing_columns:
            print(f"警告: 结果DataFrame缺少必要的列: {missing_columns}，无法生成完整HTML报告")
            
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Transformer Translation Ablation Study Results</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; color: #333; }
                h1, h2, h3 { color: #2c3e50; }
                h1 { border-bottom: 2px solid #3498db; padding-bottom: 10px; }
                h2 { border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; margin-top: 30px; }
                table { border-collapse: collapse; width: 100%; margin-top: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #e1e1e1; }
                th { background-color: #3498db; color: white; font-weight: 500; }
                tr:nth-child(even) { background-color: #f8f9fa; }
                tr:hover { background-color: #f1f1f1; }
                .results-container { margin-top: 30px; }
                .plots-container { display: flex; flex-direction: column; align-items: center; margin-top: 30px; }
                .plot-row { display: flex; justify-content: center; width: 100%; margin-bottom: 30px; }
                .plot-image { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                .plot-caption { text-align: center; margin-top: 10px; color: #7f8c8d; font-style: italic; }
                .highlight { background-color: #d5f5e3 !important; }
                .description { color: #7f8c8d; font-style: italic; }
                .summary { background-color: #f8f9fa; padding: 15px; border-left: 4px solid #3498db; margin: 20px 0; }
                .footer { margin-top: 50px; text-align: center; font-size: 0.9em; color: #7f8c8d; border-top: 1px solid #eee; padding-top: 20px; }
            </style>
        </head>
        <body>
            <h1>Transformer Translation Ablation Study Results</h1>
            <p>Generated on: """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
            
            <div class="summary">
                <p>This report presents the results of ablation studies on the Transformer-based machine translation model.
                The experiments test different configurations and components to determine their impact on translation performance.</p>
            </div>
            
            <div class="results-container">
                <h2>Results Table</h2>
                <table>
                    <tr>
                        <th>Experiment</th>
                        <th>Description</th>
                        <th>Best Val BLEU</th>
                        <th>Final Test BLEU</th>
                        <th>BLEU-1</th>
                        <th>BLEU-2</th>
                        <th>BLEU-3</th>
                        <th>BLEU-4</th>
                        <th>Parameters</th>
                    </tr>
        """
        
        # 获取最佳验证BLEU的实验
        if 'best_val_bleu' in results_df.columns:
            best_val_bleu_exp = results_df.loc[results_df['best_val_bleu'].idxmax()]
        else:
            best_val_bleu_exp = None
        
        # 为每个实验添加一行
        for idx, row in results_df.iterrows():
            # 检查是否是最佳实验
            is_best = False
            if best_val_bleu_exp is not None and 'best_val_bleu' in row:
                is_best = (row['best_val_bleu'] == best_val_bleu_exp['best_val_bleu'])
            row_class = 'highlight' if is_best else ''
            
            # 提取重要参数
            important_params = {}
            param_keys = ['d_model', 'num_heads', 'num_encoder_layers', 'num_decoder_layers', 
                        'dropout', 'learning_rate', 'label_smoothing', 'warmup_ratio',
                        'use_positional_encoding', 'use_multihead_attention']
            
            for k in param_keys:
                if k in row:
                    important_params[k] = row[k]
                else:
                    important_params[k] = '-'
            
            params_str = ", ".join([f"{k}={v}" for k, v in important_params.items() if v != '-'])
            
            # 处理安全访问
            experiment_name = row.get('experiment', f"Experiment_{idx}")
            description = row.get('description', '-')
            val_bleu = row.get('best_val_bleu', 0.0)
            test_bleu = row.get('final_test_bleu', 0.0)
            bleu_1 = row.get('final_test_bleu_1', 0.0)
            bleu_2 = row.get('final_test_bleu_2', 0.0)
            bleu_3 = row.get('final_test_bleu_3', 0.0)
            bleu_4 = row.get('final_test_bleu_4', 0.0)
            
            html_content += f"""
                    <tr class="{row_class}">
                        <td><strong>{experiment_name}</strong></td>
                        <td class="description">{description}</td>
                        <td>{val_bleu:.4f}</td>
                        <td>{test_bleu:.4f}</td>
                        <td>{bleu_1:.4f}</td>
                        <td>{bleu_2:.4f}</td>
                        <td>{bleu_3:.4f}</td>
                        <td>{bleu_4:.4f}</td>
                        <td>{params_str}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="plots-container">
                <h2>Comparison Plots</h2>
        """
                
        # 检查图像文件是否存在
        ablation_comparison_path = os.path.join(output_dir, 'ablation_comparison.png')
        ablation_test_bleu_path = os.path.join(output_dir, 'ablation_test_bleu.png')
        
        if os.path.exists(ablation_comparison_path):
            html_content += """    
                <div class="plot-row">
                    <div style="text-align: center; margin: 0 15px;">
                        <img class="plot-image" src="ablation_comparison.png" alt="Performance Comparison">
                        <p class="plot-caption">Figure 1: Validation BLEU Score and Test Loss Comparison Across Experiments</p>
                    </div>
                </div>
            """
        
        if os.path.exists(ablation_test_bleu_path):
            html_content += """
                <div class="plot-row">
                    <div style="text-align: center; margin: 0 15px;">
                        <img class="plot-image" src="ablation_test_bleu.png" alt="Test BLEU Comparison">
                        <p class="plot-caption">Figure 2: Test BLEU Score Comparison Across Experiments</p>
                    </div>
                </div>
            """
        
        # 检查n-gram BLEU图表是否存在
        ablation_ngram_bleu_path = os.path.join(output_dir, 'ablation_ngram_bleu.png')
        if os.path.exists(ablation_ngram_bleu_path):
            html_content += """
                <div class="plot-row">
                    <div style="text-align: center; margin: 0 15px;">
                        <img class="plot-image" src="ablation_ngram_bleu.png" alt="N-gram BLEU Comparison">
                        <p class="plot-caption">Figure 3: N-gram BLEU Score Comparison (BLEU-1 to BLEU-4) Across Experiments</p>
                    </div>
                </div>
            """
            
        html_content += """
            </div>
            
            <div class="footer">
                <p>Transformer Translation Model Ablation Study &copy; """ + time.strftime("%Y") + """</p>
            </div>
        </body>
        </html>
        """
        
        # 保存HTML文件
        html_file_path = os.path.join(output_dir, 'ablation_report.html')
        with open(html_file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML报告已生成: {html_file_path}")
        
        # 如果存在warmup表格，创建单独的warmup报告
        if os.path.exists(os.path.join(output_dir, 'warmup_comparison.png')):
            create_warmup_html_report(output_dir)
            
    except Exception as e:
        print(f"生成HTML报告时出错: {str(e)}")

def create_warmup_html_report(output_dir):
    """创建单独的warmup实验HTML报告"""
    try:
        # 检查图像文件是否存在
        warmup_comparison_path = os.path.join(output_dir, 'warmup_comparison.png')
        warmup_table_path = os.path.join(output_dir, 'warmup_table.png')
        
        if not os.path.exists(warmup_comparison_path) and not os.path.exists(warmup_table_path):
            print("警告: warmup实验图像文件不存在，无法生成warmup报告")
            return
            
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Warmup Ratio Ablation Study Results</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; color: #333; }
                h1, h2, h3 { color: #2c3e50; }
                h1 { border-bottom: 2px solid #3498db; padding-bottom: 10px; }
                h2 { border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; margin-top: 30px; }
                .plots-container { display: flex; flex-direction: column; align-items: center; margin-top: 30px; }
                .plot-row { display: flex; justify-content: center; width: 100%; margin-bottom: 30px; }
                .plot-image { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                .plot-caption { text-align: center; margin-top: 10px; color: #7f8c8d; font-style: italic; }
                .summary { background-color: #f8f9fa; padding: 15px; border-left: 4px solid #3498db; margin: 20px 0; }
                .footer { margin-top: 50px; text-align: center; font-size: 0.9em; color: #7f8c8d; border-top: 1px solid #eee; padding-top: 20px; }
            </style>
        </head>
        <body>
            <h1>Warmup Ratio Experiment Results</h1>
            <p>Generated on: """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
            
            <div class="summary">
                <p>This report presents the results of ablation studies on different warmup ratio settings for the Transformer-based machine translation model.</p>
            </div>
            
            <div class="plots-container">
                <h2>Warmup Ratio Impact</h2>
        """
        
        if os.path.exists(warmup_comparison_path):
            html_content += """
                <div class="plot-row">
                    <div style="text-align: center; margin: 0 15px;">
                        <img class="plot-image" src="warmup_comparison.png" alt="Warmup Comparison">
                        <p class="plot-caption">Figure 1: Impact of Different Warmup Ratios on Model Performance</p>
                    </div>
                </div>
            """
                
        if os.path.exists(warmup_table_path):
            html_content += """
                <div class="plot-row">
                    <div style="text-align: center; margin: 0 15px;">
                        <img class="plot-image" src="warmup_table.png" alt="Warmup Results Table">
                        <p class="plot-caption">Figure 2: Tabular Results of Warmup Ratio Experiments</p>
                    </div>
                </div>
            """
            
        html_content += """
            </div>
            
            <div class="footer">
                <p>Transformer Translation Model Warmup Ablation Study &copy; """ + time.strftime("%Y") + """</p>
            </div>
        </body>
        </html>
        """
        
        # 保存HTML文件
        html_file_path = os.path.join(output_dir, 'warmup_report.html')
        with open(html_file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print(f"Warmup HTML报告已生成: {html_file_path}")
    except Exception as e:
        print(f"生成warmup HTML报告时出错: {str(e)}")

def run_custom_warmup_experiments(max_concurrent=2):
    """运行不同warmup比例的实验"""
    base_dir = "./ablation"
    os.makedirs(base_dir, exist_ok=True)
    experiment_dir = os.path.join(base_dir, f"warmup_experiment_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"运行不同warmup比例的实验，结果将保存在: {experiment_dir}")
    print(f"最大并发实验数: {max_concurrent}")
    
    # 基本配置
    base_config = {
        'batch_size': 256,
        'epochs': 30, 
        'learning_rate': 1e-3,
        'd_model': 256,
        'num_heads': 8,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'd_ff': 1024,
        'dropout': 0.2,
        'label_smoothing': 0.1,
        'patience': 5
    }
    
    # 定义不同的warmup比例
    warmup_ratios = [0.01, 0.05, 0.1, 0.15, 0.2]
    
    # 创建实验配置
    experiments = []
    for ratio in warmup_ratios:
        experiments.append({
            'name': f'warmup_ratio_{ratio}',
            'config': {**base_config, 'warmup_ratio': ratio},
            'description': f'Warmup比例: {ratio}'
        })
    
    print(f"\n总共 {len(experiments)} 个warmup实验将并行执行")
    
    # 运行实验
    results = run_experiments_in_batches(experiments, experiment_dir, max_concurrent)
    
    # 保存结果
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(experiment_dir, 'warmup_results.csv'), index=False)
        
        # 绘制不同warmup比例的性能对比图
        plot_warmup_comparison(results_df, experiment_dir)
        
        print(f"\n所有warmup实验完成! 成功完成 {len(results)}/{len(experiments)} 个实验")
        print(f"结果已保存到: {experiment_dir}")
    else:
        print("\n警告: 没有成功完成的实验!")
    
    return results_df if results else pd.DataFrame()

def plot_warmup_comparison(results_df, output_dir):
    """绘制不同warmup比例的性能对比图"""
    try:
        # 检查输入DataFrame是否为空
        if results_df.empty:
            print("警告: 结果DataFrame为空，无法生成warmup比较图")
            return
            
        # 检查必要的列是否存在
        required_columns = ['warmup_ratio', 'best_val_bleu', 'final_test_bleu', 'best_val_loss', 'final_test_loss']
        missing_columns = [col for col in required_columns if col not in results_df.columns]
        if missing_columns:
            print(f"警告: 结果DataFrame缺少必要的列: {missing_columns}")
            # 如果缺少warmup_ratio列，则无法继续
            if 'warmup_ratio' in missing_columns:
                print("错误: 缺少warmup_ratio列，无法绘制warmup比较图")
                return
        
        # 设置更好的样式
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 按warmup比例排序
        results_df = results_df.sort_values('warmup_ratio')
        
        # 创建图形和子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
        
        # 绘制BLEU对比图
        has_bleu_data = 'best_val_bleu' in results_df.columns and 'final_test_bleu' in results_df.columns
        if has_bleu_data:
            line1, = ax1.plot(results_df['warmup_ratio'], results_df['best_val_bleu'], 'bo-', label='Validation BLEU', linewidth=2)
            line2, = ax1.plot(results_df['warmup_ratio'], results_df['final_test_bleu'], 'ro-', label='Test BLEU', linewidth=2)
            
            ax1.set_xlabel('Warmup Ratio', fontsize=12, fontweight='bold')
            ax1.set_ylabel('BLEU Score', fontsize=12, fontweight='bold')
            ax1.set_title('Impact of Warmup Ratio on BLEU Score', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # 为x轴设置美观的刻度
            ax1.set_xticks(results_df['warmup_ratio'])
            ax1.tick_params(axis='both', labelsize=10)
            
            # 添加参考线以提高可读性
            for ratio in results_df['warmup_ratio']:
                ax1.axvline(x=ratio, color='gray', linestyle='--', alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No BLEU data available', 
                    ha='center', va='center', fontsize=12, color='gray')
            ax1.set_title('Missing Data', fontsize=14, fontweight='bold')
        
        # 绘制损失对比图
        has_loss_data = 'best_val_loss' in results_df.columns and 'final_test_loss' in results_df.columns
        if has_loss_data:
            line3, = ax2.plot(results_df['warmup_ratio'], results_df['best_val_loss'], 'bo-', label='Best Validation Loss', linewidth=2)
            line4, = ax2.plot(results_df['warmup_ratio'], results_df['final_test_loss'], 'ro-', label='Final Test Loss', linewidth=2)
            
            ax2.set_xlabel('Warmup Ratio', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
            ax2.set_title('Impact of Warmup Ratio on Loss', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # 为x轴设置美观的刻度
            ax2.set_xticks(results_df['warmup_ratio'])
            ax2.tick_params(axis='both', labelsize=10)
            
            # 添加参考线
            for ratio in results_df['warmup_ratio']:
                ax2.axvline(x=ratio, color='gray', linestyle='--', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No loss data available', 
                    ha='center', va='center', fontsize=12, color='gray')
            ax2.set_title('Missing Data', fontsize=14, fontweight='bold')
        
        # 使图表更美观
        plt.tight_layout(pad=3.0)
        plt.savefig(os.path.join(output_dir, 'warmup_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 创建表格形式的可视化
        if not results_df.empty:
            create_warmup_table_plot(results_df, output_dir)
            
        print(f"Warmup比较图表已生成在: {output_dir}")
    except Exception as e:
        print(f"生成warmup比较图表时出错: {str(e)}")

def create_warmup_table_plot(results_df, output_dir):
    """创建一个表格样式的可视化，显示各个warmup比例的结果"""
    try:
        # 检查输入DataFrame是否为空
        if results_df.empty:
            print("警告: 结果DataFrame为空，无法生成warmup表格可视化")
            return
            
        # 检查必要的列是否存在
        required_columns = ['warmup_ratio', 'best_val_bleu', 'final_test_bleu', 'best_val_loss', 'final_test_loss']
        missing_columns = [col for col in required_columns if col not in results_df.columns]
        if missing_columns:
            print(f"警告: 结果DataFrame缺少必要的列: {missing_columns}")
            # 如果缺少warmup_ratio列，则无法继续
            if 'warmup_ratio' in missing_columns:
                print("错误: 缺少warmup_ratio列，无法创建warmup表格")
                return
        
        plt.figure(figsize=(12, 6))
        
        # 关闭坐标轴
        ax = plt.gca()
        ax.axis('off')
        
        # 创建表格数据
        data = []
        columns = ['Warmup Ratio', 'Val BLEU', 'Test BLEU', 'Val Loss', 'Test Loss']
        
        for _, row in results_df.iterrows():
            data_row = []
            # Warmup比例
            data_row.append(f"{row.get('warmup_ratio', 'N/A'):.3f}" if 'warmup_ratio' in row else 'N/A')
            # 验证BLEU
            data_row.append(f"{row.get('best_val_bleu', 0.0):.4f}" if 'best_val_bleu' in row else 'N/A')
            # 测试BLEU
            data_row.append(f"{row.get('final_test_bleu', 0.0):.4f}" if 'final_test_bleu' in row else 'N/A')
            # 验证损失
            data_row.append(f"{row.get('best_val_loss', 0.0):.4f}" if 'best_val_loss' in row else 'N/A')
            # 测试损失
            data_row.append(f"{row.get('final_test_loss', 0.0):.4f}" if 'final_test_loss' in row else 'N/A')
            
            data.append(data_row)
        
        # 创建表格
        table = plt.table(
            cellText=data,
            colLabels=columns,
            cellLoc='center',
            loc='center',
            colColours=['#f2f2f2']*5,
            cellColours=[['#f9f9f9']*5 for _ in range(len(data))],
        )
        
        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # 标题
        plt.title('Warmup Ratio Experiment Results', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'warmup_table.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Warmup表格已生成在: {output_dir}")
    except Exception as e:
        print(f"生成warmup表格时出错: {str(e)}")

def run_advanced_ablation_study(max_concurrent=2):
    """执行高级消融实验：优化器、激活函数、位置编码、层归一化等"""
    # 创建主实验目录
    base_dir = "./ablation"
    os.makedirs(base_dir, exist_ok=True)
    experiment_dir = create_experiment_dir(base_dir)
    
    print(f"运行高级消融实验，结果将保存在: {experiment_dir}")
    print(f"最大并发实验数: {max_concurrent}")
    
    # 设置基准配置
    base_config = {
        'batch_size': 256,
        'epochs': 30,
        'learning_rate': 1e-3,
        'd_model': 256,
        'num_heads': 8,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'd_ff': 1024,
        'dropout': 0.2,
        'label_smoothing': 0.1,
        'patience': 5,
        'use_positional_encoding': True,
        'use_multihead_attention': True,
        'activation': 'relu',
        'position_encoding_type': 'sinusoidal',
        'norm_position': 'post',
        'use_layer_norm': True,
        'optimizer': 'adam'
    }
    
    # 定义高级消融实验配置
    experiments = [
        # 基准模型
        {
            'name': 'baseline',
            'config': base_config.copy(),
            'description': '基准模型'
        },
        
        # 优化器对比
        {
            'name': 'optimizer_adagrad',
            'config': {**base_config, 'optimizer': 'adagrad', 'ablation_keys': ['optimizer']},
            'description': 'Adagrad优化器'
        },
        {
            'name': 'optimizer_rmsprop',
            'config': {**base_config, 'optimizer': 'rmsprop', 'ablation_keys': ['optimizer']},
            'description': 'RMSprop优化器'
        },
        
        # 激活函数对比
        {
            'name': 'activation_gelu',
            'config': {**base_config, 'activation': 'gelu', 'ablation_keys': ['activation']},
            'description': 'GELU激活函数'
        },
        {
            'name': 'activation_swish',
            'config': {**base_config, 'activation': 'swish', 'ablation_keys': ['activation']},
            'description': 'Swish激活函数'
        },
        
        # 隐藏维度对比
        {
            'name': 'd_model_128',
            'config': {**base_config, 'd_model': 128, 'd_ff': 512, 'ablation_keys': ['d_model']},
            'description': '隐藏维度128'
        },
        {
            'name': 'd_model_512',
            'config': {**base_config, 'd_model': 512, 'd_ff': 2048, 'ablation_keys': ['d_model']},
            'description': '隐藏维度512'
        },
        
        # 位置编码方式对比
        {
            'name': 'no_position_encoding',
            'config': {**base_config, 'use_positional_encoding': False, 'ablation_keys': ['use_positional_encoding']},
            'description': '移除位置编码'
        },
        {
            'name': 'position_learned',
            'config': {**base_config, 'position_encoding_type': 'learned', 'ablation_keys': ['position_encoding_type']},
            'description': '学习型位置编码'
        },
        {
            'name': 'position_relative',
            'config': {**base_config, 'position_encoding_type': 'relative', 'ablation_keys': ['position_encoding_type']},
            'description': '相对位置编码'
        },
        
        # 层归一化位置和使用对比
        {
            'name': 'norm_pre',
            'config': {**base_config, 'norm_position': 'pre', 'ablation_keys': ['norm_position']},
            'description': '前置层归一化'
        },
        {
            'name': 'no_layer_norm',
            'config': {**base_config, 'use_layer_norm': False, 'ablation_keys': ['use_layer_norm']},
            'description': '移除层归一化'
        }
    ]
    
    print(f"\n总共 {len(experiments)} 个高级消融实验将并行执行")
    
    # 使用批量并行执行实验
    results = run_experiments_in_batches(experiments, experiment_dir, max_concurrent)
    
    # 保存所有实验结果到CSV文件
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(experiment_dir, 'advanced_ablation_results.csv'), index=False)
        
        # 生成结果比较图
        plot_advanced_ablation_results(results_df, experiment_dir)
        
        print(f"\n所有高级消融实验完成! 成功完成 {len(results)}/{len(experiments)} 个实验")
        print(f"结果已保存到: {experiment_dir}")
    else:
        print("\n警告: 没有成功完成的实验!")
    
    return results_df if results else pd.DataFrame()

def plot_advanced_ablation_results(results_df, output_dir):
    """绘制高级消融实验结果比较图"""
    try:
        if results_df.empty:
            print("警告: 结果DataFrame为空，无法生成比较图")
            return
            
        # 设置更好的样式
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 按实验类型分组
        optimizer_experiments = results_df[results_df['experiment'].str.contains('optimizer|baseline')]
        activation_experiments = results_df[results_df['experiment'].str.contains('activation|baseline')]
        d_model_experiments = results_df[results_df['experiment'].str.contains('d_model|baseline')]
        position_experiments = results_df[results_df['experiment'].str.contains('position|no_position|baseline')]
        norm_experiments = results_df[results_df['experiment'].str.contains('norm|baseline')]
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        axes = axes.flatten()
        
        experiment_groups = [
            (optimizer_experiments, "优化器对比", "Optimizer Comparison"),
            (activation_experiments, "激活函数对比", "Activation Function Comparison"), 
            (d_model_experiments, "隐藏维度对比", "Hidden Dimension Comparison"),
            (position_experiments, "位置编码对比", "Position Encoding Comparison"),
            (norm_experiments, "层归一化对比", "Layer Normalization Comparison")
        ]
        
        for i, (group_df, title_cn, title_en) in enumerate(experiment_groups):
            if not group_df.empty and 'final_test_bleu' in group_df.columns:
                ax = axes[i]
                bars = ax.bar(group_df['experiment'], group_df['final_test_bleu'], 
                             color=plt.cm.Set3(i), alpha=0.8)
                ax.set_xlabel('Experiment', fontsize=10, fontweight='bold')
                ax.set_ylabel('Test BLEU Score', fontsize=10, fontweight='bold')
                ax.set_title(f'{title_cn}\n{title_en}', fontsize=12, fontweight='bold')
                ax.tick_params(axis='x', rotation=45, labelsize=8)
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                
                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 移除空的子图
        for j in range(len(experiment_groups), len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout(pad=3.0)
        plt.savefig(os.path.join(output_dir, 'advanced_ablation_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"高级消融实验比较图已生成在: {output_dir}")
        
    except Exception as e:
        print(f"生成高级消融实验比较图时出错: {str(e)}")

def run_transformer_modules_ablation(max_concurrent=2):
    """运行Transformer模块的消融实验"""
    base_dir = "./ablation"
    os.makedirs(base_dir, exist_ok=True)
    experiment_dir = os.path.join(base_dir, f"transformer_modules_ablation_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"运行Transformer模块消融实验，结果将保存在: {experiment_dir}")
    print(f"最大并发实验数: {max_concurrent}")
    
    # 基本配置
    base_config = {
        'batch_size': 256,
        'epochs': 30, 
        'learning_rate': 1e-3,
        'd_model': 256,
        'num_heads': 8,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'd_ff': 1024,
        'dropout': 0.2,
        'label_smoothing': 0.1,
        'patience': 5,
        'use_positional_encoding': True,
        'use_multihead_attention': True
    }
    
    # 定义实验
    experiments = [
        # 基准模型
        {
            'name': 'baseline',
            'config': base_config.copy(),
            'description': '基准模型 (带位置编码和多头注意力)'
        },
        # 不使用位置编码
        {
            'name': 'no_positional_encoding',
            'config': {**base_config, 'use_positional_encoding': False},
            'description': '不使用位置编码'
        },
        # 单头注意力
        {
            'name': 'single_head_attention',
            'config': {**base_config, 'use_multihead_attention': False},
            'description': '使用单头注意力'
        },
        # 不同注意力头数
        {
            'name': 'heads_4',
            'config': {**base_config, 'num_heads': 4},
            'description': '4个注意力头'
        },
        {
            'name': 'heads_16',
            'config': {**base_config, 'num_heads': 16},
            'description': '16个注意力头'
        },
        # 不同编码器/解码器层数
        {
            'name': 'encoder_layers_3',
            'config': {**base_config, 'num_encoder_layers': 3},
            'description': '3个编码器层'
        },
        {
            'name': 'decoder_layers_3',
            'config': {**base_config, 'num_decoder_layers': 3},
            'description': '3个解码器层'
        }
    ]
    
    print(f"\n总共 {len(experiments)} 个模块消融实验将并行执行")
    
    # 运行实验
    results = run_experiments_in_batches(experiments, experiment_dir, max_concurrent)
    
    # 保存结果
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(experiment_dir, 'transformer_modules_results.csv'), index=False)
        
        # 生成报告
        plot_ablation_results(results_df, experiment_dir)
        create_html_report(results_df, experiment_dir)
        
        print(f"\n所有Transformer模块消融实验完成! 成功完成 {len(results)}/{len(experiments)} 个实验")
        print(f"结果已保存到: {experiment_dir}")
    else:
        print("\n警告: 没有成功完成的实验!")
    
    return results_df if results else pd.DataFrame()

def run_all_experiments(max_concurrent=2):
    """运行所有类型的消融实验"""
    import time
    
    print("=" * 80)
    print("🚀 开始运行所有消融实验")
    print("=" * 80)
    print(f"最大并发数: {max_concurrent}")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    experiments = [
        ("基础消融实验", "Basic Ablation Study", run_ablation_study),
        ("高级消融实验", "Advanced Ablation Study", run_advanced_ablation_study),
        ("Warmup实验", "Warmup Experiments", run_custom_warmup_experiments),
        ("模块消融实验", "Module Ablation Study", run_transformer_modules_ablation)
    ]
    
    total_experiments = len(experiments)
    completed_experiments = 0
    failed_experiments = []
    
    for i, (name_cn, name_en, func) in enumerate(experiments):
        print(f"\n{'=' * 60}")
        print(f"进度: [{i+1}/{total_experiments}] 运行 {name_cn} ({name_en})")
        print(f"{'=' * 60}")
        
        try:
            start_time = time.time()
            result = func(max_concurrent=max_concurrent)
            end_time = time.time()
            
            duration = end_time - start_time
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)
            
            print(f"\n✅ {name_cn} 完成!")
            print(f"   耗时: {hours:02d}:{minutes:02d}:{seconds:02d}")
            completed_experiments += 1
            
        except Exception as e:
            print(f"\n❌ {name_cn} 失败!")
            print(f"   错误: {str(e)}")
            failed_experiments.append(name_cn)
        
        # 在实验之间等待一小段时间
        if i < total_experiments - 1:
            print(f"\n⏳ 等待5秒后继续下一个实验...")
            time.sleep(5)
    
    # 生成最终报告
    print("\n" + "=" * 80)
    print("🎉 所有消融实验完成!")
    print("=" * 80)
    print(f"总实验数: {total_experiments}")
    print(f"成功完成: {completed_experiments}")
    print(f"失败数量: {len(failed_experiments)}")
    
    if failed_experiments:
        print(f"失败的实验: {', '.join(failed_experiments)}")
    
    print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n📁 查看结果:")
    print("   ./ablation/ 目录包含所有实验结果")
    print("   每个实验都有独立的时间戳目录")
    print("=" * 80)

if __name__ == "__main__":
    # 设置multiprocessing启动方法为spawn，解决CUDA多进程问题
    init_multiprocessing()
    
    parser = argparse.ArgumentParser(description='运行Transformer翻译模型的消融实验')
    parser.add_argument('--custom', action='store_true', help='是否运行自定义消融实验')
    parser.add_argument('--warmup', action='store_true', help='运行不同warmup比例的对比实验')
    parser.add_argument('--modules', action='store_true', help='运行Transformer模块（位置编码、注意力机制等）的消融实验')
    parser.add_argument('--advanced', action='store_true', help='运行高级消融实验（优化器、激活函数、位置编码、层归一化等）')
    parser.add_argument('--all', action='store_true', help='运行所有实验类型（基础、高级、warmup、模块实验）')
    parser.add_argument('--max_concurrent', type=int, default=2, help='最大并发实验数量 (默认: 2)')
    parser.add_argument('--gpu_memory_threshold', type=float, default=0.8, help='GPU内存使用阈值，超过此值将减少并发数 (默认: 0.8)')
    args = parser.parse_args()
    
    # 检查GPU可用性和内存
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        print(f"检测到GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU总内存: {gpu_memory:.1f} GB")
        
        # 根据GPU内存调整并发数
        if gpu_memory < 8:  # 小于8GB
            recommended_concurrent = 1
            print("⚠️  GPU内存较小，建议使用 --max_concurrent 1")
        elif gpu_memory < 16:  # 8-16GB
            recommended_concurrent = 2
        else:  # 大于16GB
            recommended_concurrent = min(args.max_concurrent, 4)
            
        if args.max_concurrent > recommended_concurrent:
            print(f"⚠️  建议将并发数调整为 {recommended_concurrent}，以避免GPU内存不足")
    else:
        print("⚠️  未检测到GPU，将使用CPU训练（可能会很慢）")
        
    print(f"将使用最大并发数: {args.max_concurrent}")
    
    if args.custom:
        # 这里可以添加自定义消融实验的代码
        print("运行自定义消融实验...")
        # 示例: 自定义实验代码
    elif args.all:
        # 运行所有实验类型
        run_all_experiments(max_concurrent=args.max_concurrent)
    elif args.warmup:
        # 运行不同warmup比例的实验
        run_custom_warmup_experiments(max_concurrent=args.max_concurrent)
    elif args.modules:
        # 运行Transformer模块消融实验
        run_transformer_modules_ablation(max_concurrent=args.max_concurrent)
    elif args.advanced:
        # 运行高级消融实验
        run_advanced_ablation_study(max_concurrent=args.max_concurrent)
    else:
        # 运行预定义的消融实验
        run_ablation_study(max_concurrent=args.max_concurrent) 
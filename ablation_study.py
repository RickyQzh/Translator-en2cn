import os
import json
import time
import shutil
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from train import train_model, calculate_warmup_steps

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

def run_ablation_study():
    """执行消融实验"""
    # 创建主实验目录
    base_dir = "./ablation"
    os.makedirs(base_dir, exist_ok=True)
    experiment_dir = create_experiment_dir(base_dir)
    
    print(f"创建消融实验，结果将保存在: {experiment_dir}")
    
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
        # 不再硬编码warmup_steps，让系统自动计算
    }
    
    # 估算基本的warmup值
    # 假设每个epoch约有280步 (18000/64)
    approx_steps_per_epoch = 280
    
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
            'name': 'layers',
            'config': {**base_config, 'num_encoder_layers': 3, 'num_decoder_layers': 3, 'ablation_keys': ['num_encoder_layers', 'num_decoder_layers']},
            'description': '减少编码器和解码器层数'
        },
        {
            'name': 'layers',
            'config': {**base_config, 'num_encoder_layers': 9, 'num_decoder_layers': 9, 'ablation_keys': ['num_encoder_layers', 'num_decoder_layers']},
            'description': '增加编码器和解码器层数'
        },
        
        # 4. 变化Dropout率
        {
            'name': 'dropout',
            'config': {**base_config, 'dropout': 0.1, 'ablation_keys': ['dropout']},
            'description': '较小的Dropout率'
        },
        {
            'name': 'dropout',
            'config': {**base_config, 'dropout': 0.3, 'ablation_keys': ['dropout']},
            'description': '较大的Dropout率'
        },
        
        # 5. 变化学习率
        {
            'name': 'learning_rate',
            'config': {**base_config, 'learning_rate': 5e-4, 'ablation_keys': ['learning_rate']},
            'description': '较小的学习率'
        },
        {
            'name': 'learning_rate',
            'config': {**base_config, 'learning_rate': 2e-3, 'ablation_keys': ['learning_rate']},
            'description': '较大的学习率'
        },
        
        # 6. 变化标签平滑系数
        {
            'name': 'label_smoothing',
            'config': {**base_config, 'label_smoothing': 0.0, 'ablation_keys': ['label_smoothing']},
            'description': '无标签平滑'
        },
        {
            'name': 'label_smoothing',
            'config': {**base_config, 'label_smoothing': 0.2, 'ablation_keys': ['label_smoothing']},
            'description': '较大的标签平滑系数'
        },
        
        # 7. 变化Warmup比例
        {
            'name': 'warmup_ratio',
            'config': {**base_config, 'warmup_ratio': 0.05, 'ablation_keys': ['warmup_ratio']},
            'description': '较小的预热比例 (5%)'
        },
        {
            'name': 'warmup_ratio',
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
            'name': 'attention_heads',
            'config': {**base_config, 'num_heads': 4, 'ablation_keys': ['num_heads']},
            'description': '减少注意力头数 (4)'
        },
        {
            'name': 'attention_heads',
            'config': {**base_config, 'num_heads': 16, 'ablation_keys': ['num_heads']},
            'description': '增加注意力头数 (16)'
        }
    ]
    
    # 运行所有实验并收集结果
    results = []
    
    for i, experiment in enumerate(ablation_experiments):
        config = experiment['config']
        name = experiment['name']
        description = experiment['description']
        
        # 生成实验名称和路径save
        exp_name = get_experiment_name(config, name)
        exp_output_dir = os.path.join(experiment_dir, exp_name)
        
        print(f"\n{'='*80}")
        print(f"执行实验 [{i+1}/{len(ablation_experiments)}]: {exp_name}")
        print(f"描述: {description}")
        print(f"{'='*80}\n")
        
        # 将配置保存到该实验目录
        os.makedirs(exp_output_dir, exist_ok=True)
        
        # 训练模型
        try:
            exp_results = train_model(config, output_dir=exp_output_dir)
            
            # 收集结果
            result_entry = {
                'experiment': exp_name,
                'description': description,
                **config,
                **exp_results
            }
            results.append(result_entry)
            
            print(f"\n实验 {exp_name} 完成!")
        except Exception as e:
            print(f"实验 {exp_name} 失败: {str(e)}")
    
    # 保存所有实验结果到CSV文件
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(experiment_dir, 'all_results.csv'), index=False)
    
    # 生成结果比较图
    plot_ablation_results(results_df, experiment_dir)
    
    print("\n所有消融实验完成!")
    print(f"结果已保存到: {experiment_dir}")
    
    return results_df

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

def run_custom_warmup_experiments():
    """运行不同warmup比例的实验"""
    base_dir = "./ablation"
    os.makedirs(base_dir, exist_ok=True)
    experiment_dir = os.path.join(base_dir, f"warmup_experiment_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"运行不同warmup比例的实验，结果将保存在: {experiment_dir}")
    
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
    
    # 运行不同warmup比例的实验
    results = []
    
    for i, ratio in enumerate(warmup_ratios):
        config = {**base_config, 'warmup_ratio': ratio}
        
        exp_name = f"warmup_ratio_{ratio}"
        exp_output_dir = os.path.join(experiment_dir, exp_name)
        
        print(f"\n{'='*80}")
        print(f"执行实验 [{i+1}/{len(warmup_ratios)}]: {exp_name}")
        print(f"Warmup比例: {ratio}")
        print(f"{'='*80}\n")
        
        os.makedirs(exp_output_dir, exist_ok=True)
        
        try:
            exp_results = train_model(config, output_dir=exp_output_dir)
            
            result_entry = {
                'experiment': exp_name,
                'warmup_ratio': ratio,
                **exp_results
            }
            results.append(result_entry)
            
            print(f"\n实验 {exp_name} 完成!")
        except Exception as e:
            print(f"实验 {exp_name} 失败: {str(e)}")
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(experiment_dir, 'warmup_results.csv'), index=False)
    
    # 绘制不同warmup比例的性能对比图
    plot_warmup_comparison(results_df, experiment_dir)
    
    print("\n所有warmup实验完成!")
    print(f"结果已保存到: {experiment_dir}")
    
    return results_df

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

def run_transformer_modules_ablation():
    """运行Transformer模块的消融实验"""
    base_dir = "./ablation"
    os.makedirs(base_dir, exist_ok=True)
    experiment_dir = os.path.join(base_dir, f"transformer_modules_ablation_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"运行Transformer模块消融实验，结果将保存在: {experiment_dir}")
    
    # 基本配置
    base_config = {
        'batch_size': 512,
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
    
    # 运行实验
    results = []
    
    for i, experiment in enumerate(experiments):
        config = experiment['config']
        name = experiment['name']
        description = experiment['description']
        
        exp_output_dir = os.path.join(experiment_dir, name)
        
        print(f"\n{'='*80}")
        print(f"执行实验 [{i+1}/{len(experiments)}]: {name}")
        print(f"描述: {description}")
        print(f"{'='*80}\n")
        
        os.makedirs(exp_output_dir, exist_ok=True)
        
        try:
            exp_results = train_model(config, output_dir=exp_output_dir)
            
            result_entry = {
                'experiment': name,
                'description': description,
                **config,
                **exp_results
            }
            results.append(result_entry)
            
            print(f"\n实验 {name} 完成!")
        except Exception as e:
            print(f"实验 {name} 失败: {str(e)}")
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(experiment_dir, 'transformer_modules_results.csv'), index=False)
    
    # 生成报告
    plot_ablation_results(results_df, experiment_dir)
    create_html_report(results_df, experiment_dir)
    
    print("\n所有Transformer模块消融实验完成!")
    print(f"结果已保存到: {experiment_dir}")
    
    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='运行Transformer翻译模型的消融实验')
    parser.add_argument('--custom', action='store_true', help='是否运行自定义消融实验')
    parser.add_argument('--warmup', action='store_true', help='运行不同warmup比例的对比实验')
    parser.add_argument('--modules', action='store_true', help='运行Transformer模块（位置编码、注意力机制等）的消融实验')
    args = parser.parse_args()
    
    if args.custom:
        # 这里可以添加自定义消融实验的代码
        print("运行自定义消融实验...")
        # 示例: 自定义实验代码
    elif args.warmup:
        # 运行不同warmup比例的实验
        run_custom_warmup_experiments()
    elif args.modules:
        # 运行Transformer模块消融实验
        run_transformer_modules_ablation()
    else:
        # 运行预定义的消融实验
        run_ablation_study() 
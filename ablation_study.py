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
        'batch_size': 64,
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
        }
    ]
    
    # 运行所有实验并收集结果
    results = []
    
    for i, experiment in enumerate(ablation_experiments):
        config = experiment['config']
        name = experiment['name']
        description = experiment['description']
        
        # 生成实验名称和路径
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
    # 绘制验证BLEU比较图
    plt.figure(figsize=(14, 8))
    
    # 根据实验名称排序
    results_df = results_df.sort_values('experiment')
    
    # 绘制条形图
    bars = plt.bar(results_df['experiment'], results_df['best_val_bleu'], color='skyblue')
    plt.xlabel('Experiment')
    plt.ylabel('Validation BLEU Score')
    plt.title('Ablation Study Results: Validation BLEU Score Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # 在条形上方显示具体数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                 f'{height:.4f}', ha='center', va='bottom', rotation=0)
    
    plt.savefig(os.path.join(output_dir, 'ablation_bleu_comparison.png'))
    plt.close()
    
    # 绘制测试损失比较图
    plt.figure(figsize=(14, 8))
    bars = plt.bar(results_df['experiment'], results_df['final_test_loss'], color='salmon')
    plt.xlabel('Experiment')
    plt.ylabel('Test Loss')
    plt.title('Ablation Study Results: Test Loss Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # 在条形上方显示具体数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                 f'{height:.4f}', ha='center', va='bottom', rotation=0)
    
    plt.savefig(os.path.join(output_dir, 'ablation_loss_comparison.png'))
    plt.close()
    
    # 创建一个易读的HTML报告
    create_html_report(results_df, output_dir)

def create_html_report(results_df, output_dir):
    """创建HTML格式的实验结果报告"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Transformer Translation Ablation Study Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin-top: 20px; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            tr:hover { background-color: #f5f5f5; }
            .results-container { margin-top: 30px; }
            .plots-container { display: flex; flex-direction: column; align-items: center; margin-top: 30px; }
            .plot-image { margin-top: 20px; max-width: 100%; }
            .highlight { background-color: #e6ffe6; }
            .description { color: #666; font-style: italic; }
        </style>
    </head>
    <body>
        <h1>Transformer Translation Ablation Study Results</h1>
        <p>Generated on: """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
        
        <div class="results-container">
            <h2>Results Table</h2>
            <table>
                <tr>
                    <th>Experiment</th>
                    <th>Description</th>
                    <th>Best Val BLEU</th>
                    <th>Final Test BLEU</th>
                    <th>Parameters</th>
                </tr>
    """
    
    # 获取最佳验证BLEU的实验
    best_val_bleu_exp = results_df.loc[results_df['best_val_bleu'].idxmax()]
    
    # 为每个实验添加一行
    for _, row in results_df.iterrows():
        # 检查是否是最佳实验
        is_best = (row['best_val_bleu'] == best_val_bleu_exp['best_val_bleu'])
        row_class = 'highlight' if is_best else ''
        
        # 提取重要参数
        important_params = {
            'd_model': row.get('d_model', '-'),
            'num_heads': row.get('num_heads', '-'),
            'num_encoder_layers': row.get('num_encoder_layers', '-'),
            'num_decoder_layers': row.get('num_decoder_layers', '-'),
            'dropout': row.get('dropout', '-'),
            'learning_rate': row.get('learning_rate', '-'),
            'label_smoothing': row.get('label_smoothing', '-'),
            'warmup_ratio': row.get('warmup_ratio', '-')
        }
        
        params_str = ", ".join([f"{k}={v}" for k, v in important_params.items() if v != '-'])
        
        html_content += f"""
                <tr class="{row_class}">
                    <td>{row['experiment']}</td>
                    <td class="description">{row.get('description', '-')}</td>
                    <td>{row['best_val_bleu']:.4f}</td>
                    <td>{row['final_test_bleu']:.4f}</td>
                    <td>{params_str}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
        
        <div class="plots-container">
            <h2>Comparison Plots</h2>
            <div>
                <h3>Validation BLEU Score Comparison</h3>
                <img class="plot-image" src="ablation_bleu_comparison.png" alt="BLEU Score Comparison">
            </div>
            <div>
                <h3>Test Loss Comparison</h3>
                <img class="plot-image" src="ablation_loss_comparison.png" alt="Test Loss Comparison">
            </div>
        </div>
    </body>
    </html>
    """
    
    # 保存HTML文件
    with open(os.path.join(output_dir, 'ablation_report.html'), 'w') as f:
        f.write(html_content)

def run_custom_warmup_experiments():
    """运行不同warmup比例的实验"""
    base_dir = "./ablation"
    os.makedirs(base_dir, exist_ok=True)
    experiment_dir = os.path.join(base_dir, f"warmup_experiment_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"运行不同warmup比例的实验，结果将保存在: {experiment_dir}")
    
    # 基本配置
    base_config = {
        'batch_size': 64,
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
    # 按warmup比例排序
    results_df = results_df.sort_values('warmup_ratio')
    
    # 绘制BLEU对比图
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['warmup_ratio'], results_df['best_val_bleu'], 'bo-', label='Validation BLEU')
    plt.plot(results_df['warmup_ratio'], results_df['final_test_bleu'], 'ro-', label='Test BLEU')
    
    for i, row in results_df.iterrows():
        plt.annotate(f"{row['best_val_bleu']:.4f}", 
                    (row['warmup_ratio'], row['best_val_bleu']),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
        plt.annotate(f"{row['final_test_bleu']:.4f}", 
                    (row['warmup_ratio'], row['final_test_bleu']),
                    textcoords="offset points", 
                    xytext=(0,-15), 
                    ha='center')
    
    plt.xlabel('Warmup Ratio')
    plt.ylabel('BLEU Score')
    plt.title('Impact of Warmup Ratio on BLEU Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'warmup_bleu_comparison.png'))
    plt.close()
    
    # 绘制损失对比图
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['warmup_ratio'], results_df['best_val_loss'], 'bo-', label='Best Validation Loss')
    plt.plot(results_df['warmup_ratio'], results_df['final_test_loss'], 'ro-', label='Final Test Loss')
    
    for i, row in results_df.iterrows():
        plt.annotate(f"{row['best_val_loss']:.4f}", 
                    (row['warmup_ratio'], row['best_val_loss']),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
        plt.annotate(f"{row['final_test_loss']:.4f}", 
                    (row['warmup_ratio'], row['final_test_loss']),
                    textcoords="offset points", 
                    xytext=(0,-15), 
                    ha='center')
    
    plt.xlabel('Warmup Ratio')
    plt.ylabel('Loss')
    plt.title('Impact of Warmup Ratio on Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'warmup_loss_comparison.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='运行Transformer翻译模型的消融实验')
    parser.add_argument('--custom', action='store_true', help='是否运行自定义消融实验')
    parser.add_argument('--warmup', action='store_true', help='运行不同warmup比例的对比实验')
    args = parser.parse_args()
    
    if args.custom:
        # 这里可以添加自定义消融实验的代码
        print("运行自定义消融实验...")
        # 示例: 自定义实验代码
    elif args.warmup:
        # 运行不同warmup比例的实验
        run_custom_warmup_experiments()
    else:
        # 运行预定义的消融实验
        run_ablation_study() 
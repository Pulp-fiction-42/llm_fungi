import os
import torch
import yaml
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import (
    get_peft_model,
    get_peft_model_state_dict,
    LoraConfig,
    IA3Config,
    AdaLoraConfig,
    TaskType,
    PeftType
)
from datasets import load_dataset, Dataset, DatasetDict
import optuna
import pickle
import gc
import time

# 辅助函数，用于保存CUDA内存快照
def save_memory_snapshot(filename=None):
    """保存CUDA内存快照到文件中"""
    if not torch.cuda.is_available():
        print("CUDA不可用，无法保存内存快照")
        return
        
    filename = filename or f"memory_snapshot_{time.strftime('%Y%m%d_%H%M%S')}.pickle"
    snapshot = torch.cuda.memory._snapshot()
    with open(filename, 'wb') as f:
        pickle.dump(snapshot, f)
    print(f"内存快照已保存至 {filename}")
    print("\n--- 内存快照摘要 ---")
    print(torch.cuda.memory_summary())
    print("---------------------------\n")
    print(f"您可以在 https://pytorch.org/memory_viz 上查看内存快照详情")

# ==============================================================================
#  ConfigParser类用于解析YAML配置文件
# ==============================================================================
class ConfigParser:
    def __init__(self, config_file):
        """
        初始化配置解析器
        
        Args:
            config_file (str): YAML配置文件的路径
        """
        self.config_file = config_file
        self.config = self._load_config()
        
    def _load_config(self):
        """加载YAML配置文件"""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"配置文件 {self.config_file} 不存在")
            
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)# yaml.safe_load()返回一个字典，该字典包含配置文件中的所有键值对
        return config
    
    def get_experiment_config(self):
        """获取实验基本配置"""
        exp_config = self.config.get('experiment')
        return {
            'model_name': exp_config.get('model_name'),
            'dataset_name': exp_config.get('dataset_name'),
            'task_type': exp_config.get('task_type')
        }
    
    def get_peft_config(self):
        """获取PEFT方法和参数配置"""
        peft_config = self.config.get('peft')
        return {
            'method': peft_config.get('method'),
            'params': peft_config.get('params')
        }
    
    def get_training_config(self):
        """获取训练参数配置"""
        return self.config.get('training')
    
    def get_tuner_config(self):
        """获取超参数优化器配置"""
        tuner_config = self.config.get('tuner')
        return {
            'direction': tuner_config.get('direction'),
            'n_trials': tuner_config.get('n_trials'),
            'study_name': tuner_config.get('study_name')
        }
    
    def get_peft_search_space(self):
        """获取PEFT方法的搜索空间配置"""
        return self.config.get('search_space').get('peft')
    
    def get_training_search_space(self):
        """获取训练参数的搜索空间配置"""
        return self.config.get('search_space').get('training')

# ==============================================================================
#  PEFTExperiment类用于执行PEFT训练和评估
# ==============================================================================
class PEFTExperiment:
    def __init__(self, model_name, dataset_name, task_type, config):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.task_type = task_type
        self.config = config
        # 从本地目录加载tokenizer
        print(f"从本地目录加载tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("pad_token is None, set to eos_token")
    
    def get_peft_config(self, method, **kwargs):
        """获取不同PEFT方法的配置"""
        configs = {
            "lora": LoraConfig(
                task_type=self.task_type,
                inference_mode=False,
                r=kwargs.get("r"),
                lora_alpha=kwargs.get("lora_alpha"),# 决定delta_W的影响
                lora_dropout=kwargs.get("lora_dropout"),
                target_modules=kwargs.get("target_modules")
            ),
            "ia3": IA3Config(
                task_type=self.task_type,
                inference_mode=False,
                target_modules=kwargs.get("target_modules"),
                feedforward_modules=kwargs.get("feedforward_modules")
            )
        }
        return configs[method.lower()]
    
    def prepare_data(self):
        """准备数据集，从parsed_genome.csv文件加载"""
        print("开始准备数据...")
        # 从CSV文件加载数据
        df = pd.read_csv('parsed_genome.csv')
        
        # 计算并记录序列长度信息
        if 'contig' in df.columns:
            df['sequence_length'] = df['contig'].str.len()
            max_length = df['sequence_length'].max()
            avg_length = df['sequence_length'].mean()
            print(f"数据集序列长度统计: 最大长度={max_length}, 平均长度={avg_length}")
            
            # 记录序列长度分布
            length_bins = [0, 512, 1024, 2048, 4096, 8192, 16384, float('inf')]
            length_counts = [((df['sequence_length'] > length_bins[i]) & (df['sequence_length'] <= length_bins[i+1])).sum() 
                            for i in range(len(length_bins)-1)]
            length_labels = [f"{length_bins[i]}-{length_bins[i+1]}" for i in range(len(length_bins)-1)]
            
            print("序列长度分布:")
            for label, count in zip(length_labels, length_counts):
                print(f"  {label}: {count} 条序列")
                
            # 找出最长的序列并记录其ID和长度
            longest_seq_idx = df['sequence_length'].idxmax()
            print(f"最长序列ID: {longest_seq_idx}, 长度: {df.loc[longest_seq_idx, 'sequence_length']}")
        
        # 将数据转换为Hugging Face Dataset格式
        # 使用'contig'列作为文本输入，'organism_name'列作为标签
        train_df = df.sample(frac=0.8, random_state=42)
        val_df = df.drop(train_df.index)
        
        # 创建标签到ID的映射
        unique_labels = df['organism_name'].unique().tolist()
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        
        # 添加标签ID列
        train_df['label'] = train_df['organism_name'].map(label_to_id)
        val_df['label'] = val_df['organism_name'].map(label_to_id)
        
        # 打印调试信息
        print(f"标签类型: {train_df['label'].dtype}")
        print(f"标签示例: {train_df['label'].iloc[:5].tolist()}")
        
        dataset_dict = {
            'train': Dataset.from_pandas(train_df),
            'validation': Dataset.from_pandas(val_df)
        }
        
        # 创建一个Dataset对象
        dataset = DatasetDict(dataset_dict)
        
        # 使用profiler分析tokenize过程
        def tokenize_function(examples):
            # 使用较小的max_length来避免OOM
            max_length = 512  # 降低此值以减少内存使用，可以根据您的数据集和GPU内存调整
            
            # 使用'contig'列作为文本输入，启用截断和填充以防止OOM
            result = self.tokenizer(
                examples["contig"],
                truncation=True,
                padding="max_length",
                max_length=max_length
            )
            
            # 确保标签字段名称正确
            if "label" in examples:
                result["labels"] = examples["label"]  # 保持标签为整型
            return result
        
        print("开始tokenize数据...")
        # 记录tokenize前的内存
        if torch.cuda.is_available():
            before_mem = torch.cuda.memory_allocated() / (1024 * 1024)
            print(f"Tokenize前内存使用: {before_mem:.2f} MB")

        # 执行tokenize操作
        tokenized_dataset = dataset.map(tokenize_function, batched=True, batch_size=32)
        
        # 记录tokenize后的内存
        if torch.cuda.is_available():
            after_mem = torch.cuda.memory_allocated() / (1024 * 1024)
            print(f"Tokenize后内存使用: {after_mem:.2f} MB")
            print(f"Tokenize过程内存增加: {after_mem - before_mem:.2f} MB")
        
        # 打印处理后的数据集信息
        print(f"处理后的数据集结构: {tokenized_dataset}")
        
        # 保存标签映射信息，方便后续预测时使用
        self.label_to_id = label_to_id
        self.id_to_label = {i: label for label, i in label_to_id.items()}
        
        return tokenized_dataset
    
    def train_with_peft(self, peft_method, peft_params, training_params):
        """使用PEFT方法训练模型"""
        # 为TensorBoard创建日志目录
        log_dir = f"./tensorboard_logs/{peft_method}_{self.model_name.split('/')[-1]}_{self.dataset_name}"
        os.makedirs(log_dir, exist_ok=True)
        
        # 准备数据
        print("开始数据准备阶段...")
        dataset = self.prepare_data()
        
        # 获取标签数量
        num_labels = len(self.label_to_id)
        print(f"分类任务标签数量: {num_labels}")
        print(f"标签映射: {self.label_to_id}")
        
        print("开始加载模型...")
        # 记录模型加载前的内存
        if torch.cuda.is_available():
            before_mem = torch.cuda.memory_allocated() / (1024 * 1024)
            print(f"模型加载前内存使用: {before_mem:.2f} MB")
            
        # 从本地目录加载模型
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            problem_type="single_label_classification",
            local_files_only=True,
            trust_remote_code=True
        )
        
        # 记录模型加载后的内存
        if torch.cuda.is_available():
            after_mem = torch.cuda.memory_allocated() / (1024 * 1024)
            print(f"模型加载后内存使用: {after_mem:.2f} MB")
            print(f"模型加载内存增加: {after_mem - before_mem:.2f} MB")
        
        # 应用PEFT配置
        print("应用PEFT配置...")
        if torch.cuda.is_available():
            before_mem = torch.cuda.memory_allocated() / (1024 * 1024)
            print(f"PEFT应用前内存使用: {before_mem:.2f} MB")
            
        peft_config = self.get_peft_config(peft_method, **peft_params)
        model = get_peft_model(model, peft_config)
        
        # 记录PEFT应用后的内存
        if torch.cuda.is_available():
            after_mem = torch.cuda.memory_allocated() / (1024 * 1024)
            print(f"PEFT应用后内存使用: {after_mem:.2f} MB")
            print(f"PEFT应用内存增加: {after_mem - before_mem:.2f} MB")
        
        # 打印模型参数情况
        model.print_trainable_parameters()
        
        # 使用自定义数据整理器
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # 训练参数
        run_name = f"{peft_method}_{self.model_name}_{self.dataset_name}"
        training_args = TrainingArguments(
            output_dir=f"./llm_fne_tune_results/{run_name}",
            learning_rate=float(training_params.get("learning_rate")),  # 转换为浮点数
            per_device_train_batch_size=1,  # 设置为1以避免批处理问题
            num_train_epochs=float(training_params.get("epochs")),  # 转换为浮点数
            logging_steps=100,
            save_total_limit=1,
            report_to="tensorboard",
            logging_dir=log_dir,
            disable_tqdm=False
        )
        
        # 创建一个支持PyTorch Profiler的自定义Trainer类
        class ProfilerTrainer(Trainer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.step_counter = 0
                
            def training_step(self, model, inputs, num_items_in_batch):
                """使用PyTorch Profiler重写训练步骤"""
                self.step_counter += 1
                
                # 每100步记录一次内存使用情况
                if self.step_counter % 100 == 1:
                    # 记录当前的内存使用情况而不是使用profiler
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
                        max_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
                        print(f"\n--- 第{self.step_counter}步训练内存使用 ---")
                        print(f"已分配内存: {allocated:.2f} MB")
                        print(f"已预留内存: {reserved:.2f} MB")
                        print(f"峰值内存使用: {max_mem:.2f} MB")
                        
                        # 保存内存快照
                        save_memory_snapshot(f"step_{self.step_counter}_memory.pickle")
                
                # 正常的训练步骤
                outputs = model(**inputs)
                loss = outputs.loss
                loss.backward()
                return loss.detach()

        # Define a compute_metrics function to calculate and log metrics during training
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            
            # 多分类评估
            accuracy = metrics.accuracy_score(labels, predictions)
            f1_macro = metrics.f1_score(labels, predictions, average='macro')
            f1_weighted = metrics.f1_score(labels, predictions, average='weighted')
            
            return {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted
            }

        # Initialize the Trainer with profiler support
        trainer = ProfilerTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )

        # 启用CUDA内存历史记录
        if torch.cuda.is_available():
            torch.cuda.memory._record_memory_history(enabled=True)
            print("\n--- 训练开始，保存内存快照 ---")
            save_memory_snapshot("start_memory_snapshot.pickle")
            
        start_time = time.time()

        # 开始训练
        profiler_enabled = torch.cuda.is_available()
        if profiler_enabled:
            print("开始训练，每100步记录一次性能分析...")
        
        trainer.train()
        
        # 训练完成后生成内存快照
        if torch.cuda.is_available():
            print("\n--- 训练完成，保存内存快照 ---")
            save_memory_snapshot("final_memory_snapshot.pickle")
            print(f"最大内存使用: {torch.cuda.max_memory_allocated() / (1024 * 1024):.2f} MB")
        
        training_time = time.time() - start_time
        print(f"训练完成，总用时: {training_time:.2f}秒")

        # 记录额外的指标到TensorBoard
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainer.log({
            "trainable_params": trainable_params,
            "training_time": training_time,
            "num_labels": num_labels
        })
        
        print(f"\n训练完成! 可以使用以下命令查看训练日志:")
        print(f"tensorboard --logdir={log_dir}")
        
        return {}


# ==============================================================================
#  定义面向对象的 HyperparameterTuner 类
# ==============================================================================
class HyperparameterTuner:
    """
    一个使用 Optuna 进行超参数优化的封装类。
    
    这个类将超参数搜索空间定义、Optuna study管理和优化执行过程
    """
    
    def __init__(self, experiment, direction="minimize", study_name=None, search_space=None):
        """
        构造函数。
        
        Args:
            experiment: 实现了 .train_with_peft(...) 方法的实验对象。
            direction (str): 优化的方向，"minimize" 或 "maximize"。
            study_name (str, optional): Optuna study 的名称，用于持久化。
            search_space (dict, optional): 超参数搜索空间配置。
        """
        self.experiment = experiment
        self.direction = direction
        self.study_name = study_name
        self.search_space = search_space or {}
        
        # 在构造函数中创建 study 对象，管理整个优化过程的状态
        self.study = optuna.create_study(direction=self.direction, study_name=self.study_name)

    def _objective(self, trial):
        """
        选择参数、训练模型、返回结果
        
        Args:
            trial (optuna.trial.Trial): Optuna 的 trial 对象，用于建议参数。
            
        Returns:
            float: 需要被优化的评估指标（例如，验证集损失）。
        """
        # 1. 获取PEFT方法搜索空间
        peft_space = self.search_space.get('peft')
        peft_methods = peft_space.get('methods')
        
        # 选择PEFT方法
        peft_method = trial.suggest_categorical("peft_method", peft_methods)
        
        # 2. 根据方法选择条件依赖的参数
        peft_params = {}
        if peft_method == "lora":
            lora_space = peft_space.get('lora')
            peft_params = {
                "r": trial.suggest_int("r", 
                                      lora_space.get('r', {}).get('min'), 
                                      lora_space.get('r', {}).get('max'), 
                                      step=lora_space.get('r', {}).get('step')),
                "lora_alpha": trial.suggest_int("lora_alpha", 
                                               lora_space.get('lora_alpha', {}).get('min'), 
                                               lora_space.get('lora_alpha', {}).get('max'), 
                                               step=lora_space.get('lora_alpha', {}).get('step')),
                "lora_dropout": trial.suggest_float("lora_dropout", 
                                                   lora_space.get('lora_dropout', {}).get('min'), 
                                                   lora_space.get('lora_dropout', {}).get('max'))
            }
        elif peft_method == "ia3":
            # IA3 没有额外参数
            peft_params = {}
            
        # 3. 获取训练参数搜索空间
        training_space = self.search_space.get('training', {})
        
        # 定义通用的训练参数
        training_params = {
            "learning_rate": trial.suggest_float("learning_rate", 
                                               training_space.get('learning_rate', {}).get('min'), 
                                               training_space.get('learning_rate', {}).get('max'), 
                                               log=training_space.get('learning_rate', {}).get('log')),
            "batch_size": trial.suggest_categorical("batch_size", 
                                                  training_space.get('batch_size', {}).get('values')),
            "epochs": trial.suggest_int("epochs", 
                                       training_space.get('epochs', {}).get('min'), 
                                       training_space.get('epochs', {}).get('max')),
            "weight_decay": trial.suggest_float("weight_decay", 
                                              training_space.get('weight_decay', {}).get('min'), 
                                              training_space.get('weight_decay', {}).get('max'), 
                                              log=training_space.get('weight_decay', {}).get('log'))
        }
        
        # 4. 使用 self.experiment 调用训练和评估
        results = self.experiment.train_with_peft(peft_method, peft_params, training_params)
        
        # 5. 返回优化的目标值
        return results["eval_loss"]

    def run(self, n_trials):
        """
        启动超参数优化过程。
        
        Args:
            n_trials (int): 要运行的总试验次数。
        """
        # 将类方法 _objective 作为优化目标传入
        self.study.optimize(self._objective, n_trials=n_trials)# callback _objective -> Trail injection -> trial_suggest_*:sampler call -> suggested_parameters -> train -> return eval_loss
        
        print("\n🎉🎉🎉 优化完成! 🎉🎉🎉")

    def summarize_results(self):
        """打印最有目标值及对应超参数组合"""
        if self.study.best_trial:
            print(f"\n📊 最佳结果:")
            print(f"  - 目标值 (eval_loss): {self.study.best_value:.4f}")
            print("  - 最佳参数组合:")
            for key, value in self.study.best_params.items():
                print(f"    - {key}: {value}")
        else:
            print("尚未进行任何试验。")

if __name__ == "__main__":
    import argparse
    
    # 命令行参数
    parser = argparse.ArgumentParser(description="PEFT模型微调脚本")
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'tune'], help='执行模式: train或tune')
    parser.add_argument('--max_length', type=int, default=512, help='最大序列长度，超过此长度的序列将被截断')
    parser.add_argument('--analyze_only', action='store_true', help='仅分析数据集，不进行训练')
    args = parser.parse_args()
    
    # 加载配置
    config_parser = ConfigParser(args.config)
    
    # 获取实验配置
    exp_config = config_parser.get_experiment_config()
    
    # 初始化实验
    My_experiment = PEFTExperiment(
        model_name=exp_config['model_name'],
        dataset_name=exp_config['dataset_name'],
        task_type=exp_config['task_type'],
        config=config_parser.config
    )
    
    if args.analyze_only:
        # 只分析数据集不训练
        print("仅分析数据集...")
        
        # 记录开始时的内存
        if torch.cuda.is_available():
            before_mem = torch.cuda.memory_allocated() / (1024 * 1024)
            print(f"数据分析前内存使用: {before_mem:.2f} MB")
            
        dataset = My_experiment.prepare_data()
        
        # 记录完成后的内存
        if torch.cuda.is_available():
            after_mem = torch.cuda.memory_allocated() / (1024 * 1024)
            print(f"数据分析后内存使用: {after_mem:.2f} MB")
            print(f"数据分析内存增加: {after_mem - before_mem:.2f} MB")
            
            # 记录峰值内存
            max_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
            print(f"数据分析峰值内存使用: {max_mem:.2f} MB")
        exit(0)
    
    if args.mode == 'train':
        # 获取PEFT和训练配置
        peft_config = config_parser.get_peft_config()#.yaml文件中的peft配置
        training_config = config_parser.get_training_config()#.yaml文件中的training配置
        
        # 执行单次训练
        results = My_experiment.train_with_peft(
            peft_method=peft_config['method'],
            peft_params=peft_config['params'],
            training_params=training_config
        )
        print(f"训练完成✅")
        
    else:  # tune模式
        # 获取超参数优化器配置
        tuner_config = config_parser.get_tuner_config()
        
        # 获取搜索空间
        search_space = {
            'peft': config_parser.get_peft_search_space(),
            'training': config_parser.get_training_search_space()
        }
        
        # 执行超参数优化
        tuner = HyperparameterTuner(
            experiment=My_experiment,
            direction=tuner_config['direction'],
            study_name=tuner_config['study_name'],
            search_space=search_space
        )
        tuner.run(n_trials=tuner_config['n_trials'])
        tuner.summarize_results()
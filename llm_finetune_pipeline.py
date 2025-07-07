import os
import torch
import wandb
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
    def __init__(self, model_name, dataset_name, task_type):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.task_type = task_type
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
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
            ),
            "adalora": AdaLoraConfig(
                task_type=self.task_type,
                inference_mode=False,
                r=kwargs.get("r"),
                lora_alpha=kwargs.get("lora_alpha"),
                target_r=kwargs.get("target_r"),
                init_r=kwargs.get("init_r"),
                tinit=kwargs.get("tinit"),
                tfinal=kwargs.get("tfinal"),
                deltaT=kwargs.get("deltaT"),
                lora_dropout=kwargs.get("lora_dropout"),
                target_modules=kwargs.get("target_modules")
            )
        }
        return configs[method.lower()]
    
    def prepare_data(self):
        """准备数据集，从parsed_genome.csv文件加载"""
        # 从CSV文件加载数据
        df = pd.read_csv('parsed_genome.csv')
        
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
        
        dataset_dict = {
            'train': Dataset.from_pandas(train_df),
            'validation': Dataset.from_pandas(val_df)
        }
        
        # 创建一个Dataset对象
        dataset = DatasetDict(dataset_dict)
        
        def tokenize_function(examples):
            # 使用'contig'列作为文本输入，不使用截断和填充
            return self.tokenizer(
                examples["contig"],
                truncation=False,
                padding=False
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # 保存标签映射信息，方便后续预测时使用
        self.label_to_id = label_to_id
        self.id_to_label = {i: label for label, i in label_to_id.items()}
        
        return tokenized_dataset
    
    def train_with_peft(self, peft_method, peft_params, training_params):
        """使用PEFT方法训练模型"""

        # 初始化wandb
        run = wandb.init(
            project="fungi_LLM_finetune",
            name=f"{peft_method}_{self.model_name}_{self.dataset_name}_{run.id}",
            config={
                "model": self.model_name,
                "peft_method": peft_method,
                "peft_params": peft_params,#参数高效微调超参数
                **training_params#训练超参数
            }
        )
        
        # 准备数据
        dataset = self.prepare_data()# 返回tokenized_dataset
        
        # 获取标签数量
        num_labels = len(self.label_to_id)
        print(f"分类任务标签数量: {num_labels}")
        print(f"标签映射: {self.label_to_id}")
        
        # 加载模型
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels  # 根据organism_name的唯一值数量设置
        )   
        # 应用PEFT配置
        peft_config = self.get_peft_config(peft_method, **peft_params)
        model = get_peft_model(model, peft_config)#应用配置后的PEFT模型
        
        # 自定义数据整理器，使用原始token序列，不进行填充
        def custom_data_collator(features):
            # 只将input_ids和labels收集到一起，不进行填充操作
            batch = {}
            batch["input_ids"] = [feature["input_ids"] for feature in features]
            batch["attention_mask"] = [feature["attention_mask"] for feature in features]
            batch["labels"] = torch.tensor([feature["label"] for feature in features])
            return batch
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=f"./llm_fne_tune_results/{peft_method}_{self.model_name}_{self.dataset_name}_{run.id}",

            learning_rate=training_params.get("learning_rate"),
            per_device_train_batch_size=1,  # 设置为1以避免批处理问题
            num_train_epochs=training_params.get("epochs"),
            #weight_decay=training_params.get("weight_decay"),

            logging_steps=100,
            evaluation_strategy="epoch",# 定义evaluation时机
            save_strategy="epoch",# 定义模型保存时机
            load_best_model_at_end=True,# 确保训练结束时加载表现最佳的模型
            metric_for_best_model="eval_loss",# 评判保存的标准
            save_total_limit=1,# 每当新检查点保存时会删除旧检查点
            
            report_to="wandb",
            disable_tqdm=False  # 确保不禁用tqdm进度条
        )
        
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

        # Initialize the Trainer with compute_metrics
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"] if "validation" in dataset else dataset["test"],
            tokenizer=self.tokenizer,
            data_collator=custom_data_collator,  # 使用自定义数据整理器
            compute_metrics=compute_metrics# metrics for evaluation
        )

        # Train the model
        import time
        start_time = time.time()
        
        # 使用tqdm显示训练进度
        print("\n开始训练模型...")
        total_steps = int(len(dataset["train"]) / training_args.per_device_train_batch_size * training_args.num_train_epochs)
        with tqdm(total=total_steps, desc="训练进度") as pbar:
            # 创建一个回调函数来更新进度条
            class TqdmCallback:
                def __init__(self, pbar):
                    self.pbar = pbar
                
                def on_log(self, args, state, control, logs=None, **kwargs):
                    if state.is_local_process_zero and logs:
                        _ = logs.pop("total_flos", None)
                        if state.global_step > 0:
                            self.pbar.update(1)
            
            # 添加回调函数到trainer
            trainer.add_callback(TqdmCallback(pbar))
            trainer.train()
        
        training_time = time.time() - start_time

        # Log the training time and number of trainable parameters
        wandb.log({
            "trainable_params": model.get_nb_trainable_parameters(),
            "training_time": training_time,
            "num_labels": num_labels
        })

        wandb.finish()
        
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
        else:  # adalora
            adalora_space = peft_space.get('adalora', {})
            peft_params = {
                "r": trial.suggest_int("r", 
                                      adalora_space.get('r', {}).get('min'), 
                                      adalora_space.get('r', {}).get('max'), 
                                      step=adalora_space.get('r', {}).get('step')),
                "target_r": trial.suggest_int("target_r", 
                                             adalora_space.get('target_r', {}).get('min'), 
                                             adalora_space.get('target_r', {}).get('max'), 
                                             step=adalora_space.get('target_r', {}).get('step')),
                "lora_alpha": trial.suggest_int("lora_alpha", 
                                               adalora_space.get('lora_alpha', {}).get('min'), 
                                               adalora_space.get('lora_alpha', {}).get('max'), 
                                               step=adalora_space.get('lora_alpha', {}).get('step'))
            }
            
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
    args = parser.parse_args()
    
    # 加载配置
    config_parser = ConfigParser(args.config)
    
    # 获取实验配置
    exp_config = config_parser.get_experiment_config()
    
    # 初始化实验
    My_experiment = PEFTExperiment(
        model_name=exp_config['model_name'],
        dataset_name=exp_config['dataset_name'],
        task_type=exp_config['task_type']
    )
    
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
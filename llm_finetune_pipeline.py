import os
import torch
import wandb
import yaml
import numpy as np
import sklearn.metrics as metrics
from transformers import (
    AutoModel,
    AutoTokenizer,
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
from datasets import load_dataset
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
        """准备数据集"""
        dataset = load_dataset(self.dataset_name)
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"] if "text" in examples else examples["sentence"],
                truncation=True,
                padding=False,
                max_length=512
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset
    
    def train_with_peft(self, peft_method, peft_params, training_params):
        """使用PEFT方法训练模型"""

        # 初始化wandb
        run = wandb.init(
            project="DNA_LLM_finetune",
            name=f"{peft_method}_{self.model_name}_{self.dataset_name}",
            config={
                "model": self.model_name,
                "peft_method": peft_method,
                "peft_params": peft_params,#参数高效微调超参数
                **training_params#训练超参数
            }
        )
        
        # 加载模型
        model = AutoModel.from_pretrained(
            self.model_name,
            num_labels=2  # 根据任务调整
        )
        
        # 应用PEFT配置
        peft_config = self.get_peft_config(peft_method, **peft_params)
        model = get_peft_model(model, peft_config)#应用配置后的PEFT模型
        
        # 准备数据
        dataset = self.prepare_data()
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)#动态padding一个batch中的所有样本
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=f"./results/{peft_method}_{run.id}",
            learning_rate=training_params.get("learning_rate"),
            per_device_train_batch_size=training_params.get("batch_size"),
            num_train_epochs=training_params.get("epochs"),
            weight_decay=training_params.get("weight_decay"),
            logging_steps=10,
            evaluation_strategy="epoch",# 定义evaluation时机
            save_strategy="epoch",# 定义模型保存时机
            load_best_model_at_end=True,# 确保训练结束时加载表现最佳的模型
            metric_for_best_model="eval_loss",# 评判保存的标准
            save_total_limit=1,# 每当新检查点保存时会删除旧检查点
            report_to="wandb"
        )
        
        # Define a compute_metrics function to calculate and log metrics during training
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            precision, recall, f1, _ = metrics.precision_recall_fscore_support(labels, predictions, average='binary')
            acc = metrics.accuracy_score(labels, predictions)
            return {
                'accuracy': acc,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }

        # Initialize the Trainer with compute_metrics
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"] if "validation" in dataset else dataset["test"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics# metrics for evaluation
        )

        # Train the model
        import time
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time

        # Log the training time and number of trainable parameters
        wandb.log({
            "trainable_params": model.get_nb_trainable_parameters(),
            "training_time": training_time
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
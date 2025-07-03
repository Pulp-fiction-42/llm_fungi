# LLM Fine-tuning Pipeline

参数高效微调（PEFT）模型训练和超参数优化工具。

## 功能特性

- 支持多种PEFT方法（LoRA、IA3、AdaLoRA）
- 基于YAML配置文件的参数配置
- 支持单次训练和超参数优化两种模式
- 使用Optuna进行高效超参数优化
- 与Weights & Biases集成，用于实验跟踪

## 安装依赖

```bash
pip install torch transformers datasets peft optuna wandb pyyaml
```

## 使用方法

### 1. 准备配置文件

创建一个YAML格式的配置文件，例如`config.yaml`，设置模型、数据集和训练参数。可参考提供的示例配置文件。

### 2. 运行单次训练

```bash
python llm_finetune_pipeline.py --config config.yaml --mode train
```

### 3. 运行超参数优化

```bash
python llm_finetune_pipeline.py --config config.yaml --mode tune
```

## 配置文件说明

配置文件分为以下几个部分：

1. **experiment**: 实验基本配置，包括模型名称、数据集和任务类型
2. **peft**: PEFT方法和参数配置（用于单次训练）
3. **training**: 训练超参数配置（用于单次训练）
4. **tuner**: 超参数优化器配置
5. **search_space**: 超参数搜索空间配置（用于超参数优化）

### 示例配置

```yaml
# 实验基本配置
experiment:
  model_name: "bert-base-uncased"
  dataset_name: "imdb"
  task_type: "sequence_classification"

# PEFT方法配置
peft:
  method: "lora"
  params:
    r: 16
    lora_alpha: 32
    lora_dropout: 0.1

# 训练参数配置
training:
  learning_rate: 5e-4
  batch_size: 16
  epochs: 3
  weight_decay: 0.01
```

完整的配置示例可参考项目中的`config.yaml`文件。

## 注意事项

- 确保已安装所有依赖包
- 使用超参数优化前建议设置wandb账号，以便跟踪实验结果
- 对于大型模型，请确保有足够的GPU内存 
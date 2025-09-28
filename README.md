# Salila Text2SQL 评估体系

基于ragas框架构建的Salila Text2SQL任务评估体系，用于评估Text2SQL模型的性能。

## 项目介绍

本项目提供了一个完整的评估框架，用于评估Text2SQL模型在Salila数据集上的表现。评估体系基于ragas框架，支持多种评估指标，包括忠实度、答案相关性、上下文精确率、上下文召回率和答案正确性等。

## 环境设置

### 1. 创建conda环境

```bash
conda create -n rag-eval python=3.10 -y
conda activate rag-eval
```

### 2. 安装依赖包

```bash
pip install -r requirements.txt
```

## 评估指标解释

本评估体系使用以下指标来评估Text2SQL模型的性能：

- **Faithfulness (忠实度)**: 评估模型生成的SQL查询是否忠实于提供的上下文信息
- **Answer Relevancy (答案相关性)**: 评估生成的SQL查询与问题的相关性
- **Context Precision (上下文精确率)**: 评估上下文中包含的信息对于回答问题的精确性
- **Context Recall (上下文召回率)**: 评估上下文中包含的信息对于回答问题的完整性
- **Answer Correctness (答案正确性)**: 评估生成的SQL查询与真实SQL查询的匹配程度

## 使用方法

### 1. 准备数据集

数据集需要包含以下字段：
- `question`: 用户的自然语言问题
- `context`: 提供的数据库模式或相关上下文信息
- `ground_truth`: 正确的SQL查询

支持的文件格式：CSV或JSON

### 2. 设置OpenAI API密钥（可选）

> 注意：如果您选择使用xinference本地模型，可以跳过此步骤

某些评估指标需要使用OpenAI的模型，可以通过以下方式设置API密钥：

```bash
# Windows
set OPENAI_API_KEY=your_api_key_here

# Linux/Mac
export OPENAI_API_KEY=your_api_key_here
```

或者在代码中直接设置：

```python
evaluator = SalilaText2SQLEvaluator(openai_api_key="your_api_key_here")
```

### 3. 使用xinference本地模型

如果您没有OpenAI API密钥，可以使用xinference本地部署的模型进行评估。

#### 3.1 安装并启动xinference服务

```bash
# 安装xinference（已包含在requirements.txt中）
# pip install xinference

# 启动xinference服务
xinference serve
```

#### 3.2 部署模型

在xinference服务中部署所需的大语言模型和嵌入模型：

```bash
# 部署大语言模型（示例使用llama-2模型）
xinference launch-model --model-name llama-2-chinese-7b-chat --model-type llm

# 部署嵌入模型
xinference launch-model --model-name text-embedding-3-small --model-type embedding
```

#### 3.3 在代码中使用xinference本地模型

```python
from main import SalilaText2SQLEvaluator

# 创建评估器实例，使用xinference本地模型
evaluator = SalilaText2SQLEvaluator(
    use_xinference=True,
    xinference_server_url="http://localhost:9997",  # 默认xinference服务器地址
    # 可选：指定具体的模型UID
    # llm_model_uid="your_llm_model_uid",
    # embedding_model_uid="your_embedding_model_uid"
)

# 后续操作与使用OpenAI API相同
```

#### 3.4 使用带有API密钥的xinference本地模型

如果您的xinference服务器设置了API密钥认证，可以通过以下方式配置：

```python
from main import SalilaText2SQLEvaluator

# 方法1：直接在代码中设置API密钥
evaluator = SalilaText2SQLEvaluator(
    use_xinference=True,
    xinference_server_url="http://localhost:9997",
    llm_model_uid="your_llm_model_uid",
    embedding_model_uid="your_embedding_model_uid",
    xinference_api_key="your_xinference_api_key_here"  # 设置API密钥
)

# 方法2：从环境变量获取API密钥
import os

xinference_api_key = os.getenv('XINFERENCE_API_KEY')
evaluator = SalilaText2SQLEvaluator(
    use_xinference=True,
    xinference_server_url="http://localhost:9997",
    llm_model_uid="your_llm_model_uid",
    embedding_model_uid="your_embedding_model_uid",
    xinference_api_key=xinference_api_key
)
```

设置环境变量的方法：

```bash
# Windows
set XINFERENCE_API_KEY=your_xinference_api_key_here

# Linux/Mac
export XINFERENCE_API_KEY=your_xinference_api_key_here
```

### 3. 运行评估

```python
from main import SalilaText2SQLEvaluator

# 创建评估器实例
evaluator = SalilaText2SQLEvaluator()

# 加载数据集
evaluator.load_dataset("path_to_salila_dataset.json")

# 获取模型预测结果
# 注意：这里应该替换为实际的Text2SQL模型预测
predictions = your_text2sql_model.generate_predictions(evaluator.dataset['question'], evaluator.dataset['context'])

# 执行评估
results = evaluator.evaluate_model(predictions)

# 保存评估结果
evaluator.save_results("evaluation_results.json")

# 打印评估结果
print("评估结果:")
for metric, score in results.items():
    print(f"{metric}: {score:.4f}")
```

### 4. 运行示例

主脚本中包含了一个简单的示例，可以直接运行：

```bash
python main.py
```

这个示例会创建一个小型的测试数据集，并使用示例预测来演示评估流程。

## 项目结构

```
rag-eval/
├── main.py                          # 主脚本文件，包含评估器实现
├── requirements.txt                 # 项目依赖
├── README.md                        # 项目说明文档
├── salila_text2sql_evaluation.ipynb # 交互式评估指南（使用OpenAI API）
├── salila_text2sql_evaluation_with_xinference.ipynb # 交互式评估指南（使用xinference本地模型）
├── salila_text2sql_evaluation_with_xinference_api_key.ipynb # 交互式评估指南（使用带API密钥的xinference本地模型）
├── sample_salila_dataset.json       # 示例数据集
└── evaluation_results.json          # 评估结果示例（运行后生成）
```

## 注意事项

1. 确保您的数据集格式正确，包含所有必要的字段
2. 对于较大的数据集，评估过程可能需要较长时间
3. 使用OpenAI API需要有效的API密钥，并且可能产生费用
4. 评估结果会保存为JSON和CSV两种格式，方便查看和进一步分析

## 扩展建议

1. 根据实际需求，可以添加更多的评估指标
2. 可以实现可视化功能，直观展示评估结果
3. 可以扩展支持更多类型的Text2SQL数据集
4. 可以添加交叉验证功能，提高评估的可靠性

## License

[MIT](https://opensource.org/licenses/MIT)
import os
import pandas as pd
from ragas import evaluate
from ragas.dataset_schema import EvaluationResult
from ragas.executor import Executor
from ragas.metrics import ( faithfulness, answer_relevancy, context_precision,
                           context_recall, answer_correctness )
from datasets import Dataset
import openai
import json
import sqlite3
from typing import List, Dict, Any, Optional, Union
import logging
from xinference.client import Client

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SalilaText2SQLEvaluator:
    """用于评估Text2SQL模型性能的评估器"""
    
    def __init__(self, openai_api_key: str = None, use_xinference: bool = False,
                 xinference_server_url: str = "http://localhost:9997",
                 llm_model_uid: str = None, embedding_model_uid: str = None,
                 xinference_api_key: str = None):
        """初始化评估器
        
        Args:
            openai_api_key: OpenAI API密钥，如果未提供则尝试从环境变量获取
            use_xinference: 是否使用xinference本地模型
            xinference_server_url: xinference服务器URL
            llm_model_uid: 大语言模型的UID
            embedding_model_uid: 嵌入模型的UID
            xinference_api_key: xinference服务器的API密钥
        """
        self.use_xinference = use_xinference
        self.dataset = None
        self.evaluation_results: Union[EvaluationResult, Executor] = None
        
        if self.use_xinference:
            # 连接到xinference本地服务器
            logger.info(f"连接到xinference服务器: {xinference_server_url}")
            try:
                # 如果提供了API密钥，则在连接时使用
                if xinference_api_key:
                    self.xinference_client = Client(
                        xinference_server_url,
                        api_key=xinference_api_key
                    )
                    self.api_key = xinference_api_key
                else:
                    self.xinference_client = Client(xinference_server_url)

                self.base_url = xinference_server_url
                self.llm_model_uid = llm_model_uid
                self.embedding_model_uid = embedding_model_uid
                logger.info("成功连接到xinference服务器")
            except Exception as e:
                logger.error(f"连接xinference服务器失败: {str(e)}")
                raise
        else:
            # 使用OpenAI API
            if openai_api_key:
                openai.api_key = openai_api_key
            else:
                openai.api_key = os.getenv("OPENAI_API_KEY")
            
            if not openai.api_key:
                logger.warning("OpenAI API密钥未设置，某些评估指标可能无法使用")
        
    def load_dataset(self, file_path: str) -> None:
        """加载Salila Text2SQL数据集
        
        Args:
            file_path: 数据集文件路径（CSV或JSON格式）
        """
        logger.info(f"正在加载数据集: {file_path}")
        
        if file_path.endswith('.csv'):
            self.dataset = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.dataset = pd.DataFrame(data)
        else:
            raise ValueError("不支持的文件格式，仅支持CSV和JSON格式")
        
        # 确保数据集包含必要的列
        required_columns = ['question', 'context', 'ground_truth']
        for col in required_columns:
            if col not in self.dataset.columns:
                raise ValueError(f"数据集缺少必要的列: {col}")
        
        logger.info(f"数据集加载完成，共包含 {len(self.dataset)} 条记录")
        
    def prepare_ragas_dataset(self, predictions: List[str]) -> Dataset:
        """准备ragas评估所需的数据集格式
        
        Args:
            predictions: 模型对问题的预测答案列表
        
        Returns:
            ragas评估所需的Dataset对象
        """
        if self.dataset is None:
            raise ValueError("请先加载数据集")
        
        if len(predictions) != len(self.dataset):
            raise ValueError(f"预测数量与数据集大小不匹配: {len(predictions)} vs {len(self.dataset)}")
        
        # 准备ragas所需的数据格式
        data = {
            'question': self.dataset['question'].tolist(),
            'contexts': self.dataset['context'].apply(lambda x: [x] if isinstance(x, str) else x).tolist(),
            'answer': predictions,
            'ground_truth': self.dataset['ground_truth'].tolist()
        }
        
        return Dataset.from_dict(data)
        
    def evaluate_model(self, predictions: List[str], metrics: List = None) -> Dict[str, float]:
        """使用ragas评估模型性能
        
        Args:
            predictions: 模型对问题的预测答案列表
            metrics: 要使用的评估指标列表，默认为None（使用所有指标）
        
        Returns:
            评估结果字典，包含各项指标的得分
        """
        logger.info("开始评估模型性能")
        
        # 准备ragas数据集
        ragas_dataset = self.prepare_ragas_dataset(predictions)
        
        # 使用指定的评估指标
        if metrics is None:
            metrics = [faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness]
        
        # 执行评估
        try:
            if self.use_xinference:
                # 使用xinference本地模型进行评估
                logger.info("使用xinference本地模型进行评估")
                # 为每个指标配置使用xinference模型
                from ragas.llms import LangchainLLMWrapper
                from ragas.embeddings import LangchainEmbeddingsWrapper
                from langchain.llms import Xinference
                from langchain.embeddings import XinferenceEmbeddings
                
                # 创建xinference大语言模型
                llm = Xinference(
                    server_url=self.base_url,
                    model_uid=self.llm_model_uid if self.llm_model_uid else "default"
                )
                
                # 创建xinference嵌入模型
                embeddings = XinferenceEmbeddings(
                    server_url=self.xinference_client.base_url,
                    model_uid=self.embedding_model_uid if self.embedding_model_uid else "default"
                )
                
                # 直接在evaluate函数中传入模型配置
                self.evaluation_results = evaluate(
                    ragas_dataset, 
                    metrics=metrics, 
                    llm=LangchainLLMWrapper(llm), 
                    embeddings=LangchainEmbeddingsWrapper(embeddings)
                )
            logger.info("模型评估完成")
            # 直接返回评估结果对象，而不是尝试转换为字典
            return self.evaluation_results
        except Exception as e:
            logger.error(f"评估过程中发生错误: {str(e)}")
            raise
    
    def save_results(self, file_path: str) -> None:
        """保存评估结果
        
        Args:
            file_path: 结果文件保存路径
        """
        if self.evaluation_results is None:
            raise ValueError("请先执行评估")
        
        logger.info(f"正在保存评估结果到: {file_path}")
        
        # 保存评估结果
        # 尝试使用不同的方法处理评估结果对象
        try:
            import pandas as pd
            # 初始化results_dict变量
            results_dict = None
            results_df = None
            
            # 方法1：检查结果是否为列表类型
            if isinstance(self.evaluation_results, list):
                # 如果是列表，尝试转换为字典或DataFrame
                if self.evaluation_results:
                    # 检查第一个元素是否有score属性
                    if hasattr(self.evaluation_results[0], 'score'):
                        results_dict = {}
                        for item in self.evaluation_results:
                            if hasattr(item, 'name'):
                                results_dict[item.name] = item.score
                    else:
                        # 尝试转换为DataFrame
                        results_df = pd.DataFrame(self.evaluation_results)
            # 方法2：尝试直接转换为字典（适用于新版ragas）
            elif hasattr(self.evaluation_results, 'to_dict'):
                results_dict = self.evaluation_results.to_dict()
            # 方法3：尝试转换为DataFrame然后再转字典（适用于旧版ragas）
            elif hasattr(self.evaluation_results, 'to_pandas'):
                results_df = self.evaluation_results.to_pandas()
            # 方法4：检查是否有scores属性
            elif hasattr(self.evaluation_results, 'scores'):
                if hasattr(self.evaluation_results.scores, 'items'):
                    results_dict = dict(self.evaluation_results.scores)
            # 方法5：尝试使用items()方法（如果结果是字典类似对象）
            elif hasattr(self.evaluation_results, 'items'):
                results_dict = dict(self.evaluation_results)
            else:
                # 默认：将结果转换为字符串表示
                results_dict = {'evaluation_results': str(self.evaluation_results)}
                
            # 如果有DataFrame但没有字典，从DataFrame创建字典
            if results_dict is None and results_df is not None:
                # 处理DataFrame，创建合适的字典表示
                if len(results_df) == 1:
                    # 单行DataFrame，直接转换
                    results_dict = results_df.iloc[0].to_dict()
                else:
                    # 多行DataFrame，转换为嵌套字典
                    results_dict = results_df.to_dict(orient='records')
            
            # 保存为JSON格式
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, ensure_ascii=False, indent=2)
            
            # 尝试保存为CSV格式
            try:
                if 'results_df' not in locals():
                    if isinstance(results_dict, dict):
                        # 从字典创建DataFrame
                        results_df = pd.DataFrame([results_dict])
                    else:
                        # 假设results_dict已经是DataFrame
                        results_df = results_dict
                
                csv_file_path = file_path.replace('.json', '.csv')
                results_df.to_csv(csv_file_path, index=False, encoding='utf-8')
            except Exception as csv_error:
                logger.warning(f"保存CSV格式失败: {str(csv_error)}")
                # 继续执行，不中断程序
        except Exception as e:
            logger.error(f"保存评估结果时出错: {str(e)}")
            raise
        
        logger.info("评估结果保存完成")
        
    def generate_sample_predictions(self) -> List[str]:
        """生成示例预测，用于测试评估流程
        
        Returns:
            示例预测列表
        """
        if self.dataset is None:
            raise ValueError("请先加载数据集")
        
        # 简单的示例预测生成逻辑
        sample_predictions = []
        for _, row in self.dataset.iterrows():
            # 这里可以替换为实际的Text2SQL模型预测逻辑
            sample_predictions.append(f"-- 示例SQL查询\nSELECT * FROM table WHERE condition = 'value';")
            
        return sample_predictions

# 示例用法
if __name__ == "__main__":
    # 创建评估器实例 - 这里可以选择使用OpenAI API或xinference本地模型
    # 1. 使用OpenAI API（默认）
    # evaluator = SalilaText2SQLEvaluator()
    
    # 2. 使用xinference本地模型
    # evaluator = SalilaText2SQLEvaluator(
    #     use_xinference=True,
    #     xinference_server_url="http://10.12.130.188:9997",  # 默认xinference服务器地址
    #     # 可以指定具体的模型UID，如果不指定将使用默认模型
    #     llm_model_uid="qwen2.5-instruct",
    #     embedding_model_uid="bge-large-zh-v1.5"
    # )
    
    # 3. 使用带有API密钥的xinference本地模型
    # 方法1：直接在代码中设置API密钥
    # evaluator = SalilaText2SQLEvaluator(
    #     use_xinference=True,
    #     xinference_server_url="http://10.12.130.188:9997",
    #     llm_model_uid="qwen2.5-instruct",
    #     embedding_model_uid="bge-large-zh-v1.5",
    #     xinference_api_key="your_xinference_api_key_here"  # 设置API密钥
    # )
    
    # 方法2：从环境变量获取API密钥
    import os
    
    evaluator = SalilaText2SQLEvaluator(
        use_xinference=True,
        xinference_server_url="http://10.12.130.188:9997",
        llm_model_uid="qwen2.5-instruct",
        embedding_model_uid="bge-large-zh-v1.5",
        # xinference_api_key=os.getenv('XINFERENCE_API_KEY')  # 从环境变量获取API密钥
        xinference_api_key="sk-URHSBTj4hxIpC"  # 从环境变量获取API密钥
    )
    
    try:
        # 注意：这是一个示例，您需要替换为实际的Salila数据集路径
        # evaluator.load_dataset("path_to_salila_dataset.json")
        
        # 为了演示，我们创建一个简单的测试数据集
        test_data = [
            {
                "question": "2023年第一季度的总销售额是多少？",
                "context": "sales表包含日期(date)、产品(product)、销售额(amount)等字段。",
                "ground_truth": "SELECT SUM(amount) FROM sales WHERE date BETWEEN '2023-01-01' AND '2023-03-31';"
            },
            {
                "question": "哪些产品的销售额超过1000元？",
                "context": "products表包含产品ID(id)、产品名称(name)、类别(category)等字段。sales表包含产品ID(product_id)、销售额(amount)等字段。",
                "ground_truth": "SELECT p.name FROM products p JOIN sales s ON p.id = s.product_id WHERE s.amount > 1000;"
            }
        ]
        
        # 将测试数据保存为临时文件
        with open("test_salila_dataset.json", 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        # 加载测试数据集
        evaluator.load_dataset("test_salila_dataset.json")
        
        # 生成示例预测
        sample_predictions = evaluator.generate_sample_predictions()
        
        # 执行评估
        results = evaluator.evaluate_model(sample_predictions)
        
        # 打印评估结果
        print("评估结果:")
        # 添加兼容性处理，支持不同版本的ragas库
        try:
            # 方法1：检查结果是否为列表类型
            if isinstance(results, list):
                if results and hasattr(results[0], 'name') and hasattr(results[0], 'score'):
                    # 处理metrics列表
                    for item in results:
                        if hasattr(item, 'name') and hasattr(item, 'score'):
                            print(f"{item.name}: {item.score:.4f}")
                else:
                    # 普通列表，尝试转换为DataFrame显示
                    try:
                        import pandas as pd
                        results_df = pd.DataFrame(results)
                        print("评估结果数据框:")
                        print(results_df)
                    except:
                        print(f"列表形式的评估结果: {results}")
            # 方法2：尝试使用items()方法（适用于字典类型结果）
            elif hasattr(results, 'items'):
                for metric, score in results.items():
                    print(f"{metric}: {score:.4f}")
            # 方法3：尝试直接访问评估结果属性（适用于EvaluationResult对象）
            elif hasattr(results, 'scores'):
                if hasattr(results.scores, 'items'):
                    for metric, score in results.scores.items():
                        print(f"{metric}: {score:.4f}")
                else:
                    print(f"scores属性: {results.scores}")
            # 方法4：尝试转换为pandas DataFrame（适用于支持to_pandas()的版本）
            elif hasattr(results, 'to_pandas'):
                results_df = results.to_pandas()
                if hasattr(results_df, 'to_dict'):
                    results_dict = results_df.to_dict()
                    for metric, score_dict in results_dict.items():
                        # 提取第一个值（通常是0键对应的评估分数）
                        if score_dict and isinstance(score_dict, dict):
                            score = list(score_dict.values())[0]
                            print(f"{metric}: {score:.4f}")
            # 默认回退方案
            else:
                print(f"原始评估结果: {results}")
                # 尝试将结果转换为字符串并显示
                try:
                    print(str(results))
                except:
                    print("无法显示评估结果")
        except Exception as e:
            logger.error(f"打印评估结果时出错: {str(e)}")
            print(f"警告: 无法正常显示评估结果 - {str(e)}")
        
        # 保存评估结果
        evaluator.save_results("evaluation_results.json")
        
        print("评估流程完成！")
        
    except Exception as e:
        logger.error(f"执行过程中发生错误: {str(e)}")
        raise
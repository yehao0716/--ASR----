from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSeq2SeqLM
import torch
from pathlib import Path
import logging
from config import Config
import shutil

logger = logging.getLogger('TextProcessor')

class BasePunctuator:
    """标点处理基类"""
    def __init__(self):
        self.model = None
        self.tokenizer = None
        
    def add_punctuation(self, text: str) -> str:
        raise NotImplementedError
        
class BertPunctuator(BasePunctuator):
    """使用BERT的轻量级标点处理器"""
    def __init__(self, proxy=None):
        super().__init__()
        self._init_model(proxy)
        
    def _init_model(self, proxy=None):
        """初始化BERT模型"""
        model_path = Config.MODEL_DIR / "bert-chinese-punctuation"
        try:
            if model_path.exists():
                logger.info("加载本地BERT标点模型...")
                self.model = AutoModelForTokenClassification.from_pretrained(str(model_path))
                self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            else:
                logger.info("下载BERT标点模型...")
                proxies = {'http': proxy, 'https': proxy} if proxy else None
                self.model = AutoModelForTokenClassification.from_pretrained(
                    "ckiplab/bert-base-chinese-punctuation",
                    proxies=proxies
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "ckiplab/bert-base-chinese-punctuation",
                    proxies=proxies
                )
                # 保存到本地
                model_path.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(model_path)
                self.tokenizer.save_pretrained(model_path)
        except Exception as e:
            logger.error(f"BERT模型初始化失败: {str(e)}")
            raise

    def add_punctuation(self, text: str) -> str:
        """添加标点符号"""
        if not text:
            return text
            
        try:
            # 对输入文本进行分词
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            
            # 获取模型预测结果
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=2)
            
            # 将预测结果转换为标点符号
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            punctuated_text = ""
            
            # 修改这部分代码，改进标点添加逻辑
            for i, (token, pred) in enumerate(zip(tokens[1:-1], predictions[0][1:-1])):
                # 去除BERT的特殊标记
                token = token.replace("##", "")
                punctuated_text += token
                
                # 只在特定条件下添加标点
                if pred != 0:  # 0表示无标点
                    # 避免在某些情况下添加标点
                    if i < len(tokens) - 3:  # 不是最后一个字符
                        next_token = tokens[i + 2].replace("##", "")
                        # 如果下一个不是特殊标记，且当前位置预测需要标点
                        if next_token not in ['[SEP]', '[PAD]']:
                            punctuated_text += self.get_punctuation(pred.item())
                    else:  # 最后一个字符
                        punctuated_text += self.get_punctuation(pred.item())
                        
            return punctuated_text
            
        except Exception as e:
            logger.error(f"标点添加失败: {str(e)}")
            return text
            
    @staticmethod
    def get_punctuation(pred_id: int) -> str:
        """获取标点符号映射"""
        punct_map = {
            1: "，",  # 逗号
            2: "。",  # 句号
            3: "？",  # 问号
            4: "！",  # 感叹号
            5: "；",  # 分号
            6: "：",  # 冒号
        }
        return punct_map.get(pred_id, "")

class BartProcessor:
    """使用BART的文本纠错处理器"""
    def __init__(self, proxy=None):
        self._init_model(proxy)
        
    def _init_model(self, proxy=None):
        """初始化BART模型"""
        model_path = Config.MODEL_DIR / "bart-chinese-correction"
        try:
            if model_path.exists():
                logger.info("加载本地BART纠错模型...")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))
                self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            else:
                logger.info("下载BART纠错模型...")
                proxies = {'http': proxy, 'https': proxy} if proxy else None
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    "fnlp/bart-base-chinese",
                    proxies=proxies
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "fnlp/bart-base-chinese",
                    proxies=proxies
                )
                # 保存到本地
                model_path.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(model_path)
                self.tokenizer.save_pretrained(model_path)
        except Exception as e:
            logger.error(f"BART模型初始化失败: {str(e)}")
            raise
            
    def correct_text(self, text: str) -> str:
        """文本纠错"""
        if not text:
            return text
            
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True
            )
            
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        except Exception as e:
            logger.error(f"文本纠错失败: {str(e)}")
            return text

class TextProcessor:
    """文本处理器，整合标点和纠错功能"""
    def __init__(self, mode='full', proxy=None):
        """
        初始化文本处理器
        :param mode: 处理模式，可选 'full'（完整处理）, 'punctuation'（仅标点）
        :param proxy: 代理设置
        """
        self.mode = mode
        self.proxy = proxy
        self.punct_processor = BertPunctuator(proxy=proxy)
        self.correction_processor = None
        
        if mode == 'full':
            self.correction_processor = BartProcessor(proxy=proxy)
            
    def process_text(self, text: str, force_mode=None) -> str:
        """
        处理文本
        :param text: 输入文本
        :param force_mode: 强制使用指定模式
        :return: 处理后的文本
        """
        if not text:
            return text
            
        try:
            # 确定处理模式
            current_mode = force_mode or self.mode
            
            # 添加标点符号
            text = self.punct_processor.add_punctuation(text)
            
            # 如果是完整模式，进行错别字纠正
            if current_mode == 'full' and self.correction_processor:
                text = self.correction_processor.correct_text(text)
                
            return text
            
        except Exception as e:
            logger.error(f"文本处理失败: {str(e)}")
            return text
            
    def switch_mode(self, new_mode: str):
        """
        切换处理模式
        :param new_mode: 新的处理模式
        """
        if new_mode == self.mode:
            return
            
        self.mode = new_mode
        if new_mode == 'full' and not self.correction_processor:
            self.correction_processor = BartProcessor(proxy=self.proxy)
import pytest
import torch
from pathlib import Path
from punctuation_processor import TextProcessor, BertPunctuator, BartProcessor
from unittest.mock import Mock, patch
import logging

# 设置日志
logger = logging.getLogger('TestTextProcessor')

@pytest.fixture
def mock_bert_model():
    """模拟BERT模型"""
    class MockBertModel:
        def __call__(self, **inputs):
            # 获取输入序列长度
            seq_length = inputs['input_ids'].shape[1]
            # 创建合适维度的输出
            return Mock(
                logits=torch.zeros((1, seq_length, 7))  # 7是标点类别数（包括无标点）
            )
    return MockBertModel()

@pytest.fixture
def mock_bart_model():
    """模拟BART模型"""
    model = Mock()
    # 模拟generate方法返回一个有意义的序列
    model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])  # 增加更多token
    return model

@pytest.fixture
def mock_tokenizer():
    """模拟分词器"""
    class MockTokenizer:
        def __call__(self, text, **kwargs):
            self.last_text = text
            # 模拟分词结果
            tokens = ['[CLS]'] + list(text) + ['[SEP]']
            input_ids = torch.tensor([[i for i in range(len(tokens))]])
            return {
                'input_ids': input_ids,
                'attention_mask': torch.ones_like(input_ids)
            }
            
        def convert_ids_to_tokens(self, ids):
            # 确保返回正确数量的token
            if self.last_text:
                return ['[CLS]'] + list(self.last_text) + ['[SEP]']
            return ['[PAD]'] * len(ids)
            
        def decode(self, token_ids, skip_special_tokens=True):
            return self.last_text
            
    return MockTokenizer()

class TestBertPunctuator:
    """测试BERT标点处理器"""
    
    @pytest.fixture
    def bert_punctuator(self, mock_bert_model, mock_tokenizer):
        with patch('punctuation_processor.AutoModelForTokenClassification.from_pretrained', 
                  return_value=mock_bert_model):
            with patch('punctuation_processor.AutoTokenizer.from_pretrained',
                      return_value=mock_tokenizer):
                return BertPunctuator(proxy=None)
    
    def test_init(self, bert_punctuator):
        """测试初始化"""
        assert bert_punctuator.model is not None
        assert bert_punctuator.tokenizer is not None
    
    def test_add_punctuation(self, bert_punctuator):
        """测试添加标点"""
        text = "测试文本"
        result = bert_punctuator.add_punctuation(text)
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_empty_text(self, bert_punctuator):
        """测试空文本"""
        assert bert_punctuator.add_punctuation("") == ""
        assert bert_punctuator.add_punctuation(None) == None

class TestBartProcessor:
    """测试BART文本处理器"""
    
    @pytest.fixture
    def bart_processor(self, mock_bart_model, mock_tokenizer):
        with patch('punctuation_processor.AutoModelForSeq2SeqLM.from_pretrained',
                  return_value=mock_bart_model):
            with patch('punctuation_processor.AutoTokenizer.from_pretrained',
                      return_value=mock_tokenizer):
                return BartProcessor(proxy=None)
    
    def test_init(self, bart_processor):
        """测试初始化"""
        assert bart_processor.model is not None
        assert bart_processor.tokenizer is not None
    
    def test_correct_text(self, bart_processor):
        """测试文本纠错"""
        text = "测试文本"
        result = bart_processor.correct_text(text)
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_empty_text(self, bart_processor):
        """测试空文本"""
        assert bart_processor.correct_text("") == ""
        assert bart_processor.correct_text(None) == None

class TestTextProcessor:
    """测试文本处理器"""
    
    @pytest.fixture
    def text_processor(self, mock_bert_model, mock_bart_model, mock_tokenizer):
        """创建文本处理器实例"""
        with patch('punctuation_processor.AutoModelForTokenClassification.from_pretrained',
                  return_value=mock_bert_model):
            with patch('punctuation_processor.AutoModelForSeq2SeqLM.from_pretrained',
                      return_value=mock_bart_model):
                with patch('punctuation_processor.AutoTokenizer.from_pretrained',
                          return_value=mock_tokenizer):
                    processor = TextProcessor(mode='full', proxy=None)
                    # 确保处理器的模型和分词器被正确设置
                    processor.punct_processor.model = mock_bert_model
                    processor.punct_processor.tokenizer = mock_tokenizer
                    if processor.correction_processor:
                        processor.correction_processor.model = mock_bart_model
                        processor.correction_processor.tokenizer = mock_tokenizer
                    return processor
    
    def test_init_full_mode(self, text_processor):
        """测试完整模式初始化"""
        assert text_processor.mode == 'full'
        assert text_processor.punct_processor is not None
        assert text_processor.correction_processor is not None
    
    def test_init_punctuation_mode(self, mock_bert_model, mock_tokenizer):
        """测试仅标点模式初始化"""
        with patch('punctuation_processor.AutoModelForTokenClassification.from_pretrained',
                  return_value=mock_bert_model):
            with patch('punctuation_processor.AutoTokenizer.from_pretrained',
                      return_value=mock_tokenizer):
                processor = TextProcessor(mode='punctuation', proxy=None)
                assert processor.mode == 'punctuation'
                assert processor.punct_processor is not None
                assert processor.correction_processor is None
    
    def test_process_text_full_mode(self, text_processor):
        """测试完整模式文本处理"""
        text = "测试文本"
        # 设置BERT模型的预测结果
        def mock_bert_output(**inputs):
            seq_length = inputs['input_ids'].shape[1]
            logits = torch.zeros((1, seq_length, 7))
            # 在特定位置设置标点预测
            logits[0, 2, 1] = 1  # 在"测试"之间加逗号
            logits[0, -2, 2] = 1  # 在最后加句号
            return Mock(logits=logits)
            
        text_processor.punct_processor.model.side_effect = mock_bert_output
        
        result = text_processor.process_text(text)
        assert isinstance(result, str)
        assert len(result) > 0
        assert text.replace(" ", "") in result.replace("，", "").replace("。", "")
    
    def test_process_text_punctuation_mode(self, text_processor):
        """测试仅标点模式文本处理"""
        text = "测试文本"
        # 设置BERT模型的预测结果
        def mock_bert_output(**inputs):
            seq_length = inputs['input_ids'].shape[1]
            logits = torch.zeros((1, seq_length, 7))
            # 在特定位置设置标点预测
            logits[0, 2, 1] = 1  # 在"测试"之间加逗号
            logits[0, -2, 2] = 1  # 在最后加句号
            return Mock(logits=logits)
            
        text_processor.punct_processor.model.side_effect = mock_bert_output
        
        result = text_processor.process_text(text, force_mode='punctuation')
        assert isinstance(result, str)
        assert len(result) > 0
        assert text.replace(" ", "") in result.replace("，", "").replace("。", "")
    
    def test_switch_mode(self, text_processor):
        """测试模式切换"""
        text_processor.switch_mode('punctuation')
        assert text_processor.mode == 'punctuation'
        
        text_processor.switch_mode('full')
        assert text_processor.mode == 'full'
        assert text_processor.correction_processor is not None
    
    def test_error_handling(self, text_processor):
        """测试错误处理"""
        # 模拟处理失败
        with patch.object(text_processor.punct_processor, 'add_punctuation',
                         side_effect=Exception("测试异常")):
            result = text_processor.process_text("测试文本")
            assert result == "测试文本"  # 应该返回原文本

class TestIntegration:
    """集成测试"""
    
    @pytest.mark.asyncio
    async def test_video_processor_integration(self):
        """测试与视频处理器的集成"""
        from video_processor import VideoSubtitleExtractor
        
        # 创建测试视频处理器
        extractor = VideoSubtitleExtractor(proxy=None)
        
        # 验证文本处理器初始化
        assert extractor.text_processor is not None
        assert isinstance(extractor.text_processor, TextProcessor)
        
        # 模拟音频转文字结果
        mock_text = "测试文本没有标点和错别字"
        
        # 测试文本处理
        processed_text = extractor.text_processor.process_text(mock_text)
        assert isinstance(processed_text, str)
        assert len(processed_text) > 0

def test_model_loading():
    """测试模型加载路径"""
    from config import Config
    
    # 检查模型目录是否存在
    assert Config.MODEL_DIR.exists()
    
    # 检查BERT模型路径
    bert_path = Config.MODEL_DIR / "bert-chinese-punctuation"
    if bert_path.exists():
        assert (bert_path / "config.json").exists()
        
    # 检查BART模型路径
    bart_path = Config.MODEL_DIR / "bart-chinese-correction"
    if bart_path.exists():
        assert (bart_path / "config.json").exists() 
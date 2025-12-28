"""
epub2anki v4.0
现代风格 UI
流程：EPUB提取 → MeCab分词 → 自动过滤 → 用户筛选 → LLM生成 → Anki
"""
import json
import os
import re
import sys
import tempfile
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

# 添加 src 到路径
_src_path = Path(__file__).parent / "src"
if _src_path.exists():
    sys.path.insert(0, str(_src_path))

# ==================== 现代风格 ====================

class ModernStyle:
    """现代扁平化风格"""
    
    # 颜色方案
    BG = "#f5f5f7"           # 浅灰背景
    CARD_BG = "#ffffff"       # 卡片白色
    PRIMARY = "#007aff"       # 主色调（蓝色）
    PRIMARY_DARK = "#0056b3"  # 深蓝
    TEXT = "#1d1d1f"          # 深色文字
    TEXT_SECONDARY = "#86868b" # 次要文字
    BORDER = "#d2d2d7"        # 边框
    SUCCESS = "#34c759"       # 成功绿色
    
    @classmethod
    def apply(cls, root):
        """应用现代风格"""
        style = ttk.Style()
        
        try:
            style.theme_use('clam')
        except:
            pass
        
        # 基础样式
        style.configure('.',
            background=cls.BG,
            foreground=cls.TEXT,
            font=('Segoe UI', 10)
        )
        
        style.configure('TFrame', background=cls.BG)
        style.configure('Card.TFrame', background=cls.CARD_BG)
        style.configure('TLabel', background=cls.BG, foreground=cls.TEXT)
        style.configure('Card.TLabel', background=cls.CARD_BG)
        style.configure('Header.TLabel', font=('Segoe UI', 11, 'bold'))
        style.configure('Secondary.TLabel', foreground=cls.TEXT_SECONDARY)
        
        # 按钮
        style.configure('TButton',
            background=cls.CARD_BG,
            foreground=cls.TEXT,
            borderwidth=1,
            relief='flat',
            padding=(12, 6)
        )
        style.map('TButton',
            background=[('active', cls.BORDER)],
        )
        
        # 主要按钮
        style.configure('Primary.TButton',
            background=cls.PRIMARY,
            foreground='white',
        )
        style.map('Primary.TButton',
            background=[('active', cls.PRIMARY_DARK)],
        )
        
        # 输入框
        style.configure('TEntry',
            fieldbackground=cls.CARD_BG,
            borderwidth=1,
            relief='solid',
            padding=5
        )
        
        # 单选按钮
        style.configure('TRadiobutton', background=cls.BG)
        
        # Treeview
        style.configure('Treeview',
            background=cls.CARD_BG,
            fieldbackground=cls.CARD_BG,
            foreground=cls.TEXT,
            borderwidth=0,
            font=('Segoe UI', 10),
            rowheight=28
        )
        style.configure('Treeview.Heading',
            background=cls.BG,
            foreground=cls.TEXT,
            font=('Segoe UI', 10, 'bold'),
            borderwidth=0
        )
        style.map('Treeview',
            background=[('selected', cls.PRIMARY)],
            foreground=[('selected', 'white')]
        )
        
        # 进度条
        style.configure('TProgressbar',
            background=cls.PRIMARY,
            troughcolor=cls.BORDER,
            borderwidth=0,
            thickness=6
        )
        
        root.configure(bg=cls.BG)


# ==================== 常量 ====================

CACHE_DIR = Path(tempfile.gettempdir()) / "epub2anki"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

JLPT_LEVELS = ["N5", "N4", "N3", "N2", "N1"]


# ==================== 数据类 ====================

@dataclass
class WordCandidate:
    """词汇候选"""
    surface: str          # 表层形式
    base_form: str        # 基本形
    reading: str          # 读音
    pos: str              # 品词
    sentence: str         # 出处句子
    chapter: str          # 章节
    selected: bool = True # 是否选中


# ==================== 用户词典 ====================

class UserVocabulary:
    """用户词典"""
    
    def __init__(self):
        self.learned: Set[str] = set()
        self.ignored: Set[str] = set()
        self._load()
    
    def _get_path(self) -> Path:
        config_dir = Path.home() / ".epub2anki"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / "vocabulary.json"
    
    def _load(self):
        path = self._get_path()
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.learned = set(data.get('learned', []))
                self.ignored = set(data.get('ignored', []))
                print(f"[INFO] 加载用户词典: {len(self.learned)} 已学, {len(self.ignored)} 已忽略")
            except:
                pass
    
    def save(self):
        path = self._get_path()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'learned': sorted(self.learned),
                'ignored': sorted(self.ignored)
            }, f, ensure_ascii=False, indent=2)
    
    def add_learned(self, words: List[str]):
        self.learned.update(words)
        self.save()
    
    def add_ignored(self, words: List[str]):
        self.ignored.update(words)
        self.save()
    
    def is_known(self, word: str) -> bool:
        return word in self.learned or word in self.ignored
    
    def get_stats(self) -> str:
        return f"已学: {len(self.learned)} | 已忽略: {len(self.ignored)}"


# ==================== MeCab 分词器 ====================

class MeCabTokenizer:
    """
    MeCab + UniDic 分词器
    
    UniDic 输出格式（字段数可能因版本不同而变化）：
    常见格式：品詞,品詞細分1,品詞細分2,品詞細分3,活用型,活用形,
              書字形出現形,発音形出現形,書字形基本形,発音形基本形,
              語彙素読み,語彙素,語彙素細分類,類,語種,...
    """
    
    # 保留的品词（内容词）
    CONTENT_POS = {
        '名詞',      # 名词
        '動詞',      # 动词
        '形容詞',    # い形容词
        '形状詞',    # な形容词
        '副詞',      # 副词
        '連体詞',    # 连体词
        '感動詞',    # 感叹词
    }
    
    # 跳过的名词子类
    SKIP_NOUN_TYPES = {
        '数詞',      # 数词：一、二、三
        '助数詞',    # 量词：個、本、枚
        '代名詞',    # 代词：私、彼、これ
    }
    
    # 跳过的动词子类
    SKIP_VERB_TYPES = {
        '非自立可能',  # 补助动词：ている、てしまう
    }
    
    def __init__(self):
        self._tagger = None
        self._dict_type = 'unknown'
        try:
            import MeCab
            try:
                import unidic
                self._tagger = MeCab.Tagger(f'-d "{unidic.DICDIR}"')
                self._dict_type = 'unidic'
            except:
                self._tagger = MeCab.Tagger()
                self._dict_type = 'ipadic'
            self._tagger.parse("テスト")
            print(f"[INFO] MeCab 已加载 (dict: {self._dict_type})")
        except Exception as e:
            print(f"[ERROR] MeCab 加载失败: {e}")
            raise RuntimeError("请安装: pip install mecab-python3 unidic && python -m unidic download")
    
    def _parse_feature(self, surface: str, feature: str) -> dict:
        """
        解析 MeCab 特征字符串
        
        UniDic 字段顺序（不同版本可能略有差异）：
        常见格式：pos1,pos2,pos3,pos4,cType,cForm,lemma,reading,...
        
        pip install unidic 版本通常是：
        0: pos1 (品詞大分類)
        1: pos2 (品詞中分類)
        2-5: 其他品词信息和活用
        6: 書字形基本形 / lemma (汉字)
        7: 発音形基本形 / reading (片假名)
        或者：
        6: lemma (汉字)
        7: reading (片假名)
        """
        parts = feature.split(',')
        n = len(parts)
        
        result = {
            'surface': surface,
            'pos1': parts[0] if n > 0 else '',
            'pos2': parts[1] if n > 1 else '',
            'base_form': surface,
            'reading': '',
        }
        
        if n >= 8:
            # 检测字段内容：汉字字段 vs 假名字段
            # 通过判断是否包含片假名来区分
            
            def is_katakana(s):
                """判断字符串是否主要是片假名"""
                if not s or s == '*':
                    return False
                kata_count = sum(1 for c in s if '\u30a0' <= c <= '\u30ff')
                return kata_count > len(s) / 2
            
            def has_kanji(s):
                """判断字符串是否包含汉字"""
                if not s or s == '*':
                    return False
                return any('\u4e00' <= c <= '\u9fff' for c in s)
            
            # 尝试从多个位置找基本形和读音
            # 策略：找到包含汉字的作为base_form，片假名的作为reading
            
            base_candidates = []
            reading_candidates = []
            
            for i in range(6, min(n, 14)):
                val = parts[i]
                if val and val != '*':
                    if is_katakana(val):
                        reading_candidates.append((i, val))
                    elif has_kanji(val):
                        base_candidates.append((i, val))
            
            # 选择第一个合适的
            if base_candidates:
                result['base_form'] = base_candidates[0][1]
            if reading_candidates:
                result['reading'] = reading_candidates[0][1]
            
            # 如果没找到合适的，尝试固定位置
            if result['base_form'] == surface:
                for i in [6, 7, 10, 11]:
                    if n > i and parts[i] and parts[i] != '*' and not is_katakana(parts[i]):
                        result['base_form'] = parts[i]
                        break
            
            if not result['reading']:
                for i in [7, 6, 9, 11]:
                    if n > i and parts[i] and parts[i] != '*' and is_katakana(parts[i]):
                        result['reading'] = parts[i]
                        break
        
        return result
    
    def debug_parse(self, word: str) -> str:
        """调试：显示单词的完整解析结果"""
        node = self._tagger.parseToNode(word)
        result = []
        while node:
            if node.surface:
                parts = node.feature.split(',')
                result.append(f"surface={node.surface}")
                for i, p in enumerate(parts):
                    result.append(f"  [{i}] {p}")
            node = node.next
        return '\n'.join(result)
    
    def tokenize(self, text: str) -> List[dict]:
        """分词，返回所有token信息"""
        tokens = []
        node = self._tagger.parseToNode(text)
        
        while node:
            surface = node.surface
            feature = node.feature
            
            if surface and feature:
                token = self._parse_feature(surface, feature)
                tokens.append(token)
            
            node = node.next
        
        return tokens
    
    def extract_vocabulary(self, text: str, user_vocab: 'UserVocabulary') -> List[dict]:
        """从文本中提取词汇（去重、过滤）"""
        tokens = self.tokenize(text)
        
        seen = set()
        vocabulary = []
        
        for token in tokens:
            pos1 = token['pos1']
            pos2 = token['pos2']
            surface = token['surface']
            base_form = token['base_form'] or surface
            
            # 1. 只保留内容词
            if pos1 not in self.CONTENT_POS:
                continue
            
            # 2. 跳过特定子类
            if pos1 == '名詞' and pos2 in self.SKIP_NOUN_TYPES:
                continue
            if pos1 == '動詞' and pos2 in self.SKIP_VERB_TYPES:
                continue
            
            # 3. 跳过太短的词
            if len(base_form) < 2:
                continue
            
            # 4. 必须包含汉字（跳过纯假名词）
            if not any('\u4e00' <= c <= '\u9fff' for c in surface):
                continue
            
            # 5. 跳过用户已知的词
            if user_vocab.is_known(base_form):
                continue
            
            # 6. 去重（按基本形）
            if base_form in seen:
                continue
            seen.add(base_form)
            
            vocabulary.append({
                'surface': surface,
                'base_form': base_form,
                'reading': self._kata_to_hira(token['reading']),
                'pos': pos1,
                'pos_detail': pos2,
            })
        
        return vocabulary
    
    def _kata_to_hira(self, text: str) -> str:
        """片假名转平假名"""
        if not text:
            return ""
        return ''.join(
            chr(ord(c) - 96) if '\u30a1' <= c <= '\u30f6' else c
            for c in text
        )


# ==================== 例句查找器 ====================

class SentenceFinder:
    """
    从原文中查找包含指定词汇的例句
    
    滑动窗口法：
    1. 在原文中搜索词汇（表层形式 → 基本形 → 词干）
    2. 找到位置后，向前后扩展到句子边界
    3. 评分选择最佳例句（长度适中 + 词汇位置居中）
    """
    
    # 句子边界标记
    SENTENCE_END = set('。！？」』）】\n')
    SENTENCE_START = set('「『（【\n')
    
    @staticmethod
    def find_sentence(word_surface: str, word_base: str, 
                      text: str, 
                      target_length: int = 50,
                      max_length: int = 80) -> str:
        """
        查找包含指定词汇的例句
        """
        if not text:
            return ""
        
        # 搜索词列表（优先级：表层形式 > 基本形 > 词干）
        search_terms = [word_surface]
        if word_base and word_base != word_surface:
            search_terms.append(word_base)
            # 动词词干（去掉最后一个假名，但至少保留2字符）
            if len(word_base) >= 3:
                stem = word_base[:-1]
                if stem not in search_terms:
                    search_terms.append(stem)
        
        text_len = len(text)
        best_sentence = ""
        best_score = float('inf')
        
        for term in search_terms:
            pos = 0
            while True:
                pos = text.find(term, pos)
                if pos == -1:
                    break
                
                # 向前找句子开始（最多80字符）
                start = pos
                search_start = max(0, pos - 80)
                for i in range(pos - 1, search_start - 1, -1):
                    if i < 0:
                        start = 0
                        break
                    if text[i] in SentenceFinder.SENTENCE_END or text[i] in SentenceFinder.SENTENCE_START:
                        start = i + 1
                        break
                else:
                    start = search_start
                
                # 向后找句子结束（最多80字符）
                end = pos + len(term)
                search_end = min(text_len, end + 80)
                for i in range(end, search_end):
                    if text[i] in SentenceFinder.SENTENCE_END:
                        end = i + 1
                        break
                else:
                    end = search_end
                
                # 提取句子
                sentence = text[start:end].strip()
                
                # 清理换行，但保留引号
                sentence = sentence.replace('\n', '')
                
                # 去掉首尾多余的引号（只去一层）
                if sentence.startswith('「') and sentence.endswith('」'):
                    pass  # 保留完整对话
                elif sentence.startswith('「') and '」' not in sentence:
                    sentence = sentence[1:]  # 去掉不完整的开始引号
                elif sentence.endswith('」') and '「' not in sentence:
                    sentence = sentence[:-1]  # 去掉不完整的结束引号
                
                sentence = sentence.strip()
                
                if len(sentence) < 8:
                    pos += 1
                    continue
                
                # 评分
                length_score = abs(len(sentence) - target_length)
                
                # 词汇位置分数（越靠近中间越好）
                word_pos = sentence.find(word_surface)
                if word_pos == -1:
                    word_pos = sentence.find(term)
                if word_pos >= 0:
                    center_score = abs(word_pos - len(sentence) / 2)
                else:
                    center_score = 50
                
                # 表层形式匹配加分
                term_bonus = 0 if term == word_surface else 10
                
                score = length_score + center_score * 0.3 + term_bonus
                
                if score < best_score and len(sentence) <= max_length:
                    best_score = score
                    best_sentence = sentence
                
                pos += 1
        
        return best_sentence


# ==================== LLM 处理器 ====================

class LLMProcessor:
    """LLM 闪卡生成器"""
    
    def __init__(self, api_key: str, base_url: str, model: str, output_lang: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=180.0)
        self.model = model
        self.meaning_lang = "中文" if output_lang == "Chinese" else "English"
    
    def generate_card(self, word: WordCandidate) -> Optional[dict]:
        """为单个词汇生成闪卡内容"""
        
        # 构建提示词
        sentence_part = f"\n原句：{word.sentence}" if word.sentence else ""
        sentence_instruction = "原句（用<b></b>标记词汇）" if word.sentence else "创建一个简单的例句（用<b></b>标记词汇）"
        
        prompt = f"""为以下日语词汇生成Anki闪卡内容：

词汇：{word.base_form}
读音：{word.reading}
词性：{word.pos}{sentence_part}

请用JSON格式输出：
{{
    "word": "词汇",
    "reading": "平假名读音",
    "meaning": "{self.meaning_lang}释义",
    "sentence": "{sentence_instruction}",
    "sentence_meaning": "{self.meaning_lang}翻译",
    "notes": "用法说明或记忆技巧（可选）"
}}

只输出JSON。"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            content = response.choices[0].message.content
            
            # 提取 JSON
            if '```' in content:
                match = re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
                if match:
                    content = match.group(1)
            
            match = re.search(r'\{[\s\S]*\}', content)
            if match:
                result = json.loads(match.group())
                result['chapter'] = word.chapter
                return result
            
            return None
        except Exception as e:
            print(f"[LLM Error] {word.base_form}: {e}")
            return None


# ==================== 主应用 ====================

class Epub2AnkiApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("epub2anki")
        self.root.geometry("950x720")
        self.root.resizable(True, True)
        
        # 应用现代风格
        ModernStyle.apply(self.root)
        
        # 数据
        self.candidates: List[WordCandidate] = []
        self.user_vocab = UserVocabulary()
        self.stop_flag = threading.Event()
        
        # 组件
        self.tokenizer = None
        
        self._create_ui()
    
    def _create_ui(self):
        # 主容器
        container = ttk.Frame(self.root, padding=20)
        container.pack(fill=tk.BOTH, expand=True)
        
        # === 顶部：标题和统计 ===
        header = ttk.Frame(container)
        header.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(header, text="epub2anki", 
                  font=('Segoe UI', 18, 'bold')).pack(side=tk.LEFT)
        
        self.stats_var = tk.StringVar(value=self.user_vocab.get_stats())
        ttk.Label(header, textvariable=self.stats_var,
                  style='Secondary.TLabel').pack(side=tk.RIGHT)
        
        # === 设置区域 ===
        settings_card = ttk.Frame(container, style='Card.TFrame', padding=15)
        settings_card.pack(fill=tk.X, pady=(0, 10))
        
        # 文件选择行
        file_row = ttk.Frame(settings_card, style='Card.TFrame')
        file_row.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(file_row, text="EPUB 文件", style='Card.TLabel').pack(side=tk.LEFT)
        self.file_path = tk.StringVar()
        file_entry = ttk.Entry(file_row, textvariable=self.file_path, width=55)
        file_entry.pack(side=tk.LEFT, padx=(10, 5))
        ttk.Button(file_row, text="选择文件", command=self._select_file).pack(side=tk.LEFT)
        ttk.Button(file_row, text="提取词汇", style='Primary.TButton', 
                   command=self._extract_words).pack(side=tk.LEFT, padx=(10, 0))
        
        # API 设置行
        api_row = ttk.Frame(settings_card, style='Card.TFrame')
        api_row.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(api_row, text="API Key", style='Card.TLabel').pack(side=tk.LEFT)
        self.api_key = tk.StringVar()
        ttk.Entry(api_row, textvariable=self.api_key, width=28, show="•").pack(side=tk.LEFT, padx=(10, 15))
        
        ttk.Label(api_row, text="Base URL", style='Card.TLabel').pack(side=tk.LEFT)
        self.base_url = tk.StringVar(value="https://api.deepseek.com")
        ttk.Entry(api_row, textvariable=self.base_url, width=28).pack(side=tk.LEFT, padx=(10, 15))
        
        ttk.Label(api_row, text="Model", style='Card.TLabel').pack(side=tk.LEFT)
        self.model = tk.StringVar(value="deepseek-chat")
        ttk.Entry(api_row, textvariable=self.model, width=15).pack(side=tk.LEFT, padx=(10, 0))
        
        # 选项行
        opt_row = ttk.Frame(settings_card, style='Card.TFrame')
        opt_row.pack(fill=tk.X)
        
        ttk.Label(opt_row, text="释义语言", style='Card.TLabel').pack(side=tk.LEFT)
        self.output_lang = tk.StringVar(value="Chinese")
        ttk.Radiobutton(opt_row, text="中文", variable=self.output_lang, 
                        value="Chinese").pack(side=tk.LEFT, padx=(10, 5))
        ttk.Radiobutton(opt_row, text="English", variable=self.output_lang, 
                        value="English").pack(side=tk.LEFT)
        
        # === 词汇列表区域 ===
        list_card = ttk.Frame(container, style='Card.TFrame', padding=15)
        list_card.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 工具栏
        toolbar = ttk.Frame(list_card, style='Card.TFrame')
        toolbar.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(toolbar, text="词汇列表", style='Header.TLabel').pack(side=tk.LEFT)
        
        self.word_count_var = tk.StringVar(value="0 个词汇")
        ttk.Label(toolbar, textvariable=self.word_count_var, 
                  style='Secondary.TLabel').pack(side=tk.LEFT, padx=(15, 0))
        
        # 右侧按钮
        ttk.Button(toolbar, text="全选", command=self._select_all).pack(side=tk.RIGHT, padx=2)
        ttk.Button(toolbar, text="全不选", command=self._deselect_all).pack(side=tk.RIGHT, padx=2)
        ttk.Button(toolbar, text="反选", command=self._toggle_selection).pack(side=tk.RIGHT, padx=2)
        
        # Treeview 容器
        tree_frame = ttk.Frame(list_card, style='Card.TFrame')
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        # 词汇列表
        columns = ('selected', 'word', 'reading', 'pos', 'sentence')
        self.tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=14)
        
        self.tree.heading('selected', text='✓')
        self.tree.heading('word', text='词汇')
        self.tree.heading('reading', text='读音')
        self.tree.heading('pos', text='词性')
        self.tree.heading('sentence', text='例句')
        
        self.tree.column('selected', width=40, anchor='center')
        self.tree.column('word', width=100)
        self.tree.column('reading', width=100)
        self.tree.column('pos', width=70)
        self.tree.column('sentence', width=500)
        
        self.tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # 双击切换选中
        self.tree.bind('<Double-1>', self._toggle_item)
        
        # === 底部操作区域 ===
        action_card = ttk.Frame(container, style='Card.TFrame', padding=15)
        action_card.pack(fill=tk.X)
        
        # 按钮行
        btn_row = ttk.Frame(action_card, style='Card.TFrame')
        btn_row.pack(fill=tk.X)
        
        ttk.Button(btn_row, text="生成 Anki 卡组", style='Primary.TButton',
                   command=self._generate_cards).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(btn_row, text="停止", command=self._stop).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(btn_row, text="标记已学", command=self._mark_learned).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(btn_row, text="标记忽略", command=self._mark_ignored).pack(side=tk.LEFT)
        
        # 进度信息
        progress_frame = ttk.Frame(action_card, style='Card.TFrame')
        progress_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.progress_var = tk.StringVar(value="就绪")
        ttk.Label(progress_frame, textvariable=self.progress_var, 
                  style='Card.TLabel').pack(side=tk.LEFT)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate', length=300)
        self.progress_bar.pack(side=tk.RIGHT)
    
    def _select_file(self):
        path = filedialog.askopenfilename(filetypes=[("EPUB", "*.epub")])
        if path:
            self.file_path.set(path)
    
    def _log(self, msg: str):
        self.progress_var.set(msg)
        self.root.update_idletasks()
    
    def _extract_words(self):
        """提取词汇"""
        if not self.file_path.get():
            messagebox.showerror("错误", "请先选择EPUB文件")
            return
        
        def run():
            try:
                self._log("初始化分词器...")
                
                # 初始化 MeCab
                if not self.tokenizer:
                    self.tokenizer = MeCabTokenizer()
                
                # 读取 EPUB
                self._log("读取EPUB...")
                from epub2anki.epub.reader import EPUBReader
                reader = EPUBReader(self.file_path.get())
                contents = reader.get_contents()
                
                self._log(f"发现 {len(contents)} 个章节，正在分词...")
                
                # 收集所有文本，记录每段文本的章节
                all_text = ""
                chapter_ranges = []  # [(start, end, chapter_title), ...]
                
                for content in contents:
                    text = content.clean_text
                    start = len(all_text)
                    all_text += text + "\n"
                    end = len(all_text)
                    chapter_ranges.append((start, end, content.chapter.title))
                
                self._log(f"共 {len(all_text)} 字符")
                
                # 直接对全文分词
                self._log("MeCab 分词中...")
                vocabulary = self.tokenizer.extract_vocabulary(all_text, self.user_vocab)
                
                self._log(f"发现 {len(vocabulary)} 个新词汇，正在查找例句...")
                
                # 为每个词汇找例句
                self.candidates = []
                for i, word in enumerate(vocabulary):
                    # 滑动窗口找例句
                    sentence = SentenceFinder.find_sentence(
                        word['surface'], 
                        word['base_form'], 
                        all_text
                    )
                    
                    # 根据词汇在文本中的位置确定章节
                    chapter = ""
                    word_pos = all_text.find(word['surface'])
                    if word_pos >= 0:
                        for start, end, title in chapter_ranges:
                            if start <= word_pos < end:
                                chapter = title
                                break
                    
                    self.candidates.append(WordCandidate(
                        surface=word['surface'],
                        base_form=word['base_form'],
                        reading=word['reading'],
                        pos=word['pos'],
                        sentence=sentence,
                        chapter=chapter
                    ))
                    
                    # 每100个词更新一次进度
                    if (i + 1) % 100 == 0:
                        self._log(f"查找例句: {i+1}/{len(vocabulary)}")
                
                self._log(f"提取完成：{len(self.candidates)} 个新词汇")
                
                # 更新列表
                self.root.after(0, self._update_word_list)
                
            except Exception as e:
                self._log(f"错误: {e}")
                import traceback
                traceback.print_exc()
        
        threading.Thread(target=run, daemon=True).start()
    
    def _update_word_list(self):
        """更新词汇列表"""
        self.tree.delete(*self.tree.get_children())
        
        for i, word in enumerate(self.candidates):
            self.tree.insert('', 'end', iid=str(i), values=(
                '✓' if word.selected else '',
                word.base_form,
                word.reading,
                word.pos,
                word.sentence[:50] + '...' if len(word.sentence) > 50 else word.sentence
            ))
        
        self.word_count_var.set(f"{len(self.candidates)} 个词汇")
    
    def _toggle_item(self, event):
        """双击切换选中状态"""
        item = self.tree.selection()
        if item:
            idx = int(item[0])
            self.candidates[idx].selected = not self.candidates[idx].selected
            self.tree.set(item[0], 'selected', '✓' if self.candidates[idx].selected else '')
    
    def _select_all(self):
        for i, word in enumerate(self.candidates):
            word.selected = True
            self.tree.set(str(i), 'selected', '✓')
    
    def _deselect_all(self):
        for i, word in enumerate(self.candidates):
            word.selected = False
            self.tree.set(str(i), 'selected', '')
    
    def _toggle_selection(self):
        for i, word in enumerate(self.candidates):
            word.selected = not word.selected
            self.tree.set(str(i), 'selected', '✓' if word.selected else '')
    
    def _mark_learned(self):
        """将选中的词标记为已学习"""
        selected = self.tree.selection()
        if not selected:
            messagebox.showinfo("提示", "请先选择词汇（点击选中）")
            return
        
        words = [self.candidates[int(i)].base_form for i in selected]
        self.user_vocab.add_learned(words)
        
        # 从列表中移除
        for i in sorted([int(x) for x in selected], reverse=True):
            del self.candidates[i]
        
        self._update_word_list()
        self.stats_var.set(self.user_vocab.get_stats())
        messagebox.showinfo("完成", f"已将 {len(words)} 个词标记为已学习")
    
    def _mark_ignored(self):
        """将选中的词标记为忽略"""
        selected = self.tree.selection()
        if not selected:
            messagebox.showinfo("提示", "请先选择词汇（点击选中）")
            return
        
        words = [self.candidates[int(i)].base_form for i in selected]
        self.user_vocab.add_ignored(words)
        
        # 从列表中移除
        for i in sorted([int(x) for x in selected], reverse=True):
            del self.candidates[i]
        
        self._update_word_list()
        self.stats_var.set(self.user_vocab.get_stats())
        messagebox.showinfo("完成", f"已将 {len(words)} 个词标记为忽略")
    
    def _stop(self):
        self.stop_flag.set()
        self._log("正在停止...")
    
    def _generate_cards(self):
        """生成 Anki 卡组"""
        if not self.api_key.get():
            messagebox.showerror("错误", "请输入API Key")
            return
        
        selected_words = [w for w in self.candidates if w.selected]
        if not selected_words:
            messagebox.showerror("错误", "请先选择要学习的词汇")
            return
        
        def run():
            try:
                self.stop_flag.clear()
                
                # 初始化 LLM
                llm = LLMProcessor(
                    api_key=self.api_key.get(),
                    base_url=self.base_url.get(),
                    model=self.model.get(),
                    output_lang=self.output_lang.get()
                )
                
                results = []
                total = len(selected_words)
                
                for i, word in enumerate(selected_words):
                    if self.stop_flag.is_set():
                        self._log("已停止")
                        break
                    
                    self.progress_var.set(f"生成: {i+1}/{total} - {word.base_form}")
                    self.progress_bar['value'] = int((i + 1) / total * 100)
                    self.root.update_idletasks()
                    
                    result = llm.generate_card(word)
                    if result:
                        results.append(result)
                
                if results:
                    # 生成 Anki
                    self._log("生成 Anki 卡组...")
                    
                    from epub2anki.anki import AnkiGenerator
                    
                    book_name = Path(self.file_path.get()).stem
                    output_file = CACHE_DIR / f"{book_name}.apkg"
                    
                    generator = AnkiGenerator(production_ratio=0.15)
                    
                    # 转换格式
                    for r in results:
                        r['headword'] = r.get('word', '')
                        r['reading_hiragana'] = r.get('reading', '')
                        r['meaning_in_context'] = r.get('meaning', '')
                        r['sentence_full'] = r.get('sentence', '')
                        r['sentence_ruby'] = r.get('sentence', '')
                        r['sentence_meaning_zh'] = r.get('sentence_meaning', '')
                        r['cloze_sentence'] = ''
                        r['cloze_ruby'] = ''
                        r['pitch_accent'] = ''
                        r['everyday_score'] = 1
                    
                    generator.generate(results, book_name, str(output_file))
                    
                    # 更新用户词典
                    learned_words = [w.base_form for w in selected_words]
                    self.user_vocab.add_learned(learned_words)
                    
                    self._log(f"完成！生成 {len(results)} 张卡片")
                    self.progress_bar['value'] = 100
                    
                    # 打开文件夹
                    if output_file.exists():
                        messagebox.showinfo("完成", 
                            f"生成成功！\n{len(results)} 张卡片\n\n文件: {output_file}")
                        if os.name == 'nt':
                            os.startfile(CACHE_DIR)
                        else:
                            os.system(f'open "{CACHE_DIR}"')
                
            except Exception as e:
                self._log(f"错误: {e}")
                import traceback
                traceback.print_exc()
        
        threading.Thread(target=run, daemon=True).start()
    
    def run(self):
        self.root.mainloop()


def main():
    app = Epub2AnkiApp()
    app.run()


if __name__ == "__main__":
    main()

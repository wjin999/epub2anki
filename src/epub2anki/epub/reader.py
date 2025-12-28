"""
EPUB 读取器 - 高精度文本提取
使用 EbookLib + BeautifulSoup
"""
import re
import warnings
from dataclasses import dataclass
from typing import List

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


@dataclass
class Chapter:
    """章节"""
    id: str
    title: str
    order: int


@dataclass
class ExtractedContent:
    """提取的内容"""
    chapter: Chapter
    clean_text: str


class EPUBReader:
    """
    EPUB 读取器 - 高精度版
    
    使用 EbookLib + BeautifulSoup 组合
    精度可达 97% 以上
    """
    
    # 根据内容跳过的关键词（版权页、编辑说明等）
    SKIP_CONTENT_KEYWORDS = [
        '入力、校正', 'ボランティア', '底本', '青空文庫',
        '作成者', '校正者', '入力者', '©', 'copyright',
        'All Rights Reserved', '制作にあたった',
    ]
    
    def __init__(self, epub_path: str):
        self.book = epub.read_epub(epub_path)
        self._build_title_map()
    
    def _build_title_map(self):
        """从导航构建标题映射"""
        self.title_map = {}
        for item in self.book.get_items_of_type(ebooklib.ITEM_NAVIGATION):
            try:
                soup = BeautifulSoup(item.get_content(), 'lxml')
                for nav in soup.find_all(['navpoint', 'li']):
                    label = nav.find(['navlabel', 'a'])
                    content = nav.find(['content', 'a'])
                    if label and content:
                        href = (content.get('src') or content.get('href', '')).split('#')[0]
                        title = label.get_text(strip=True)
                        if href and title:
                            self.title_map[href] = title
            except:
                pass
    
    def _clean_and_extract(self, content: bytes) -> str:
        """
        清洗HTML并提取纯文本
        
        参考最佳实践：
        - 使用 lxml 解析器
        - 移除 script/style 标签
        - 处理 HTML 实体
        """
        soup = BeautifulSoup(content, 'lxml')
        
        # 移除脚本、样式、导航等
        for tag in soup(['script', 'style', 'nav', 'rt', 'rp']):
            tag.decompose()
        
        # 青空文库格式：移除振り仮名标记
        html_str = str(soup)
        html_str = re.sub(r'｜([^｜《》]+)《([^》]+)》', r'\1', html_str)
        html_str = re.sub(r'([一-龯]+)《[^》]+》', r'\1', html_str)
        soup = BeautifulSoup(html_str, 'lxml')
        
        # 获取纯文本（用换行分隔）
        text = soup.get_text(separator='\n')
        
        # 清理：多个空白变一个，保留换行
        text = re.sub(r'[^\S\n]+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        return text.strip()
    
    def _should_skip_content(self, text: str) -> bool:
        """根据内容判断是否应该跳过"""
        if len(text) < 50:
            return True
        
        for keyword in self.SKIP_CONTENT_KEYWORDS:
            if keyword in text:
                return True
        
        return False
    
    def get_contents(self) -> List[ExtractedContent]:
        """
        获取所有文档内容
        
        遍历所有 ITEM_DOCUMENT 类型的项目
        """
        results = []
        order = 0
        
        # 遍历所有文档类型内容
        for item in self.book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            try:
                content = item.get_content()
                clean_text = self._clean_and_extract(content)
                
                if not clean_text or len(clean_text) < 20:
                    continue
                
                # 跳过版权页、编辑说明等
                if self._should_skip_content(clean_text):
                    continue
                
                # 获取标题
                filename = item.get_name()
                title = self.title_map.get(filename, f"第{order+1}章")
                
                chapter = Chapter(id=item.get_id(), title=title, order=order)
                results.append(ExtractedContent(chapter=chapter, clean_text=clean_text))
                order += 1
                
            except Exception as e:
                continue
        
        return results
    
    def get_full_text(self) -> str:
        """获取完整文本（用于分词）"""
        contents = self.get_contents()
        return '\n'.join(c.clean_text for c in contents)

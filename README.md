# epub2anki

从日语 EPUB 电子书提取生词，生成 Anki 闪卡。

## 技术栈

| 组件 | 用途 |
|------|------|
| EbookLib + BeautifulSoup | EPUB 文本提取（精度 97%+） |
| MeCab + UniDic | 日语分词 |
| OpenAI API | 生成释义和例句翻译 |
| genanki | 生成 Anki 卡组 |
| Tkinter | 桌面 GUI |

## 安装

```bash
pip install -r requirements.txt
python -m unidic download
```

## 运行

```bash
python app.py
```

## 流程

```
EPUB → 文本提取 → MeCab分词 → 过滤 → 用户筛选 → LLM生成 → Anki卡组
```

## 用户词典

已学习和已忽略的词汇保存在 `~/.epub2anki/vocabulary.json`，下次提取时自动跳过。

# epub2anki

从日语 EPUB 电子书提取生词，生成 Anki 闪卡。

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

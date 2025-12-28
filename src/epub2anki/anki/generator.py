"""
Anki 卡组生成器 - 简化版
"""
import random
from typing import List
import genanki


# 卡片模板
RECOGNITION_TEMPLATE = genanki.Model(
    1607392319,
    'Japanese Recognition',
    fields=[
        {'name': 'Word'},
        {'name': 'Reading'},
        {'name': 'Meaning'},
        {'name': 'Sentence'},
        {'name': 'SentenceMeaning'},
        {'name': 'Notes'},
    ],
    templates=[{
        'name': 'Recognition',
        'qfmt': '''
<div class="word">{{Word}}</div>
<div class="reading">{{Reading}}</div>
<div class="sentence">{{Sentence}}</div>
''',
        'afmt': '''
{{FrontSide}}
<hr>
<div class="meaning">{{Meaning}}</div>
<div class="sentence-meaning">{{SentenceMeaning}}</div>
<div class="notes">{{Notes}}</div>
''',
    }],
    css='''
.card { font-family: "Hiragino Kaku Gothic Pro", "MS Gothic", sans-serif; font-size: 20px; text-align: center; }
.word { font-size: 48px; color: #333; margin: 20px; }
.reading { font-size: 24px; color: #666; }
.sentence { font-size: 18px; margin: 20px; padding: 15px; background: #f5f5f5; border-radius: 8px; }
.meaning { font-size: 24px; color: #2196F3; margin: 15px; }
.sentence-meaning { font-size: 16px; color: #666; }
.notes { font-size: 14px; color: #999; margin-top: 10px; }
'''
)


class AnkiGenerator:
    """Anki 卡组生成器"""
    
    def __init__(self, production_ratio: float = 0.0):
        self.production_ratio = production_ratio
    
    def generate(self, words: List[dict], deck_name: str, output_path: str):
        """生成 Anki 卡组"""
        deck_id = random.randrange(1 << 30, 1 << 31)
        deck = genanki.Deck(deck_id, deck_name)
        
        for word in words:
            note = genanki.Note(
                model=RECOGNITION_TEMPLATE,
                fields=[
                    word.get('word', word.get('headword', '')),
                    word.get('reading', word.get('reading_hiragana', '')),
                    word.get('meaning', word.get('meaning_in_context', '')),
                    word.get('sentence', word.get('sentence_full', '')),
                    word.get('sentence_meaning', word.get('sentence_meaning_zh', '')),
                    word.get('notes', ''),
                ]
            )
            deck.add_note(note)
        
        genanki.Package(deck).write_to_file(output_path)

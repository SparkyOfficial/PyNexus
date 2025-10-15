"""
Модуль для обчислювальної лінгвістики в PyNexus.
Містить функції для обробки природних мов, аналізу тексту, машинного перекладу та інших задач лінгвістики.
"""

import re
import math
import random
from typing import List, Dict, Tuple, Set, Optional, Union
from collections import Counter, defaultdict
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import entropy

# Константи для лінгвістики
VOWELS = set('aeiouAEIOU')
CONSONANTS = set('bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ')
STOP_WORDS_EN = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
    'had', 'what', 'said', 'each', 'which', 'their', 'time', 'if', 'will',
    'way', 'about', 'many', 'then', 'them', 'would', 'been', 'more',
    'these', 'so', 'no', 'may', 'my', 'out', 'up', 'some', 'there',
    'only', 'into', 'its', 'most', 'than', 'two', 'first', 'could',
    'other', 'like', 'new', 'do', 'me', 'any', 'now', 'very', 'just',
    'our', 'man', 'did', 'get', 'can', 'know', 'us', 'go', 'day',
    'good', 'one', 'made', 'see', 'make', 'well', 'back', 'year',
    'still', 'come', 'work', 'life', 'even', 'here', 'take', 'last',
    'long', 'great', 'small', 'another', 'found', 'house', 'home',
    'sound', 'place', 'live', 'old', 'too', 'high', 'own', 'such',
    'both', 'big', 'four', 'boy', 'often', 'does', 'tell', 'face',
    'set', 'ask', 'end', 'move', 'want', 'air', 'put', 'also', 'part',
    'need', 'play', 'little', 'since', 'around', 'should', 'mean',
    'used', 'hand', 'school', 'world', 'number', 'head', 'men', 'think',
    'under', 'water', 'room', 'young', 'might', 'next', 'white', 'children',
    'call', 'point', 'large', 'person', 'until', 'side', 'word', 'saw',
    'though', 'line', 'far', 'down', 'form', 'off', 'help', 'made',
    'turn', 'cause', 'much', 'before', 'right', 'boy', 'change', 'three',
    'kind', 'came', 'let', 'close', 'power', 'live', 'seen', 'name',
    'among', 'stood', 'stood', 'stood', 'stood', 'stood', 'stood',
    'stood', 'stood', 'stood', 'stood', 'stood', 'stood', 'stood',
    'stood', 'stood', 'stood', 'stood', 'stood', 'stood', 'stood'
}

STOP_WORDS_UA = {
    'і', 'в', 'не', 'на', 'що', 'як', 'з', 'за', 'до', 'то', 'та', 'це',
    'у', 'про', 'чи', 'або', 'але', 'бо', 'так', 'же', 'ну', 'від', 'по',
    'лиш', 'вже', 'ще', 'тільки', 'лише', 'всього', 'всі', 'кожен', 'один',
    'два', 'три', 'чотири', 'п\'ять', 'багато', 'декілька', 'кілька',
    'більшість', 'меншість', 'деякі', 'інші', 'інший', 'той', 'такий',
    'самий', 'також', 'разом', 'окремо', 'разом', 'окремо', 'через',
    'під', 'над', 'перед', 'після', 'між', 'серед', 'біля', 'коло',
    'поруч', 'далеко', 'близько', 'верх', 'низ', 'ліво', 'право',
    'початок', 'кінець', 'середина', 'перший', 'останній', 'новий',
    'старий', 'молодий', 'давній', 'теперішній', 'майбутній', 'минулий',
    'великий', 'малий', 'середній', 'високий', 'низький', 'широкий',
    'вузький', 'товстий', 'тонкий', 'довгий', 'короткий', 'гарячий',
    'холодний', 'теплий', 'прохолодний', 'світлий', 'темний', 'яскравий',
    'тьмяний', 'гучний', 'тихий', 'швидкий', 'повільний', 'легкий',
    'важкий', 'м\'який', 'твердий', 'гладкий', 'шорсткий', 'чистий',
    'брудний', 'сухий', 'мокрий', 'живий', 'мертвий', 'здоровий',
    'хворий', 'гарний', 'поганий', 'красивий', 'потворний', 'приємний',
    'неприємний', 'солодкий', 'гіркий', 'кислий', 'солоний', 'пріркий',
    'смачний', 'несмачний', 'дорогий', 'дешевий', 'багатий', 'бідний',
    'вільний', 'зайнятий', 'відкритий', 'закритий', 'повний', 'порожній',
    'легкий', 'важкий', 'швидкий', 'повільний', 'рано', 'пізно',
    'раніше', 'пізніше', 'завжди', 'ніколи', 'іноді', 'часто', 'рідко',
    'ніколи', 'завжди', 'весь', 'вся', 'все', 'усі', 'кожен', 'кожна',
    'кожне', 'кожні', 'будь-який', 'будь-яка', 'будь-яке', 'будь-які',
    'ніякий', 'ніяка', 'ніяке', 'ніякі', 'сам', 'сама', 'саме', 'самі',
    'один', 'одна', 'одне', 'одні', 'два', 'дві', 'двоє', 'три', 'троє',
    'чотири', 'четверо', 'п\'ять', 'п\'ятеро', 'багато', 'багато',
    'декілька', 'кілька', 'більше', 'менше', 'більшість', 'меншість'
}

def tokenize_text(text: str, language: str = 'en') -> List[str]:
    """
    Розбити текст на токени (слова).
    
    Параметри:
        text: Вхідний текст
        language: Мова тексту ('en' або 'ua'), за замовчуванням 'en'
    
    Повертає:
        Список токенів
    """
    if not text:
        return []
    
    # Видаляємо пунктуацію та розбиваємо на слова
    if language == 'ua':
        # Для української мови
        tokens = re.findall(r'\b[а-яА-ЯіїєґІЇЄҐ]+\b', text)
    else:
        # Для англійської мови
        tokens = re.findall(r'\b[a-zA-Z]+\b', text)
    
    # Перетворюємо в нижній регістр
    tokens = [token.lower() for token in tokens]
    
    return tokens

def remove_stop_words(tokens: List[str], language: str = 'en') -> List[str]:
    """
    Видалити стоп-слова зі списку токенів.
    
    Параметри:
        tokens: Список токенів
        language: Мова тексту ('en' або 'ua'), за замовчуванням 'en'
    
    Повертає:
        Список токенів без стоп-слів
    """
    if language == 'ua':
        stop_words = STOP_WORDS_UA
    else:
        stop_words = STOP_WORDS_EN
    
    return [token for token in tokens if token not in stop_words]

def calculate_term_frequency(tokens: List[str]) -> Dict[str, float]:
    """
    Обчислити частоту термінів (TF).
    
    Параметри:
        tokens: Список токенів
    
    Повертає:
        Словник {термін: частота}
    """
    if not tokens:
        return {}
    
    total_tokens = len(tokens)
    term_counts = Counter(tokens)
    
    # Обчислюємо частоту для кожного терміна
    tf_dict = {term: count / total_tokens for term, count in term_counts.items()}
    
    return tf_dict

def calculate_inverse_document_frequency(documents: List[List[str]]) -> Dict[str, float]:
    """
    Обчислити обернену частоту документів (IDF).
    
    Параметри:
        documents: Список документів (кожен документ - список токенів)
    
    Повертає:
        Словник {термін: IDF}
    """
    if not documents:
        return {}
    
    total_documents = len(documents)
    idf_dict = {}
    
    # Збираємо всі унікальні терміни
    all_terms = set()
    for doc in documents:
        all_terms.update(doc)
    
    # Обчислюємо IDF для кожного терміна
    for term in all_terms:
        # Підраховуємо кількість документів, що містять термін
        documents_with_term = sum(1 for doc in documents if term in doc)
        
        # Обчислюємо IDF
        if documents_with_term > 0:
            idf_dict[term] = math.log(total_documents / documents_with_term)
        else:
            idf_dict[term] = 0
    
    return idf_dict

def calculate_tfidf(documents: List[List[str]]) -> List[Dict[str, float]]:
    """
    Обчислити TF-IDF для набору документів.
    
    Параметри:
        documents: Список документів (кожен документ - список токенів)
    
    Повертає:
        Список словників TF-IDF для кожного документа
    """
    if not documents:
        return []
    
    # Обчислюємо IDF для всіх термінів
    idf_dict = calculate_inverse_document_frequency(documents)
    
    # Обчислюємо TF-IDF для кожного документа
    tfidf_documents = []
    
    for doc in documents:
        # Обчислюємо TF для поточного документа
        tf_dict = calculate_term_frequency(doc)
        
        # Обчислюємо TF-IDF
        tfidf_dict = {}
        for term, tf in tf_dict.items():
            tfidf_dict[term] = tf * idf_dict.get(term, 0)
        
        tfidf_documents.append(tfidf_dict)
    
    return tfidf_documents

def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """
    Обчислити косинусну подібність між двома векторами.
    
    Параметри:
        vec1: Перший вектор (словник {термін: значення})
        vec2: Другий вектор (словник {термін: значення})
    
    Повертає:
        Косинусна подібність (від 0 до 1)
    """
    # Знаходимо всі унікальні терміни
    all_terms = set(vec1.keys()) | set(vec2.keys())
    
    if not all_terms:
        return 0.0
    
    # Створюємо вектори
    vector1 = [vec1.get(term, 0) for term in all_terms]
    vector2 = [vec2.get(term, 0) for term in all_terms]
    
    # Обчислюємо косинусну подібність
    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    magnitude1 = math.sqrt(sum(a * a for a in vector1))
    magnitude2 = math.sqrt(sum(b * b for b in vector2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)

def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """
    Обчислити подібність Жаккара між двома множинами.
    
    Параметри:
        set1: Перша множина
        set2: Друга множина
    
    Повертає:
        Подібність Жаккара (від 0 до 1)
    """
    if not set1 and not set2:
        return 1.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    if union == 0:
        return 0.0
    
    return intersection / union

def levenshtein_distance(str1: str, str2: str) -> int:
    """
    Обчислити відстань Левенштейна між двома рядками.
    
    Параметри:
        str1: Перший рядок
        str2: Другий рядок
    
    Повертає:
        Відстань Левенштейна
    """
    if not str1:
        return len(str2)
    if not str2:
        return len(str1)
    
    # Створюємо матрицю
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Ініціалізуємо перший рядок та стовпець
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Заповнюємо матрицю
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # видалення
                    dp[i][j-1],    # вставка
                    dp[i-1][j-1]   # заміна
                )
    
    return dp[m][n]

def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """
    Згенерувати n-грами зі списку токенів.
    
    Параметри:
        tokens: Список токенів
        n: Розмір n-грами
    
    Повертає:
        Список n-грам
    """
    if n <= 0:
        raise ValueError("Розмір n-грами повинен бути додатнім")
    if not tokens:
        return []
    
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def frequency_analysis(text: str, language: str = 'en') -> Dict[str, int]:
    """
    Провести частотний аналіз тексту.
    
    Параметри:
        text: Вхідний текст
        language: Мова тексту ('en' або 'ua'), за замовчуванням 'en'
    
    Повертає:
        Словник частот символів
    """
    # Вибираємо алфавіт залежно від мови
    if language == 'ua':
        alphabet = 'абвгґдеєжзиіїйклмнопрстуфхцчшщьюя'
    else:
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
    
    # Перетворюємо текст в нижній регістр
    text = text.lower()
    
    # Підраховуємо частоти символів
    char_counts = {}
    for char in text:
        if char in alphabet:
            char_counts[char] = char_counts.get(char, 0) + 1
    
    return char_counts

def text_statistics(text: str) -> Dict[str, Union[int, float]]:
    """
    Обчислити статистику тексту.
    
    Параметри:
        text: Вхідний текст
    
    Повертає:
        Словник зі статистикою тексту
    """
    if not text:
        return {
            'character_count': 0,
            'word_count': 0,
            'sentence_count': 0,
            'paragraph_count': 0,
            'avg_word_length': 0.0,
            'avg_sentence_length': 0.0
        }
    
    # Кількість символів
    character_count = len(text)
    
    # Кількість слів
    words = re.findall(r'\b\w+\b', text)
    word_count = len(words)
    
    # Кількість речень
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentences if s.strip()])
    
    # Кількість абзаців
    paragraphs = text.split('\n\n')
    paragraph_count = len([p for p in paragraphs if p.strip()])
    
    # Середня довжина слова
    if word_count > 0:
        total_word_length = sum(len(word) for word in words)
        avg_word_length = total_word_length / word_count
    else:
        avg_word_length = 0.0
    
    # Середня довжина речення
    if sentence_count > 0:
        total_sentence_length = sum(len(re.findall(r'\b\w+\b', sentence)) 
                                  for sentence in sentences if sentence.strip())
        avg_sentence_length = total_sentence_length / sentence_count
    else:
        avg_sentence_length = 0.0
    
    return {
        'character_count': character_count,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'paragraph_count': paragraph_count,
        'avg_word_length': avg_word_length,
        'avg_sentence_length': avg_sentence_length
    }

def syllable_count(word: str) -> int:
    """
    Підрахувати кількість складів у слові.
    
    Параметри:
        word: Слово
    
    Повертає:
        Кількість складів
    """
    if not word:
        return 0
    
    word = word.lower()
    
    # Проста евристика для підрахунку складів
    vowels = 'aeiouаеєиіїоуюя'
    syllable_count = 0
    prev_was_vowel = False
    
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_was_vowel:
            syllable_count += 1
        prev_was_vowel = is_vowel
    
    # Коригуємо для закінчень, які не змінюють кількість складів
    if word.endswith(('e', 'є')) and syllable_count > 1:
        syllable_count -= 1
    
    return max(1, syllable_count)

def flesch_reading_ease(text: str, language: str = 'en') -> float:
    """
    Обчислити індекс зручності читання Флеша.
    
    Параметри:
        text: Вхідний текст
        language: Мова тексту ('en' або 'ua'), за замовчуванням 'en'
    
    Повертає:
        Індекс зручності читання (0-100, де більше - легше)
    """
    stats = text_statistics(text)
    
    if stats['word_count'] == 0 or stats['sentence_count'] == 0:
        return 0.0
    
    # Підраховуємо кількість складів
    words = re.findall(r'\b\w+\b', text)
    total_syllables = sum(syllable_count(word) for word in words)
    
    # Середня кількість складів на слово
    avg_syllables_per_word = total_syllables / stats['word_count']
    
    # Середня кількість слів на речення
    avg_words_per_sentence = stats['word_count'] / stats['sentence_count']
    
    # Формула залежить від мови
    if language == 'ua':
        # Для української мови (приблизна формула)
        score = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
    else:
        # Для англійської мови
        score = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
    
    return max(0, min(100, score))

def sentiment_analysis(text: str, language: str = 'en') -> Dict[str, float]:
    """
    Провести аналіз настрою тексту.
    
    Параметри:
        text: Вхідний текст
        language: Мова тексту ('en' або 'ua'), за замовчуванням 'en'
    
    Повертає:
        Словник з оцінками настрою
    """
    # Проста реалізація на основі словника настроїв
    positive_words_en = {
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
        'awesome', 'brilliant', 'outstanding', 'superb', 'marvelous',
        'terrific', 'splendid', 'fabulous', 'incredible', 'perfect',
        'love', 'like', 'enjoy', 'happy', 'pleased', 'satisfied',
        'delighted', 'thrilled', 'excited', 'enthusiastic', 'optimistic'
    }
    
    negative_words_en = {
        'bad', 'terrible', 'awful', 'horrible', 'dreadful', 'atrocious',
        'abysmal', 'appalling', 'disgusting', 'revolting', 'nauseating',
        'hateful', 'despicable', 'loathsome', 'repugnant', 'vile',
        'hate', 'dislike', 'disgust', 'sad', 'unhappy', 'depressed',
        'disappointed', 'frustrated', 'angry', 'annoyed', 'irritated',
        'upset', 'distressed', 'worried', 'anxious', 'nervous'
    }
    
    positive_words_ua = {
        'хороший', 'чудовий', 'відмінний', 'чудесний', 'фантастичний',
        'дивовижний', 'надзвичайний', 'видатний', 'чудовий', 'прекрасний',
        'чудний', 'неймовірний', 'ідеальний', 'люблю', 'подобається',
        'насолоджуюся', 'щасливий', 'задоволений', 'радісний', 'в захваті',
        'схвильований', 'ентузіаст', 'оптимістичний', 'приємний', 'смачний'
    }
    
    negative_words_ua = {
        'поганий', 'жахливий', 'страшний', 'відразливий', 'мерзенний',
        'огидний', 'ненависний', 'відразливий', 'гидкий', 'мерзотний',
        'неприємний', 'ненавиджу', 'не подобається', 'відчуваю огиду',
        'сумний', 'нещасний', 'пригнічений', 'розчарований', 'сварливий',
        'засмучений', 'роздратований', 'сумний', 'стурбований', 'тривожний'
    }
    
    # Вибираємо словники залежно від мови
    if language == 'ua':
        positive_words = positive_words_ua
        negative_words = negative_words_ua
    else:
        positive_words = positive_words_en
        negative_words = negative_words_en
    
    # Токенізуємо текст
    tokens = tokenize_text(text, language)
    
    # Підраховуємо позитивні та негативні слова
    positive_count = sum(1 for token in tokens if token in positive_words)
    negative_count = sum(1 for token in tokens if token in negative_words)
    
    total_emotional_words = positive_count + negative_count
    
    if total_emotional_words == 0:
        return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
    
    positive_score = positive_count / total_emotional_words
    negative_score = negative_count / total_emotional_words
    neutral_score = 1.0 - (positive_score + negative_score)
    
    return {
        'positive': positive_score,
        'negative': negative_score,
        'neutral': neutral_score
    }

def keyword_extraction(text: str, language: str = 'en', top_k: int = 10) -> List[Tuple[str, float]]:
    """
    Видобути ключові слова з тексту.
    
    Параметри:
        text: Вхідний текст
        language: Мова тексту ('en' або 'ua'), за замовчуванням 'en'
        top_k: Кількість ключових слів для повернення, за замовчуванням 10
    
    Повертає:
        Список кортежів (ключове слово, оцінка важливості)
    """
    # Токенізуємо текст
    tokens = tokenize_text(text, language)
    
    # Видаляємо стоп-слова
    filtered_tokens = remove_stop_words(tokens, language)
    
    # Підраховуємо частоти
    word_freq = Counter(filtered_tokens)
    
    # Сортуємо за частотою
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Повертаємо топ-k слів
    return sorted_words[:top_k]

def text_summarization(text: str, language: str = 'en', sentences_count: int = 3) -> str:
    """
    Створити короткий зміст тексту.
    
    Параметри:
        text: Вхідний текст
        language: Мова тексту ('en' або 'ua'), за замовчуванням 'en'
        sentences_count: Кількість речень у змісті, за замовчуванням 3
    
    Повертає:
        Зміст тексту
    """
    # Розбиваємо текст на речення
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= sentences_count:
        return '. '.join(sentences) + '.'
    
    # Токенізуємо кожне речення
    sentence_tokens = [tokenize_text(sentence, language) for sentence in sentences]
    
    # Видаляємо стоп-слова
    filtered_sentences = [remove_stop_words(tokens, language) for tokens in sentence_tokens]
    
    # Обчислюємо TF-IDF для всіх речень
    all_documents = filtered_sentences
    tfidf_documents = calculate_tfidf(all_documents)
    
    # Обчислюємо оцінку важливості для кожного речення
    sentence_scores = []
    for i, tfidf_dict in enumerate(tfidf_documents):
        # Сума TF-IDF значень для всіх термінів у реченні
        score = sum(tfidf_dict.values())
        sentence_scores.append((i, score))
    
    # Сортуємо речення за оцінкою
    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Вибираємо топ речень
    top_sentence_indices = sorted([idx for idx, score in sentence_scores[:sentences_count]])
    
    # Формуємо зміст
    summary_sentences = [sentences[i] for i in top_sentence_indices]
    
    return '. '.join(summary_sentences) + '.'

def language_detection(text: str) -> str:
    """
    Визначити мову тексту.
    
    Параметри:
        text: Вхідний текст
    
    Повертає:
        Код мови ('en', 'ua', або 'unknown')
    """
    if not text:
        return 'unknown'
    
    # Підраховуємо частоти символів для кожної мови
    en_chars = len(re.findall(r'[a-zA-Z]', text))
    ua_chars = len(re.findall(r'[а-яА-ЯіїєґІЇЄҐ]', text))
    
    total_chars = en_chars + ua_chars
    
    if total_chars == 0:
        return 'unknown'
    
    en_ratio = en_chars / total_chars
    ua_ratio = ua_chars / total_chars
    
    # Визначаємо мову на основі частот
    if en_ratio > 0.7:
        return 'en'
    elif ua_ratio > 0.7:
        return 'ua'
    else:
        return 'unknown'

def word_cloud_data(text: str, language: str = 'en', max_words: int = 100) -> Dict[str, int]:
    """
    Підготувати дані для хмари слів.
    
    Параметри:
        text: Вхідний текст
        language: Мова тексту ('en' або 'ua'), за замовчуванням 'en'
        max_words: Максимальна кількість слів, за замовчуванням 100
    
    Повертає:
        Словник {слово: частота}
    """
    # Токенізуємо текст
    tokens = tokenize_text(text, language)
    
    # Видаляємо стоп-слова
    filtered_tokens = remove_stop_words(tokens, language)
    
    # Підраховуємо частоти
    word_freq = Counter(filtered_tokens)
    
    # Повертаємо топ слів
    return dict(word_freq.most_common(max_words))

def text_similarity(doc1: str, doc2: str, language: str = 'en') -> Dict[str, float]:
    """
    Обчислити подібність між двома текстами різними методами.
    
    Параметри:
        doc1: Перший документ
        doc2: Другий документ
        language: Мова текстів ('en' або 'ua'), за замовчуванням 'en'
    
    Повертає:
        Словник з різними метриками подібності
    """
    # Токенізуємо документи
    tokens1 = tokenize_text(doc1, language)
    tokens2 = tokenize_text(doc2, language)
    
    # Видаляємо стоп-слова
    filtered_tokens1 = remove_stop_words(tokens1, language)
    filtered_tokens2 = remove_stop_words(tokens2, language)
    
    # Подібність Жаккара
    jaccard_sim = jaccard_similarity(set(filtered_tokens1), set(filtered_tokens2))
    
    # Відстань Левенштейна
    levenshtein_dist = levenshtein_distance(' '.join(filtered_tokens1), ' '.join(filtered_tokens2))
    # Нормалізуємо відстань до подібності
    max_len = max(len(' '.join(filtered_tokens1)), len(' '.join(filtered_tokens2)))
    levenshtein_sim = 1.0 - (levenshtein_dist / max_len) if max_len > 0 else 1.0
    
    # TF-IDF подібність
    tfidf_docs = calculate_tfidf([filtered_tokens1, filtered_tokens2])
    if len(tfidf_docs) >= 2:
        tfidf_sim = cosine_similarity(tfidf_docs[0], tfidf_docs[1])
    else:
        tfidf_sim = 0.0
    
    return {
        'jaccard_similarity': jaccard_sim,
        'levenshtein_similarity': levenshtein_sim,
        'tfidf_similarity': tfidf_sim
    }

def generate_ngram_model(text: str, n: int = 2, language: str = 'en') -> Dict[Tuple[str, ...], float]:
    """
    Згенерувати n-грамну модель для тексту.
    
    Параметри:
        text: Вхідний текст
        n: Розмір n-грами, за замовчуванням 2
        language: Мова тексту ('en' або 'ua'), за замовчуванням 'en'
    
    Повертає:
        Словник {n-грама: ймовірність}
    """
    # Токенізуємо текст
    tokens = tokenize_text(text, language)
    
    if len(tokens) < n:
        return {}
    
    # Генеруємо n-грами
    ngrams_list = ngrams(tokens, n)
    
    # Підраховуємо частоти
    ngram_counts = Counter(ngrams_list)
    total_ngrams = len(ngrams_list)
    
    # Обчислюємо ймовірності
    ngram_model = {ngram: count / total_ngrams for ngram, count in ngram_counts.items()}
    
    return ngram_model

def text_generation(seed_text: str, ngram_model: Dict[Tuple[str, ...], float], 
                   length: int = 50, language: str = 'en') -> str:
    """
    Згенерувати текст на основі n-грамної моделі.
    
    Параметри:
        seed_text: Початковий текст
        ngram_model: N-грамна модель
        length: Довжина генерованого тексту, за замовчуванням 50
        language: Мова тексту ('en' або 'ua'), за замовчуванням 'en'
    
    Повертає:
        Згенерований текст
    """
    if not ngram_model:
        return seed_text
    
    # Визначаємо розмір n-грами
    n = len(next(iter(ngram_model.keys())))
    
    # Токенізуємо початковий текст
    tokens = tokenize_text(seed_text, language)
    
    # Якщо токенів менше, ніж розмір n-грами, додаємо їх
    while len(tokens) < n - 1:
        tokens.append(random.choice(list(set(token for ngram in ngram_model.keys() for token in ngram))))
    
    generated_tokens = tokens[:]
    
    # Генеруємо нові токени
    for _ in range(length):
        # Отримуємо останні n-1 токенів
        context = tuple(generated_tokens[-(n-1):]) if n > 1 else ()
        
        # Знаходимо всі n-грами, що починаються з контексту
        possible_ngrams = {ngram: prob for ngram, prob in ngram_model.items() 
                          if ngram[:-1] == context}
        
        if not possible_ngrams:
            # Якщо немає відповідних n-грам, вибираємо випадкову
            next_token = random.choice(list(set(token for ngram in ngram_model.keys() for token in ngram)))
        else:
            # Вибираємо наступний токен на основі ймовірностей
            ngrams_list = list(possible_ngrams.keys())
            probs = list(possible_ngrams.values())
            chosen_ngram = random.choices(ngrams_list, weights=probs)[0]
            next_token = chosen_ngram[-1]
        
        generated_tokens.append(next_token)
    
    return ' '.join(generated_tokens)

def spelling_correction(word: str, dictionary: Set[str]) -> List[str]:
    """
    Запропонувати варіанти виправлення орфографії.
    
    Параметри:
        word: Слово з можливою помилкою
        dictionary: Словник правильних слів
    
    Повертає:
        Список можливих варіантів виправлення
    """
    if not word or not dictionary:
        return []
    
    word = word.lower()
    
    # Якщо слово вже в словнику, воно правильне
    if word in dictionary:
        return [word]
    
    # Знаходимо слова з відстанню Левенштейна <= 2
    candidates = []
    for dict_word in dictionary:
        distance = levenshtein_distance(word, dict_word)
        if distance <= 2:
            candidates.append((dict_word, distance))
    
    # Сортуємо за відстанню
    candidates.sort(key=lambda x: x[1])
    
    # Повертаємо топ-5 кандидатів
    return [word for word, distance in candidates[:5]]

def phonetic_similarity(word1: str, word2: str) -> float:
    """
    Обчислити фонетичну подібність між двома словами.
    
    Параметри:
        word1: Перше слово
        word2: Друге слово
    
    Повертає:
        Фонетична подібність (від 0 до 1)
    """
    # Проста реалізація на основі відстані Левенштейна для звуків
    # Це спрощена версія, в реальних системах використовують IPA та інші методи
    
    # Перетворюємо слова в фонетичні коди (спрощений підхід)
    def phonetic_code(word):
        # Замінюємо групи літер на звуки
        word = word.lower()
        word = re.sub(r'[aeiouy]', 'V', word)  # Голосні
        word = re.sub(r'[bcdfgjklmnpqsvwxz]', 'C', word)  # Приголосні 1
        word = re.sub(r'[hrt]', 'D', word)  # Приголосні 2
        return word
    
    code1 = phonetic_code(word1)
    code2 = phonetic_code(word2)
    
    # Обчислюємо відстань Левенштейна для фонетичних кодів
    distance = levenshtein_distance(code1, code2)
    max_len = max(len(code1), len(code2))
    
    if max_len == 0:
        return 1.0
    
    return 1.0 - (distance / max_len)

def text_clustering(documents: List[str], language: str = 'en', n_clusters: int = 3) -> List[int]:
    """
    Кластеризувати текстові документи.
    
    Параметри:
        documents: Список документів
        language: Мова текстів ('en' або 'ua'), за замовчуванням 'en'
        n_clusters: Кількість кластерів, за замовчуванням 3
    
    Повертає:
        Список номерів кластерів для кожного документа
    """
    if not documents or n_clusters <= 0:
        return []
    
    # Токенізуємо документи
    tokenized_docs = [tokenize_text(doc, language) for doc in documents]
    
    # Видаляємо стоп-слова
    filtered_docs = [remove_stop_words(tokens, language) for tokens in tokenized_docs]
    
    # Обчислюємо TF-IDF
    tfidf_docs = calculate_tfidf(filtered_docs)
    
    # Перетворюємо в матрицю
    all_terms = set()
    for tfidf_dict in tfidf_docs:
        all_terms.update(tfidf_dict.keys())
    
    all_terms = sorted(list(all_terms))
    tfidf_matrix = []
    for tfidf_dict in tfidf_docs:
        vector = [tfidf_dict.get(term, 0) for term in all_terms]
        tfidf_matrix.append(vector)
    
    # Проста кластеризація методом k-середніх
    if len(documents) <= n_clusters:
        return list(range(len(documents)))
    
    # Ініціалізуємо центроїди випадково
    centroids = random.sample(tfidf_matrix, min(n_clusters, len(tfidf_matrix)))
    
    # Ітераційний процес кластеризації
    max_iterations = 100
    for _ in range(max_iterations):
        # Призначаємо кожен документ до найближчого кластера
        clusters = []
        for doc_vector in tfidf_matrix:
            distances = []
            for centroid in centroids:
                # Обчислюємо евклідову відстань
                distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(doc_vector, centroid)))
                distances.append(distance)
            
            # Призначаємо до кластера з мінімальною відстанню
            cluster_id = distances.index(min(distances))
            clusters.append(cluster_id)
        
        # Оновлюємо центроїди
        new_centroids = []
        for i in range(n_clusters):
            cluster_docs = [tfidf_matrix[j] for j in range(len(tfidf_matrix)) if clusters[j] == i]
            if cluster_docs:
                # Обчислюємо середнє значення для кожного виміру
                centroid = []
                for dim in range(len(all_terms)):
                    dim_values = [doc[dim] for doc in cluster_docs]
                    centroid.append(sum(dim_values) / len(dim_values))
                new_centroids.append(centroid)
            else:
                # Якщо кластер порожній, залишаємо старий центроїд
                new_centroids.append(centroids[i] if i < len(centroids) else [0] * len(all_terms))
        
        # Перевіряємо збіжність
        if new_centroids == centroids:
            break
        
        centroids = new_centroids
    
    return clusters

def named_entity_recognition(text: str, language: str = 'en') -> List[Tuple[str, str]]:
    """
    Видобути іменовані сутності з тексту.
    
    Параметри:
        text: Вхідний текст
        language: Мова тексту ('en' або 'ua'), за замовчуванням 'en'
    
    Повертає:
        Список кортежів (сутність, тип)
    """
    # Проста реалізація на основі регулярних виразів
    entities = []
    
    # Видобуваємо числа
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    for number in numbers:
        entities.append((number, 'NUMBER'))
    
    # Видобуваємо дати
    dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text)
    for date in dates:
        entities.append((date, 'DATE'))
    
    # Видобуваємо email-адреси
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    for email in emails:
        entities.append((email, 'EMAIL'))
    
    # Видобуваємо URL
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for url in urls:
        entities.append((url, 'URL'))
    
    # Для більш складної NER потрібні спеціалізовані моделі
    # Це спрощена версія
    
    return entities

def text_normalization(text: str, language: str = 'en') -> str:
    """
    Нормалізувати текст (видалити зайві символи, стандартизувати формат).
    
    Параметри:
        text: Вхідний текст
        language: Мова тексту ('en' або 'ua'), за замовчуванням 'en'
    
    Повертає:
        Нормалізований текст
    """
    if not text:
        return ""
    
    # Видаляємо зайві пробіли
    text = re.sub(r'\s+', ' ', text)
    
    # Видаляємо початкові та кінцеві пробіли
    text = text.strip()
    
    # Стандартизуємо лапки
    text = re.sub(r'[""`´]', '"', text)
    
    # Стандартизуємо апострофи
    text = re.sub(r"['`´]", "'", text)
    
    # Видаляємо керівні символи
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    return text

def vocabulary_richness(text: str, language: str = 'en') -> Dict[str, float]:
    """
    Обчислити показники багатства словника.
    
    Параметри:
        text: Вхідний текст
        language: Мова тексту ('en' або 'ua'), за замовчуванням 'en'
    
    Повертає:
        Словник з показниками багатства словника
    """
    # Токенізуємо текст
    tokens = tokenize_text(text, language)
    
    if not tokens:
        return {
            'type_token_ratio': 0.0,
            'hapax_legomena_ratio': 0.0,
            'vocabulary_size': 0,
            'text_length': 0
        }
    
    # Підраховуємо частоти слів
    word_freq = Counter(tokens)
    
    # Тип-токенне співвідношення
    type_token_ratio = len(word_freq) / len(tokens)
    
    # Співвідношення гапаксів (слів, що зустрічаються лише один раз)
    hapax_legomena = sum(1 for freq in word_freq.values() if freq == 1)
    hapax_legomena_ratio = hapax_legomena / len(tokens)
    
    return {
        'type_token_ratio': type_token_ratio,
        'hapax_legomena_ratio': hapax_legomena_ratio,
        'vocabulary_size': len(word_freq),
        'text_length': len(tokens)
    }

def collocation_extraction(text: str, language: str = 'en', window_size: int = 2) -> List[Tuple[str, str, float]]:
    """
    Видобути колокації (часто спільно вжувані слова) з тексту.
    
    Параметри:
        text: Вхідний текст
        language: Мова тексту ('en' або 'ua'), за замовчуванням 'en'
        window_size: Розмір вікна для пошуку колокацій, за замовчуванням 2
    
    Повертає:
        Список кортежів (слово1, слово2, міцність зв'язку)
    """
    # Токенізуємо текст
    tokens = tokenize_text(text, language)
    
    if len(tokens) < 2:
        return []
    
    # Підраховуємо частоти окремих слів
    word_freq = Counter(tokens)
    total_words = len(tokens)
    
    # Підраховуємо частоти біграм
    bigrams = ngrams(tokens, 2)
    bigram_freq = Counter(bigrams)
    total_bigrams = len(bigrams)
    
    # Обчислюємо міцність зв'язку для кожної біграми (Pointwise Mutual Information)
    collocations = []
    for bigram, freq in bigram_freq.items():
        word1, word2 = bigram
        prob_bigram = freq / total_bigrams
        prob_word1 = word_freq[word1] / total_words
        prob_word2 = word_freq[word2] / total_words
        
        # Обчислюємо PMI
        if prob_word1 > 0 and prob_word2 > 0:
            pmi = math.log(prob_bigram / (prob_word1 * prob_word2))
            collocations.append((word1, word2, pmi))
    
    # Сортуємо за міцністю зв'язку
    collocations.sort(key=lambda x: x[2], reverse=True)
    
    return collocations

def readability_metrics(text: str, language: str = 'en') -> Dict[str, float]:
    """
    Обчислити різні метрики зручності читання.
    
    Параметри:
        text: Вхідний текст
        language: Мова тексту ('en' або 'ua'), за замовчуванням 'en'
    
    Повертає:
        Словник з метриками зручності читання
    """
    stats = text_statistics(text)
    
    if stats['word_count'] == 0 or stats['sentence_count'] == 0:
        return {
            'flesch_reading_ease': 0.0,
            'flesch_kincaid_grade': 0.0,
            'avg_words_per_sentence': 0.0,
            'avg_syllables_per_word': 0.0
        }
    
    # Підраховуємо кількість складів
    words = re.findall(r'\b\w+\b', text)
    total_syllables = sum(syllable_count(word) for word in words)
    
    # Середня кількість складів на слово
    avg_syllables_per_word = total_syllables / stats['word_count'] if stats['word_count'] > 0 else 0
    
    # Середня кількість слів на речення
    avg_words_per_sentence = stats['word_count'] / stats['sentence_count'] if stats['sentence_count'] > 0 else 0
    
    # Індекс зручності читання Флеша
    flesch_reading_ease = flesch_reading_ease(text, language)
    
    # Рівень шкільного класу за Флешем-Кінкайдом
    flesch_kincaid_grade = 0.39 * avg_words_per_sentence + 11.8 * avg_syllables_per_word - 15.59
    
    return {
        'flesch_reading_ease': flesch_reading_ease,
        'flesch_kincaid_grade': max(0, flesch_kincaid_grade),
        'avg_words_per_sentence': avg_words_per_sentence,
        'avg_syllables_per_word': avg_syllables_per_word
    }

def semantic_similarity(text1: str, text2: str, language: str = 'en') -> float:
    """
    Обчислити семантичну подібність між двома текстами.
    
    Параметри:
        text1: Перший текст
        text2: Другий текст
        language: Мова текстів ('en' або 'ua'), за замовчуванням 'en'
    
    Повертає:
        Семантична подібність (від 0 до 1)
    """
    # Токенізуємо текст
    tokens1 = tokenize_text(text1, language)
    tokens2 = tokenize_text(text2, language)
    
    # Видаляємо стоп-слова
    filtered_tokens1 = remove_stop_words(tokens1, language)
    filtered_tokens2 = remove_stop_words(tokens2, language)
    
    # Створюємо множини унікальних слів
    set1 = set(filtered_tokens1)
    set2 = set(filtered_tokens2)
    
    # Обчислюємо подібність Жаккара
    return jaccard_similarity(set1, set2)

def text_classification(text: str, categories: List[str], 
                       category_keywords: Dict[str, Set[str]],
                       language: str = 'en') -> Dict[str, float]:
    """
    Класифікувати текст за категоріями.
    
    Параметри:
        text: Вхідний текст
        categories: Список категорій
        category_keywords: Словник {категорія: множина ключових слів}
        language: Мова тексту ('en' або 'ua'), за замовчуванням 'en'
    
    Повертає:
        Словник {категорія: ймовірність}
    """
    # Токенізуємо текст
    tokens = tokenize_text(text, language)
    
    # Видаляємо стоп-слова
    filtered_tokens = remove_stop_words(tokens, language)
    
    # Створюємо множину унікальних слів
    text_words = set(filtered_tokens)
    
    # Обчислюємо оцінки для кожної категорії
    scores = {}
    total_score = 0
    
    for category in categories:
        keywords = category_keywords.get(category, set())
        # Кількість спільних слів
        common_words = len(text_words & keywords)
        # Оцінка пропорційна кількості спільних слів
        score = common_words / len(keywords) if keywords else 0
        scores[category] = score
        total_score += score
    
    # Нормалізуємо оцінки
    if total_score > 0:
        normalized_scores = {cat: score / total_score for cat, score in scores.items()}
    else:
        # Якщо немає спільних слів, рівномірно розподіляємо ймовірність
        normalized_scores = {cat: 1 / len(categories) for cat in categories}
    
    return normalized_scores

def discourse_analysis(text: str, language: str = 'en') -> Dict[str, Union[int, List[str]]]:
    """
    Провести дискурс-аналіз тексту.
    
    Параметри:
        text: Вхідний текст
        language: Мова тексту ('en' або 'ua'), за замовчуванням 'en'
    
    Повертає:
        Словник з результатами дискурс-аналізу
    """
    # Розбиваємо текст на речення
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Видобуваємо питання
    questions = [s for s in sentences if re.search(r'\b(?:what|where|when|why|how|who|which|whose|whom)\b', s.lower()) or s.strip().endswith('?')]
    
    # Видобуваємо окличні речення
    exclamations = [s for s in sentences if s.strip().endswith('!')]
    
    # Видобуваємо умовні речення
    conditionals = [s for s in sentences if re.search(r'\b(if|unless|provided that|assuming that)\b', s.lower())]
    
    # Видобуваємо причинно-наслідкові зв'язки
    causal = [s for s in sentences if re.search(r'\b(because|since|as|due to|owing to|therefore|thus|hence|consequently)\b', s.lower())]
    
    return {
        'total_sentences': len(sentences),
        'questions': questions,
        'exclamations': exclamations,
        'conditionals': conditionals,
        'causal_statements': causal,
        'question_count': len(questions),
        'exclamation_count': len(exclamations),
        'conditional_count': len(conditionals),
        'causal_count': len(causal)
    }

def morphological_analysis(word: str, language: str = 'en') -> Dict[str, str]:
    """
    Провести морфологічний аналіз слова.
    
    Параметри:
        word: Слово для аналізу
        language: Мова слова ('en' або 'ua'), за замовчуванням 'en'
    
    Повертає:
        Словник з морфологічними характеристиками
    """
    if not word:
        return {}
    
    word = word.lower()
    
    # Проста реалізація для англійської мови
    if language == 'en':
        features = {}
        
        # Визначаємо частину мови (спрощено)
        if word.endswith(('ing', 'ed', 'ly')):
            features['pos'] = 'verb/adverb'
        elif word.endswith(('tion', 'sion', 'ness', 'ment', 'ity', 'ty')):
            features['pos'] = 'noun'
        elif word.endswith(('ful', 'less', 'ous', 'ive', 'able', 'ible')):
            features['pos'] = 'adjective'
        else:
            features['pos'] = 'unknown'
        
        # Визначаємо число
        if word.endswith('s') and not word.endswith(('ss', 'us')):
            features['number'] = 'plural'
        else:
            features['number'] = 'singular'
        
        # Визначаємо час (для дієслів)
        if word.endswith('ing'):
            features['tense'] = 'present_participle'
        elif word.endswith('ed'):
            features['tense'] = 'past'
        else:
            features['tense'] = 'present'
        
        # Визначаємо ступінь порівняння (для прикметників)
        if word.endswith('er') and len(word) > 3:
            features['degree'] = 'comparative'
        elif word.endswith('est') and len(word) > 4:
            features['degree'] = 'superlative'
        else:
            features['degree'] = 'positive'
        
        features['word'] = word
        
        return features
    
    # Для української мови
    elif language == 'ua':
        features = {}
        
        # Визначаємо частину мови (спрощено)
        if word.endswith(('ти', 'тий', 'тих', 'тим')):
            features['pos'] = 'verb'
        elif word.endswith(('ість', 'ість', 'ість', 'ість')):
            features['pos'] = 'noun'
        elif word.endswith(('ий', 'а', 'е', 'і', 'ій', 'ою')):
            features['pos'] = 'adjective'
        else:
            features['pos'] = 'unknown'
        
        # Визначаємо число
        if word.endswith(('и', 'і', 'а')) and not word.endswith(('ія')):
            features['number'] = 'plural'
        else:
            features['number'] = 'singular'
        
        # Визначаємо відмінок (для іменників та прикметників)
        if word.endswith('а'):
            features['case'] = 'nominative'
        elif word.endswith('и'):
            features['case'] = 'genitive'
        elif word.endswith('і'):
            features['case'] = 'dative'
        elif word.endswith('у'):
            features['case'] = 'accusative'
        elif word.endswith('ою'):
            features['case'] = 'instrumental'
        elif word.endswith('і'):
            features['case'] = 'locative'
        else:
            features['case'] = 'vocative'
        
        # Визначаємо рід
        if word.endswith(('а', 'я')):
            features['gender'] = 'feminine'
        elif word.endswith(('о', 'е')):
            features['gender'] = 'neuter'
        else:
            features['gender'] = 'masculine'
        
        features['word'] = word
        
        return features
    
    return {}

def text_coherence_analysis(text: str, language: str = 'en') -> Dict[str, float]:
    """
    Проаналізувати зв'язність тексту.
    
    Параметри:
        text: Вхідний текст
        language: Мова тексту ('en' або 'ua'), за замовчуванням 'en'
    
    Повертає:
        Словник з показниками зв'язності
    """
    # Розбиваємо текст на речення
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) < 2:
        return {
            'coherence_score': 1.0,
            'avg_sentence_similarity': 1.0,
            'sentence_count': len(sentences)
        }
    
    # Токенізуємо речення
    tokenized_sentences = [tokenize_text(sentence, language) for sentence in sentences]
    
    # Видаляємо стоп-слова
    filtered_sentences = [remove_stop_words(tokens, language) for tokens in tokenized_sentences]
    
    # Обчислюємо подібність між послідовними реченнями
    similarities = []
    for i in range(len(filtered_sentences) - 1):
        set1 = set(filtered_sentences[i])
        set2 = set(filtered_sentences[i + 1])
        similarity = jaccard_similarity(set1, set2)
        similarities.append(similarity)
    
    # Середня подібність між послідовними реченнями
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    
    # Оцінка зв'язності (чим більше подібність, тим краще зв'язність)
    coherence_score = avg_similarity
    
    return {
        'coherence_score': coherence_score,
        'avg_sentence_similarity': avg_similarity,
        'sentence_count': len(sentences)
    }

def language_model_perplexity(text: str, ngram_model: Dict[Tuple[str, ...], float], 
                            n: int = 2, language: str = 'en') -> float:
    """
    Обчислити перплексію тексту відносно мовної моделі.
    
    Параметри:
        text: Вхідний текст
        ngram_model: N-грамна модель
        n: Розмір n-грами, за замовчуванням 2
        language: Мова тексту ('en' або 'ua'), за замовчуванням 'en'
    
    Повертає:
        Перплексія (чим менше, тим краще)
    """
    if not text or not ngram_model:
        return float('inf')
    
    # Токенізуємо текст
    tokens = tokenize_text(text, language)
    
    if len(tokens) < n:
        return float('inf')
    
    # Генеруємо n-грами
    text_ngrams = ngrams(tokens, n)
    
    # Обчислюємо логарифмічну ймовірність
    log_prob = 0.0
    ngram_count = 0
    
    for ngram in text_ngrams:
        prob = ngram_model.get(ngram, 1e-10)  # Мала ймовірність для невідомих n-грам
        log_prob += math.log(prob)
        ngram_count += 1
    
    if ngram_count == 0:
        return float('inf')
    
    # Обчислюємо перплексію
    avg_log_prob = log_prob / ngram_count
    perplexity = math.exp(-avg_log_prob)
    
    return perplexity

def text_diversity_analysis(text: str, language: str = 'en') -> Dict[str, float]:
    """
    Проаналізувати різноманітність тексту.
    
    Параметри:
        text: Вхідний текст
        language: Мова тексту ('en' або 'ua'), за замовчуванням 'en'
    
    Повертає:
        Словник з показниками різноманітності
    """
    # Токенізуємо текст
    tokens = tokenize_text(text, language)
    
    if not tokens:
        return {
            'lexical_diversity': 0.0,
            'unique_words_ratio': 0.0,
            'repetition_rate': 0.0,
            'total_words': 0,
            'unique_words': 0
        }
    
    # Підраховуємо унікальні слова
    unique_words = set(tokens)
    
    # Лексичне різноманіття (TTR - Type-Token Ratio)
    lexical_diversity = len(unique_words) / len(tokens)
    
    # Відношення унікальних слів
    unique_words_ratio = len(unique_words) / len(tokens)
    
    # Рівень повторень
    repetition_rate = 1.0 - lexical_diversity
    
    return {
        'lexical_diversity': lexical_diversity,
        'unique_words_ratio': unique_words_ratio,
        'repetition_rate': repetition_rate,
        'total_words': len(tokens),
        'unique_words': len(unique_words)
    }

def syntactic_analysis(text: str, language: str = 'en') -> Dict[str, int]:
    """
    Провести синтаксичний аналіз тексту.
    
    Параметри:
        text: Вхідний текст
        language: Мова тексту ('en' або 'ua'), за замовчуванням 'en'
    
    Повертає:
        Словник з синтаксичними характеристиками
    """
    # Розбиваємо текст на речення
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Аналіз речень
    sentence_analysis = {
        'total_sentences': len(sentences),
        'simple_sentences': 0,
        'complex_sentences': 0,
        'compound_sentences': 0,
        'avg_sentence_length': 0
    }
    
    total_words = 0
    
    for sentence in sentences:
        # Токенізуємо речення
        words = tokenize_text(sentence, language)
        total_words += len(words)
        
        # Підраховуємо кількість сполучників
        conjunctions = 0
        if language == 'ua':
            conjunction_words = {'і', 'та', 'але', 'однак', 'проте', 'зате', 'тому', 'бо', 'оскільки', 'тож'}
        else:
            conjunction_words = {'and', 'but', 'or', 'nor', 'for', 'yet', 'so', 'because', 'since', 'although', 'though'}
        
        for word in words:
            if word in conjunction_words:
                conjunctions += 1
        
        # Класифікуємо речення
        if conjunctions == 0:
            sentence_analysis['simple_sentences'] += 1
        elif conjunctions == 1:
            sentence_analysis['compound_sentences'] += 1
        else:
            sentence_analysis['complex_sentences'] += 1
    
    # Середня довжина речення
    if sentences:
        sentence_analysis['avg_sentence_length'] = total_words / len(sentences)
    
    return sentence_analysis

def semantic_network_analysis(text: str, language: str = 'en') -> Dict[str, Union[int, float, List[Tuple[str, str]]]]:
    """
    Проаналізувати семантичну мережу тексту.
    
    Параметри:
        text: Вхідний текст
        language: Мова тексту ('en' або 'ua'), за замовчуванням 'en'
    
    Повертає:
        Словник з характеристиками семантичної мережі
    """
    # Токенізуємо текст
    tokens = tokenize_text(text, language)
    
    # Видаляємо стоп-слова
    filtered_tokens = remove_stop_words(tokens, language)
    
    if len(filtered_tokens) < 2:
        return {
            'nodes': 0,
            'edges': 0,
            'density': 0.0,
            'central_words': [],
            'clustering_coefficient': 0.0
        }
    
    # Створюємо семантичну мережу (спрощена версія)
    # Вузли - унікальні слова
    unique_words = list(set(filtered_tokens))
    
    # Ребра - спільне вживання в вікні
    window_size = 3
    edges = set()
    
    for i in range(len(filtered_tokens)):
        word1 = filtered_tokens[i]
        # Дивимося на слова в межах вікна
        for j in range(max(0, i - window_size), min(len(filtered_tokens), i + window_size + 1)):
            if i != j:
                word2 = filtered_tokens[j]
                # Сортуємо слова для уніфікації ребер
                edge = tuple(sorted([word1, word2]))
                edges.add(edge)
    
    # Густина мережі
    n = len(unique_words)
    max_edges = n * (n - 1) / 2 if n > 1 else 0
    density = len(edges) / max_edges if max_edges > 0 else 0
    
    # Найцентральніші слова (за кількістю зв'язків)
    word_connections = defaultdict(int)
    for word1, word2 in edges:
        word_connections[word1] += 1
        word_connections[word2] += 1
    
    # Сортуємо за кількістю зв'язків
    central_words = sorted(word_connections.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Коефіцієнт кластеризації (спрощений)
    # Для кожного слова рахуємо, яка частина можливих зв'язків між його сусідами реалізована
    clustering_sum = 0
    nodes_with_neighbors = 0
    
    for word in unique_words:
        neighbors = set()
        for edge in edges:
            if word in edge:
                neighbor = edge[0] if edge[1] == word else edge[1]
                neighbors.add(neighbor)
        
        if len(neighbors) > 1:
            # Підраховуємо кількість зв'язків між сусідами
            actual_edges = 0
            for n1 in neighbors:
                for n2 in neighbors:
                    if n1 != n2:
                        edge = tuple(sorted([n1, n2]))
                        if edge in edges:
                            actual_edges += 1
            
            # Кількість можливих зв'язків між сусідами
            possible_edges = len(neighbors) * (len(neighbors) - 1)
            
            if possible_edges > 0:
                clustering_coeff = actual_edges / possible_edges
                clustering_sum += clustering_coeff
                nodes_with_neighbors += 1
    
    clustering_coefficient = clustering_sum / nodes_with_neighbors if nodes_with_neighbors > 0 else 0
    
    return {
        'nodes': n,
        'edges': len(edges),
        'density': density,
        'central_words': central_words,
        'clustering_coefficient': clustering_coefficient
    }

def text_compression(text: str, language: str = 'en') -> Tuple[str, float]:
    """
    Стиснути текст за допомогою простого алгоритму.
    
    Параметри:
        text: Вхідний текст
        language: Мова тексту ('en' або 'ua'), за замовчуванням 'en'
    
    Повертає:
        Кортеж (стиснутий текст, коефіцієнт стиснення)
    """
    if not text:
        return "", 0.0
    
    # Токенізуємо текст
    tokens = tokenize_text(text, language)
    
    if not tokens:
        return text, 1.0
    
    # Створюємо словник частот
    word_freq = Counter(tokens)
    
    # Створюємо просте кодування (найчастіші слова отримують коротші коди)
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Створюємо словник кодів
    code_dict = {}
    for i, (word, freq) in enumerate(sorted_words):
        # Просте кодування: частіше слово - коротший код
        code_dict[word] = str(i)
    
    # Кодуємо текст
    encoded_tokens = [code_dict.get(token, token) for token in tokens]
    encoded_text = ' '.join(encoded_tokens)
    
    # Коефіцієнт стиснення
    original_size = len(text.encode('utf-8'))
    compressed_size = len(encoded_text.encode('utf-8'))
    compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
    
    return encoded_text, compression_ratio

def text_decompression(encoded_text: str, code_dict: Dict[str, str]) -> str:
    """
    Розпакувати стиснутий текст.
    
    Параметри:
        encoded_text: Стиснутий текст
        code_dict: Словник кодів
    
    Повертає:
        Розпакований текст
    """
    if not encoded_text:
        return ""
    
    # Створюємо зворотній словник
    reverse_dict = {code: word for word, code in code_dict.items()}
    
    # Розкодовуємо текст
    tokens = encoded_text.split()
    decoded_tokens = [reverse_dict.get(token, token) for token in tokens]
    
    return ' '.join(decoded_tokens)
import streamlit as st
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
from dotenv import load_dotenv

# Завантажуємо змінні середовища
load_dotenv()

class QSInsightsGenerator:
    """
    LLM модель для генерації інсайтів на основі експериментальних даних QS рейтингу
    """
    
    def __init__(self):
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Ініціалізація клієнта Google Gemini"""
        try:
            from google import genai
            if self.api_key:
                self.client = genai.Client(api_key=self.api_key)
                print("✅ LLM клієнт ініціалізовано")
            else:
                print("⚠️ GOOGLE_API_KEY не знайдено в змінних середовища")
        except ImportError:
            print("⚠️ Бібліотека google-genai не встановлена")
        except Exception as e:
            print(f"⚠️ Помилка ініціалізації LLM клієнта: {e}")
    
    def generate_insights(self, experiment_data: Dict[str, Any], 
                         current_qs: float, max_ru: float) -> Dict[str, Any]:
        """
        Генерує інсайти на основі одного експерименту
        
        Args:
            experiment_data: Дані одного експерименту
            current_qs: Поточний QS Score
            max_ru: Максимальний бюджет RU
            
        Returns:
            Словник зі статусом та текстовою відповіддю
        """
        if not experiment_data:
            return {
                "status": "no_data",
                "text": "Немає даних для аналізу"
            }
        
        if not self.client:
            return {
                "status": "no_api",
                "text": "API ключ Google Gemini не налаштовано. Додайте GOOGLE_API_KEY у файл .env"
            }
        
        try:
            print("\n🚀 ПОЧАТОК ГЕНЕРАЦІЇ ІНСАЙТІВ")
            print(f"📊 Експеримент: {experiment_data.get('algorithm', 'Unknown')}")
            print(f"📈 Поточний QS: {current_qs:.3f} → Цільовий: {experiment_data.get('qs_score', 0):.3f}")
            print(f"💰 Бюджет: {experiment_data.get('ru_used', 0):.1f} / {max_ru:.0f} RU")
            
            # Генерація промпту напряму з даних експерименту
            prompt = self._create_single_experiment_prompt(experiment_data, current_qs, max_ru)
            
            print(f"\n📄 Промпт створено (довжина: {len(prompt)} символів)")
            
            # Виклик LLM
            response = self._call_llm(prompt)
            
            # Парсинг відповіді
            result = self._parse_llm_response(response)
            
            print("✅ ІНСАЙТИ УСПІШНО ЗГЕНЕРОВАНО\n")
            
            return result
            
        except Exception as e:
            print(f"❌ ПОМИЛКА ГЕНЕРАЦІЇ ІНСАЙТІВ: {e}\n")
            return {
                "status": "error",
                "text": f"Помилка аналізу: {str(e)}"
            }
    
    
    
    def _create_single_experiment_prompt(self, experiment_data: Dict[str, Any], 
                                        current_qs: float, max_ru: float) -> str:
        """Створення промпту для аналізу одного експерименту"""
        
        # Отримуємо дані з експерименту
        algorithm = experiment_data.get("algorithm", "Unknown")
        qs_score = experiment_data.get("qs_score", 0)
        ru_used = experiment_data.get("ru_used", 0)
        execution_time = experiment_data.get("execution_time", 0)
        improved_indicators = experiment_data.get("improved_indicators", [])
        timestamp = experiment_data.get("timestamp", "")
        
        # Метрики порівняння
        comparison_metrics = experiment_data.get("comparison_metrics", {})
        improvement = comparison_metrics.get("improvement", 0)
        improvement_percent = comparison_metrics.get("improvement_percent", 0)
        efficiency = comparison_metrics.get("efficiency", 0)
        budget_utilization = comparison_metrics.get("budget_utilization", 0)
        
        # Формуємо список покращених показників з відсотками (якщо є)
        improved_details = []
        for indicator in improved_indicators:
            # Тут можна додати деталі про зміни показників
            improved_details.append(indicator)
        
        prompt = f"""
Ти - експерт-консультант з покращення позицій університетів у рейтингу QS World University Rankings.
Твоя задача - допомогти університету зрозуміти, ЯК ПРАКТИЧНО покращити QS Score на основі результатів експерименту.

═══════════════════════════════════════════════════════════════════
📊 КОНТЕКСТ: МЕТОДОЛОГІЯ QS WORLD UNIVERSITY RANKINGS 2025
═══════════════════════════════════════════════════════════════════

QS оцінює університети за 9 показниками з різною вагою:

1. Academic Reputation (Академічна репутація) - 30%
   Вимірюється через глобальне опитування науковців про престиж університету.

2. Employer Reputation (Репутація серед роботодавців) - 15%
   Опитування роботодавців про якість випускників університету.

3. Faculty/Student Ratio (Співвідношення викладачів до студентів) - 10%
   Показує ресурсне забезпечення навчального процесу.

4. Citations per Faculty (Цитування на викладача) - 20%
   Науковий вплив: кількість цитувань публікацій, нормалізованих на кількість викладачів.

5. International Faculty Ratio (Частка іноземних викладачів) - 5%
   Рівень інтернаціоналізації викладацького складу.

6. International Student Ratio (Частка іноземних студентів) - 5%
   Привабливість університету для міжнародних студентів.

7. International Research Network (Міжнародна дослідницька мережа) - 5%
   Здатність налагоджувати глобальні наукові партнерства.

8. Employment Outcomes (Результати працевлаштування) - 5%
   Успішність випускників на ринку праці після закінчення.

9. Sustainability (Сталий розвиток) - 5%
   Внесок університету у сталий розвиток та екологічні ініціативи.

═══════════════════════════════════════════════════════════════════
📈 KPI РЕКТОРА ЗА 2024 РІК (ПЛАН / ФАКТ)
═══════════════════════════════════════════════════════════════════

НАУКОВА ДІЯЛЬНІСТЬ:
• Публікації в Scopus та WoS: План 14% / Факт 11% ❌ (недовиконано)
• Відрахування на наукові дослідження: План 9% / Факт 22.8% ✅ (перевиконано)

ІНТЕРНАЦІОНАЛІЗАЦІЯ:
• Іноземні студенти та фахівці: План 45% / Факт 35% ❌ (недовиконано на 10%)
• Академічна мобільність: План 3% / Факт 3.6% ✅
• Міжнародна мобільність (подвійні дипломи): План 6% / Факт 6% ✅

ПРАЦЕВЛАШТУВАННЯ:
• Випускники працевлаштовані за спеціальністю: План 3% / Факт 3.3% ✅

ЯКІСТЬ ОСВІТИ:
• Startup-проєкти студентів: План 4% / Факт 4% ✅
• Автоматизована система контролю якості: Впроваджено ✅
• Оновлення навчальних програм англійською: Виконано ✅

═══════════════════════════════════════════════════════════════════
🔗 ЗВ'ЯЗОК МІЖ KPI РЕКТОРА ТА ПОКАЗНИКАМИ QS
═══════════════════════════════════════════════════════════════════

KPI "Публікації в Scopus/WoS" → Citations per Faculty (20% ваги QS)
KPI "Відрахування на дослідження" → Citations + Academic Reputation (50% ваги)
KPI "Іноземні студенти/фахівці" → International Student/Faculty Ratio (10% ваги)
KPI "Академічна мобільність" → International Research Network (5% ваги)
KPI "Працевлаштування випускників" → Employment Outcomes (5% ваги)
KPI "Англомовні програми" → International Student Ratio + Employer Reputation

ВІДСУТНІ KPI (які варто додати):
• Частка іноземних викладачів (% від загальної кількості)
• Міжнародні співавторства (% публікацій з іноземними співавторами)
• Середня кількість цитувань на публікацію
• Участь у міжнародних дослідницьких проєктах (кількість/обсяг фінансування)
• Співвідношення студенти/викладачі (target: <15:1)

═══════════════════════════════════════════════════════════════════
🎯 КЛЮЧОВІ СТРАТЕГІЇ ПОКРАЩЕННЯ QS (ЄВРОПЕЙСЬКИЙ ДОСВІД)
═══════════════════════════════════════════════════════════════════

1. НАУКОВА АКТИВНІСТЬ:
   • Стимулювання публікацій у топ-журналах (Q1-Q2)
   • Міжнародні співавторства (підвищують цитованість на 30-50%)
   • Участь у міжнародних конференціях та проєктах
   • Інвестиції в наукову інфраструктуру

2. ІНТЕРНАЦІОНАЛІЗАЦІЯ:
   • Англомовні програми (магістратура, PhD)
   • Спрощення процедур для іноземців
   • Стипендійні програми для іноземних студентів
   • Запрошення іноземних викладачів (гостьові лекції, спільні курси)
   • Програми академічної мобільності (Erasmus+, подвійні дипломи)

3. РЕПУТАЦІЯ:
   • Промоція досягнень на міжнародних платформах
   • Участь в опитуваннях QS (Academic/Employer surveys)
   • Зв'язок з випускниками та роботодавцями
   • Міжнародні акредитації програм

4. ЯКІСТЬ ОСВІТИ:
   • Оновлення програм з урахуванням вимог ринку
   • Залучення практиків до викладання
   • Інтеграція з підприємствами (стажування, проєкти)
   • Підвищення кваліфікації викладачів за кордоном

═══════════════════════════════════════════════════════════════════
📊 РЕЗУЛЬТАТИ ЕКСПЕРИМЕНТУ
═══════════════════════════════════════════════════════════════════
QS Score для університету за 2025 рік дорівнює {current_qs:.3f}

За допомогою алгоритму "{algorithm}" було знайдено рішення:
• Можна покращити QS Score до {qs_score:.3f}
• Покращення: +{improvement:.3f} балів (+{improvement_percent:.1f}%)
• Показники для покращення: {', '.join(improved_indicators) if improved_indicators else 'не визначено'}
Усі ці експеримент проводяться на рік, тобто якщо в нас QS score 2025 = {current_qs:.3f}, то {qs_score:.3f} - це потенційний QS score 2026 року.
Ресурси:
• Витрачено RU: {ru_used:.1f} з {max_ru:.0f} доступних ({budget_utilization:.1%})
• Ефективність: {efficiency:.4f} балів QS на одиницю RU
• Час виконання: {execution_time:.1f} сек

═══════════════════════════════════════════════════════════════════
❓ ПИТАННЯ ДЛЯ АНАЛІЗУ
═══════════════════════════════════════════════════════════════════

1. ЯК ПРАКТИЧНО ПОКРАЩИТИ ЦІ ПОКАЗНИКИ? (розписуй без RU одиниць)
   Для кожного показника ({', '.join(improved_indicators)}):
   - Які конкретні дії має зробити університет?
   - Які ресурси потрібні?
   - Який реалістичний термін реалізації?

2. СТРАТЕГІЇ ДЛЯ KPI РЕКТОРА:
   
   а) ІСНУЮЧІ KPI:
      - Які KPI з 2024 року допоможуть покращити ці показники QS?
      - Чому деякі KPI недовиконані? (Публікації 11%, Іноземці 35%)
      - Які цілі встановити на наступний рік для цих KPI?
   
   б) НОВІ KPI:
      - Які нові KPI варто додати для цілеспрямованого покращення показників QS?
      - Які метрики відстежувати?
      - Які реалістичні цільові значення?
   
   в) ЗВ'ЯЗОК KPI → QS:
      - Як саме кожен KPI впливає на показники QS?
      - Яка пріоритетність KPI для досягнення цільового QS Score {qs_score:.3f}?

═══════════════════════════════════════════════════════════════════
📝 ФОРМАТ ВІДПОВІДІ
═══════════════════════════════════════════════════════════════════

Надай детальну текстову відповідь з такою структурою:

**1. ПРАКТИЧНІ ДІЇ ДЛЯ ПОКРАЩЕННЯ ПОКАЗНИКІВ**
(для кожного показника окремо, але включай інформацію наскільки покращити)

**2. НОВІ KPI ДЛЯ ВПРОВАДЖЕННЯ**
(конкретні нові показники з цільовими значеннями)

**3. ПЛАН РЕАЛІЗАЦІЇ**
(завжди на рік)

**4. ЗАГАЛЬНІ ВИСНОВКИ**
(підсумок та рекомендації)

Відповідай КОНКРЕТНО, з числами, термінами та практичними діями. Уникай загальних фраз.
Зважай на реальні можливості українського університету в умовах обмежених ресурсів.

ВАЖЛИВО: Давай СТИСЛІ відповіді - максимум 2-3 абзаци на кожен розділ. 
Фокусуйся на найважливіших діях та рекомендаціях без зайвих деталей.
"""
        
        return prompt
    
    
    def _call_llm(self, prompt: str) -> str:
        """Виклик LLM моделі"""
        try:
            print("\n" + "="*80)
            print("🤖 ВИКЛИК LLM...")
            print("="*80)
            
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config={
                    "temperature": 0.7,
                    "max_output_tokens": 4000
                }
            )
            
            print("\n✅ ВІДПОВІДЬ ОТРИМАНО")
            print("="*80)
            print(response.text)
            print("="*80 + "\n")
            
            return response.text
        except Exception as e:
            print(f"⚠️ Помилка виклику LLM: {e}")
            raise
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Парсинг текстової відповіді LLM
        Тепер очікуємо структуровану текстову відповідь, а не JSON
        """
        try:
            print("📝 ПАРСИНГ ВІДПОВІДІ...")
            
            if not response or len(response.strip()) == 0:
                print("⚠️ Отримано порожню відповідь")
                return {
                    "status": "empty",
                    "text": "Отримано порожню відповідь від LLM",
                    "raw_response": response
                }
            
            print(f"✅ Відповідь успішно оброблена (довжина: {len(response)} символів)")
            
            return {
                "status": "success",
                "text": response.strip(),
                "raw_response": response
            }
                
        except Exception as e:
            print(f"⚠️ Помилка парсингу відповіді: {e}")
            return {
                "status": "error",
                "text": f"Помилка обробки відповіді: {str(e)}",
                "raw_response": response
            }
    

# Глобальний екземпляр генератора інсайтів
_insights_generator = None

def get_insights_generator() -> QSInsightsGenerator:
    """Отримати глобальний генератор інсайтів"""
    global _insights_generator
    if _insights_generator is None:
        _insights_generator = QSInsightsGenerator()
    return _insights_generator

def generate_qs_insights(experiment_data: Dict[str, Any], 
                        current_qs: float, max_ru: float) -> Dict[str, Any]:
    """
    Зручна функція для генерації інсайтів для одного експерименту
    
    Args:
        experiment_data: Дані одного експерименту
        current_qs: Поточний QS Score
        max_ru: Максимальний бюджет RU
        
    Returns:
        Словник з інсайтами та рекомендаціями
    """
    generator = get_insights_generator()
    return generator.generate_insights(experiment_data, current_qs, max_ru)

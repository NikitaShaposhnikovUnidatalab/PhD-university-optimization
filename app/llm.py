import streamlit as st
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

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
            
            # Debug: виводимо частину промпту з розділу ДЕТАЛЬНІ ЗМІНИ ПОКАЗНИКІВ
            if "ДЕТАЛЬНІ ЗМІНИ ПОКАЗНИКІВ:" in prompt:
                start = prompt.find("ДЕТАЛЬНІ ЗМІНИ ПОКАЗНИКІВ:")
                end = prompt.find("РЕСУРСИ:", start)
                if end > start:
                    print(f"\n🔍 DEBUG - Фрагмент промпту:")
                    print(prompt[start:end])
            
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
        
        # Отримуємо поточні та нові значення показників
        QS_INPUT = experiment_data.get("QS_INPUT", {})
        solution = experiment_data.get("solution", [])
        
        # Конвертуємо solution з numpy array в list якщо потрібно
        if solution is not None and hasattr(solution, '__len__') and not isinstance(solution, (list, tuple)):
            try:
                solution = list(solution)
            except:
                pass
        
        # Debug logging
        solution_available = solution is not None and len(solution) > 0 if hasattr(solution, '__len__') else False
        print(f"\n🔍 DEBUG: QS_INPUT наявний: {bool(QS_INPUT)}, кількість ключів: {len(QS_INPUT) if QS_INPUT else 0}")
        print(f"🔍 DEBUG: solution наявний: {solution_available}, кількість елементів: {len(solution) if solution_available else 0}")
        
        # Словник з описами показників
        indicator_names = {
            "AR": "Academic Reputation (Академічна репутація)",
            "ER": "Employer Reputation (Репутація серед роботодавців)",
            "FSR": "Faculty/Student Ratio (Співвідношення викладачів до студентів)",
            "CPF": "Citations per Faculty (Цитування на викладача)",
            "IFR": "International Faculty Ratio (Частка іноземних викладачів)",
            "ISR": "International Student Ratio (Частка іноземних студентів)",
            "IRN": "International Research Network (Міжнародна дослідницька мережа)",
            "EO": "Employment Outcomes (Результати працевлаштування)",
            "SUS": "Sustainability (Сталий розвиток)"
        }
        
        # Формуємо детальний список змін показників
        improved_details = []
        all_indicators_info = []  # Список для всіх показників (на випадок якщо немає змін)
        
        # Перевіряємо наявність даних (безпечно для numpy arrays)
        has_data = bool(QS_INPUT) and solution is not None and len(solution) > 0
        
        if has_data:
            all_keys = list(QS_INPUT.keys())
            for i, key in enumerate(all_keys):
                current_value = float(QS_INPUT[key])
                new_value = float(solution[i]) if i < len(solution) else current_value
                delta = new_value - current_value
                
                indicator_full_name = indicator_names.get(key, key)
                
                # Додаємо всі показники до загального списку
                all_indicators_info.append(
                    f"   • {indicator_full_name}: {current_value:.2f} → {new_value:.2f} (зміна: {delta:+.2f})"
                )
                
                # Додаємо до списку змінених тільки ті, що покращились
                if abs(delta) > 0.001:  # Якщо є зміна
                    improved_details.append(
                        f"   • {indicator_full_name}: {current_value:.2f} → {new_value:.2f} (зміна: {delta:+.2f})"
                    )
        
        # Якщо немає змін, показуємо всі показники
        if not improved_details and all_indicators_info:
            print(f"⚠️ DEBUG: Немає значних змін показників, показуємо всі {len(all_indicators_info)} показників")
            improved_details = all_indicators_info
        else:
            print(f"✅ DEBUG: Знайдено {len(improved_details)} показників зі значними змінами")
        
        prompt = f"""
Ти - експерт-консультант з покращення позицій університетів у рейтингу QS World University Rankings.
Твоя задача - допомогти університету зрозуміти, ЯК ПРАКТИЧНО покращити QS Score на основі результатів експерименту.

═══════════════════════════════════════════════════════════════════
📊 КОНТЕКСТ: МЕТОДОЛОГІЯ QS WORLD UNIVERSITY RANKINGS 2025/2026
═══════════════════════════════════════════════════════════════════

QS Rankings використовує систему LENS → INDICATORS → METRICS:

СТРУКТУРА РЕЙТИНГУ (5 LENS З ІНДИКАТОРАМИ):

┌─────────────────────────────────────────────────────────────────┐
│ 1. RESEARCH AND DISCOVERY (Дослідження) — 50% (НАЙВАЖЛИВІШЕ!)  │
├─────────────────────────────────────────────────────────────────┤
│  • Academic Reputation (Академічна репутація) — 30%             │
│    Вимірюється через глобальне опитування науковців про престиж │
│                                                                  │
│  • Citations per Faculty (Цитування на викладача) — 20%         │
│    Науковий вплив: кількість цитувань публікацій,               │
│    нормалізованих на кількість викладачів                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 2. EMPLOYABILITY AND OUTCOMES (Працевлаштування) — 20%          │
├─────────────────────────────────────────────────────────────────┤
│  • Employer Reputation (Репутація серед роботодавців) — 15%     │
│    Опитування роботодавців про якість випускників               │
│                                                                  │
│  • Employment Outcomes (Результати працевлаштування) — 5%       │
│    Успішність випускників на ринку праці після закінчення      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 3. GLOBAL ENGAGEMENT (Міжнародна активність) — 15%              │
├─────────────────────────────────────────────────────────────────┤
│  • International Faculty Ratio (Частка іноземних викладачів) — 5%│
│    Рівень інтернаціоналізації викладацького складу             │
│                                                                  │
│  • International Student Ratio (Частка іноземних студентів) — 5%│
│    Привабливість університету для міжнародних студентів         │
│                                                                  │
│  • International Research Network (Міжнародна мережа) — 5%      │
│    Здатність налагоджувати глобальні наукові партнерства        │
│                                                                  │
│  • International Student Diversity — 0% (тільки для топ-500)    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 4. LEARNING EXPERIENCE (Якість навчання) — 10%                  │
├─────────────────────────────────────────────────────────────────┤
│  • Faculty/Student Ratio (Співвідношення викладачів/студентів) — 10%│
│    Показує ресурсне забезпечення навчального процесу            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 5. SUSTAINABILITY (Сталий розвиток) — 5%                        │
├─────────────────────────────────────────────────────────────────┤
│  • Sustainability (Сталість розвитку) — 5%                      │
│    Внесок університету у сталий розвиток та екологічні ініціативи│
└─────────────────────────────────────────────────────────────────┘

КЛЮЧОВІ ВИСНОВКИ З МЕТОДОЛОГІЇ:
▸ 50% рейтингу залежить від ДОСЛІДНИЦЬКОЇ ДІЯЛЬНОСТІ
  (Academic Reputation 30% + Citations per Faculty 20%)
▸ 20% від ПРАЦЕВЛАШТУВАННЯ та РЕПУТАЦІЇ серед роботодавців
▸ 15% від МІЖНАРОДНОЇ АКТИВНОСТІ (студенти, викладачі, мережі)
▸ 10% від ЯКОСТІ НАВЧАННЯ (співвідношення викладачі/студенти)
▸ 5% від СТАЛОГО РОЗВИТКУ

ПРІОРИТЕТИ ДЛЯ ПОКРАЩЕННЯ:
1️⃣ НАЙВИЩИЙ ПРІОРИТЕТ (70%): Academic Reputation (30%) + Citations (20%) + Employer Reputation (15%) + Employment (5%)
2️⃣ СЕРЕДНІЙ ПРІОРИТЕТ (15%): International Faculty/Student Ratio + Research Network
3️⃣ БАЗОВИЙ РІВЕНЬ (15%): Faculty/Student Ratio (10%) + Sustainability (5%)

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
📊 ФАКТИЧНІ ДАНІ УНІВЕРСИТЕТУ (QS WUR 2025 vs 2026)
═══════════════════════════════════════════════════════════════════

ВИКЛАДАЧІ (Faculty Staff):
                                    2025 (2022/23)      2026 (2023/24)
• Всього FTE:                       1355                1333.5 ⬇️ (-21.5)
• Іноземних викладачів:             37                  45 ⬆️ (+8)
• З PhD:                            1210                1190 ⬇️ (-20)

СТУДЕНТИ (Students Overall):
                                    2025 (2022/23)      2026 (2023/24)
• Всього FTE:                       11572               11134 ⬇️ (-438)
• Бакалаври FTE:                    8833                8352.6 ⬇️
• Магістри/Аспіранти FTE:           2739                2781 ⬆️

МІЖНАРОДНІ СТУДЕНТИ:
                                    2025 (2022/23)      2026 (2023/24)
• Всього міжнародних FTE:           1558                1558 ➡️ (0)
• Бакалаври міжнародні:             1273                1273 ➡️
• Магістри міжнародні:              285                 285 ➡️
• % від загальної кількості:        13.5%               14.0% ⬆️

АКАДЕМІЧНА МОБІЛЬНІСТЬ:
                                    2025 (2022/23)      2026 (2023/24)
• Outbound exchange (виїзд):        80                  215 ⬆️ (+135, +169%)
• Кількість національностей:        47                  47 ➡️

СПІВВІДНОШЕННЯ СТУДЕНТИ/ВИКЛАДАЧІ:
                                    2025 (2022/23)      2026 (2023/24)
• Student/Faculty Ratio:            8.5:1               8.35:1 ⬆️ (покращення)

ПРОГРАМИ:
                                    2025 (2022/23)      2026 (2023/24)
• Бакалаврські програми:            45                  74 ⬆️ (+64%)
• Магістерські програми:            76                  137 ⬆️ (+80%)

ЯКІСТЬ ОСВІТИ:
                                    2025 (2022/23)      2026 (2023/24)
• Student Retention Rate:           98.2%               99.1% ⬆️
• Completion Rate:                  93.2%               94.0% ⬆️
• Student Continuation Rate:        91%                 95% ⬆️

ПРАЦЕВЛАШТУВАННЯ (Employment Statistics):
                                    2025 (2022/23)      2026 (2023/24)
• Випускників всього:               1524                1483
• Відповіли на опитування:          706 (46.3%)         816 (56.2%) ⬆️
• QS Employment Rate:               96.2%               98.1% ⬆️
• Працевлаштовані:                  54%                 96.2% ⬆️⬆️⬆️
• Медіанна зарплата (USD):          $500                $550 ⬆️
• Посилання на звіт:                http://career.kharkov.ua/?page_id=2332

СТИПЕНДІЇ:
                                    2025 (2022/23)      2026 (2023/24)
• 100% покриття:                    0                   175 ⬆️⬆️⬆️
• 50%+ покриття:                    0                   0 ➡️

ПЛАТА ЗА НАВЧАННЯ (USD):
                                    2025 (2022/23)      2026 (2023/24)
• Бакалавр (міжнародні):            $1715               $1400 ⬇️
• Магістр (міжнародні):             $1995               $2123 ⬆️

КЛЮЧОВІ СПОСТЕРЕЖЕННЯ З ДАНИХ:
✅ ПОЗИТИВНІ ТРЕНДИ:
   • Значне збільшення кількості програм (+64% бакалавр, +80% магістр)
   • Покращення працевлаштування (96.2% → 98.1% QS Employment Rate)
   • Збільшення академічної мобільності на виїзд (+169%)
   • Впровадження стипендійних програм (175 студентів з 100% покриттям)
   • Збільшення іноземних викладачів (+8)

⚠️ ПРОБЛЕМНІ ЗОНИ:
   • Зменшення викладачів з PhD (1210 → 1190, -20)
   • Зменшення загальної кількості студентів (-438 FTE)
   • Відсутність зростання кількості міжнародних студентів (1558 → 1558)
   • Немає студентів на дистанційному навчанні (упущена можливість)
   • Низька частка іноземних викладачів (45 з 1333.5 = 3.4%)

═══════════════════════════════════════════════════════════════════
🎯 СТРАТЕГІЇ ПОКРАЩЕННЯ QS (ПРІОРИТИЗОВАНІ ЗА ВАГОЮ)
═══════════════════════════════════════════════════════════════════

🔴 КРИТИЧНИЙ ПРІОРИТЕТ (50% рейтингу) — RESEARCH AND DISCOVERY:

1. ACADEMIC REPUTATION (30%):
   • Активна участь у QS Academic Survey (науковці, які відповідають)
   • Промоція досягнень на міжнародних платформах та конференціях
   • Публікації в топ-журналах з високим impact factor
   • Міжнародні акредитації та членство в престижних асоціаціях
   • Організація міжнародних конференцій та літніх шкіл

2. CITATIONS PER FACULTY (20%):
   • Стимулювання публікацій у топ-журналах (Q1-Q2)
   • Міжнародні співавторства (підвищують цитованість на 30-50%)
   • Залучення викладачів з високим h-index
   • Інвестиції в наукову інфраструктуру та дослідження
   • Підтримка молодих науковців у публікаційній діяльності

🟠 ВИСОКИЙ ПРІОРИТЕТ (20% рейтингу) — EMPLOYABILITY:

3. EMPLOYER REPUTATION (15%):
   • Активна участь у QS Employer Survey (роботодавці-партнери)
   • Практичні проєкти зі студентами для компаній
   • Advisory boards з представників бізнесу
   • Кар'єрні ярмарки та зустрічі з роботодавцями
   • Промоція успішних випускників та їхніх досягнень

4. EMPLOYMENT OUTCOMES (5%):
   • Центр кар'єри: консультації, тренінги, підготовка до співбесід
   • Обов'язкові стажування та практики
   • Відстеження працевлаштування випускників (6 місяців після випуску)
   • Залучення практиків до викладання
   • Soft skills та підприємницькі програми

🟡 СЕРЕДНІЙ ПРІОРИТЕТ (15% рейтингу) — GLOBAL ENGAGEMENT:

5. ІНТЕРНАЦІОНАЛІЗАЦІЯ (International Faculty 5% + Student 5% + Network 5%):
   • Англомовні програми (магістратура, PhD)
   • Стипендійні програми для іноземних студентів
   • Залучення іноземних викладачів (full-time та visiting)
   • Міжнародні дослідницькі проєкти та гранти (Horizon Europe, Erasmus+)
   • Подвійні дипломи та спільні програми з іноземними ВНЗ
   • Спрощення візових процедур та адаптаційна підтримка

🟢 БАЗОВИЙ ПРІОРИТЕТ (15% рейтингу):

6. LEARNING EXPERIENCE — Faculty/Student Ratio (10%):
   • Оптимізація навантаження викладачів
   • Залучення нових PhD-викладачів
   • Малі групи для практичних занять

7. SUSTAINABILITY (5%):
   • Екологічні ініціативи та green campus
   • Курси про сталий розвиток
   • Енергоефективність та управління відходами

═══════════════════════════════════════════════════════════════════
📊 РЕЗУЛЬТАТИ ЕКСПЕРИМЕНТУ
═══════════════════════════════════════════════════════════════════
QS Score для університету за 2025 рік дорівнює {current_qs:.3f}

За допомогою алгоритму "{algorithm}" було знайдено рішення:
• Можна покращити QS Score до {qs_score:.3f}
• Покращення: +{improvement:.3f} балів (+{improvement_percent:.1f}%)
• Усі ці експеримент проводяться на рік, тобто якщо в нас QS score 2025 = {current_qs:.3f}, то {qs_score:.3f} - це потенційний QS score 2026 року.

ДЕТАЛЬНІ ЗМІНИ ПОКАЗНИКІВ:
{chr(10).join(improved_details) if improved_details else '   • Показники не змінювались'}

РЕСУРСИ:
• Витрачено RU: {ru_used:.1f} з {max_ru:.0f} доступних ({budget_utilization:.1%})
• Ефективність: {efficiency:.4f} балів QS на одиницю RU
• Час виконання: {execution_time:.1f} сек

═══════════════════════════════════════════════════════════════════
❓ ПИТАННЯ ДЛЯ АНАЛІЗУ
═══════════════════════════════════════════════════════════════════

1. ЯК ПРАКТИЧНО ПОКРАЩИТИ ЦІ ПОКАЗНИКИ? (розписуй без RU одиниць)
   Для кожного показника з розділу "ДЕТАЛЬНІ ЗМІНИ ПОКАЗНИКІВ" вище:
   - Які конкретні дії має зробити університет для досягнення вказаних змін?
   - Які ресурси потрібні?
   - ВРАХОВУЙ фактичні дані університету з розділу "ФАКТИЧНІ ДАНІ УНІВЕРСИТЕТУ"

2. СТРАТЕГІЇ ДЛЯ KPI РЕКТОРА:
   
   а) ІСНУЮЧІ KPI:
      - Які KPI з 2024 року допоможуть покращити ці показники QS?
      - Чому деякі KPI недовиконані? (Публікації 11%, Іноземці 35%)
      - Які цілі встановити на наступний рік для цих KPI?
   
   б) НОВІ KPI НА ОСНОВІ ФАКТИЧНИХ ДАНИХ:
      - Враховуючи ПРОБЛЕМНІ ЗОНИ з фактичних даних (зменшення викладачів з PhD, стагнація міжнародних студентів тощо)
      - Які нові KPI варто додати для цілеспрямованого покращення показників QS?
      - Які метрики відстежувати та які реалістичні цільові значення на 2026 рік?
      - Як використати ПОЗИТИВНІ ТРЕНДИ (збільшення програм, працевлаштування, мобільність)?
   
   в) ЗВ'ЯЗОК KPI → QS З УРАХУВАННЯМ РЕАЛЬНИХ ДАНИХ:
      - Як саме кожен KPI впливає на показники QS?
      - Яка пріоритетність KPI для досягнення цільового QS Score {qs_score:.3f}?
      - Які швидкі перемоги ("quick wins") можливі на основі існуючих досягнень?

═══════════════════════════════════════════════════════════════════
📝 ФОРМАТ ВІДПОВІДІ
═══════════════════════════════════════════════════════════════════

Надай детальну текстову відповідь з такою структурою:

**1. ПРАКТИЧНІ ДІЇ ДЛЯ ПОКРАЩЕННЯ ПОКАЗНИКІВ**
(для кожного показника з розділу "ДЕТАЛЬНІ ЗМІНИ ПОКАЗНИКІВ" вище:
 - вкажи повну назву показника
 - вкажи поточне значення → цільове значення
 - конкретні кроки для досягнення цієї зміни)

**2. НОВІ KPI ДЛЯ ВПРОВАДЖЕННЯ**
(конкретні нові показники з цільовими значеннями на 2026 рік)

**3. ЗАГАЛЬНІ ВИСНОВКИ**
(підсумок та рекомендації щодо пріоритетності дій)

Відповідай КОНКРЕТНО, з числами, термінами та практичними діями. Уникай загальних фраз.
Зважай на реальні можливості українського університету в умовах обмежених ресурсів.

ВАЖЛИВО: Давай СТИСЛІ відповіді - максимум 2-3 абзаци на кожен розділ. 
Фокусуйся на найважливіших діях та рекомендаціях без зайвих деталей.
"""
        
        return prompt
    
    
    def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Виклик LLM моделі з retry логікою"""
        import time
        
        for attempt in range(max_retries):
            try:
                print("\n" + "="*80)
                print(f"🤖 ВИКЛИК LLM... (спроба {attempt + 1}/{max_retries})")
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
                error_message = str(e)
                if "503" in error_message or "overloaded" in error_message.lower():
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt)
                        print(f"⚠️ Модель перевантажена. Чекаємо {wait_time}с перед наступною спробою...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"❌ Модель перевантажена після {max_retries} спроб")
                        raise Exception("Gemini API перевантажений. Спробуйте пізніше (через 1-2 хвилини)")
                else:
                    print(f"⚠️ Помилка виклику LLM: {e}")
                    raise
        
        raise Exception("Не вдалося отримати відповідь від LLM")
    
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

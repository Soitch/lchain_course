"""
prompt_template_demo.py
Демонстрация параметризованного PromptTemplate (LangChain) для задачи техподдержки
"""
# support_chain_stepwise.py

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from ..modelca import MODELCA  # ваш кастомный LLM-клиент

llm = MODELCA

# === Шаг 1: Генерация уточняющих вопросов ===
questions_prompt = PromptTemplate.from_template(
    """\
Ты — инженер службы поддержки (уровень L2). Пользователь сообщил следующее:

Пользователь: {user_name}
Продукт/сервис: {product}
Платформа: {platform}
Цель: {user_goal}
Симптомы: {issue_description}
Когда началось: {when_started}
Текст ошибки: {error_text}

Сформулируй **не более 5 кратких и конкретных уточняющих вопросов**, которые помогут локализовать проблему. 
Не повторяй уже известную информацию. Задавай только то, чего не хватает для диагностики.

Выведи только вопросы, по одному на строке, без вводных слов:
1. ...
2. ...
"""
)

# === Шаг 2: Пошаговый план действий ===
plan_prompt = PromptTemplate.from_template(
    """\
Ты — инженер службы поддержки (уровень L2). На основе следующего контекста:

Пользователь: {user_name}
Продукт/сервис: {product}
Платформа: {platform}
Цель: {user_goal}
Симптомы: {issue_description}
Когда началось: {when_started}
Текст ошибки: {error_text}

Составь **пошаговый план диагностики и решения**, соблюдая правила:
- Действия от простого к сложному.
- Если шаг рискует потерей данных — отметь [Риск данных] и предложи резервную копию.
- Не предлагай переустановку ОС или радикальные меры без крайней необходимости.
- Не выдумывай — если не хватает данных, предложи проверку, а не решение.

Выведи только нумерованный список шагов:
1. ...
2. ...
"""
)

# === Шаг 3: Альтернативы, если не помогло ===
fallback_prompt = PromptTemplate.from_template(
    """\
На основе той же проблемы:

Пользователь: {user_name}
Продукт: {product}
Симптомы: {issue_description}
Платформа: {platform}

Предложи **2 альтернативных варианта действий**, если базовый план не сработал. 
Например: обращение к другому специалисту, использование альтернативного метода, эскалация.

Формат:
- Вариант A: ...
- Вариант B: ...
"""
)

# === Шаг 4: Критерий успеха ===
success_prompt = PromptTemplate.from_template(
    """\
Как пользователь поймёт, что проблема решена? Сформулируй **один-два критерия успеха** 
для следующей ситуации:

Цель: {user_goal}
Симптомы: {issue_description}
Продукт: {product}

Пример: "Пользователь успешно входит в аккаунт и видит главный экран."

Ответ:
- ...
"""
)

# === Сборка цепочки ===
# Передаём исходные входные данные (без модификаций)
input_keys = [
    "user_name", "product", "platform", "user_goal",
    "issue_description", "when_started", "error_text"
]

chain = (
    # 1. Передаём исходный контекст
    {"user_name": RunnablePassthrough(), "product": RunnablePassthrough(), "platform": RunnablePassthrough(),
     "user_goal": RunnablePassthrough(), "issue_description": RunnablePassthrough(),
     "when_started": RunnablePassthrough(), "error_text": RunnablePassthrough()}
    # 2. Генерируем вопросы (не влияет на последующие шаги — только для финального вывода)
    | RunnablePassthrough.assign(
        clarifying_questions=lambda x: (questions_prompt | llm | StrOutputParser()).invoke(x)
    )
    # 3. Генерируем план
    | RunnablePassthrough.assign(
        action_plan=lambda x: (plan_prompt | llm | StrOutputParser()).invoke(x)
    )
    # 4. Генерируем fallback
    | RunnablePassthrough.assign(
        fallback_options=lambda x: (fallback_prompt | llm | StrOutputParser()).invoke(x)
    )
    # 5. Генерируем критерий успеха
    | RunnablePassthrough.assign(
        success_criteria=lambda x: (success_prompt | llm | StrOutputParser()).invoke(x)
    )
    # 6. Финальная сборка ответа
    | (lambda x: f"""\
### Уточняющие вопросы
{x['clarifying_questions']}

### План действий (по шагам)
{x['action_plan']}

### Если не помогло
{x['fallback_options']}

### Критерий успеха
{x['success_criteria']}
""")
)

# === Пример запуска ===
if __name__ == "__main__":
    example = {
        "user_name": "Иван",
        "product": "FinApp (мобильное приложение)",
        "platform": "Android 14, Samsung Galaxy S23, приложение v5.8.1",
        "user_goal": "войти в аккаунт",
        "issue_description": "После ввода SMS-кода экран зависает на 'Проверяем…', затем возвращает на экран входа.",
        "when_started": "Сегодня после обновления приложения. Сеть: Wi‑Fi и LTE пробовал.",
        "error_text": "нет (сообщение об ошибке не показывается)",
    }

    result = chain.invoke(example)
    print("=" * 80)
    print("ФИНАЛЬНЫЙ ОТВЕТ ТЕХПОДДЕРЖКИ")
    print("=" * 80)
    print(result)
"""
Шаблон для генерации ответа службы поддержки в два этапа:
1. Сначала модель формулирует уточняющие вопросы.
2. Затем — пошаговый план решения.

Параметры шаблона:
- {product}: название продукта или сервиса
- {issue}: краткое описание проблемы
- {platform}: ОС, устройство, версия — где возникла проблема
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from modelca import MODELCA  # ваш LLM

llm = MODELCA

# --- Шаг 1: сгенерировать уточняющие вопросы ---
prompt_questions = PromptTemplate.from_template(
    "Ты — инженер поддержки. Пользователь сообщает: "
    "«Проблема с {product}: {issue} на {platform}». "
    "Задай не более 3 кратких уточняющих вопросов, чтобы понять суть проблемы."
)

# --- Шаг 2: на основе контекста и вопросов — дать план ---
prompt_plan = PromptTemplate.from_template(
    "Ты — инженер поддержки. На основе проблемы: "
    "«{issue} в {product} на {platform}», "
    "и уточняющих вопросов: {questions} — "
    "предложи краткий пошаговый план диагностики (макс. 4 шага)."
)

# --- Цепочка: сначала вопросы, потом план ---
chain = (
    {"product": RunnablePassthrough(), "issue": RunnablePassthrough(), "platform": RunnablePassthrough()}
    | RunnablePassthrough.assign(
        questions=lambda x: (prompt_questions | llm | StrOutputParser()).invoke(x)
    )
    | prompt_plan
    | llm
    | StrOutputParser()
)

# --- Примеры использования ---
examples = [
    {
        "product": "мобильное приложение FinApp",
        "issue": "не приходит SMS с кодом подтверждения",
        "platform": "Android 14, Samsung Galaxy S23"
    },
    {
        "product": "веб-сервис StreamPlus",
        "issue": "видео не грузится, круглая загрузка крутится бесконечно",
        "platform": "Chrome 131 на Windows 11"
    }
]

if __name__ == "__main__":
    for i, params in enumerate(examples, 1):
        print(f"\n{'='*60}")
        print(f"Пример {i}")
        print(f"{'='*60}")
        result = chain.invoke(params)
        print(result)
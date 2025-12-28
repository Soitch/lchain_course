
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from modelca import MODELCA

LLM = MODELCA

template = "Отвечай как ассистент: Привет, {name}! Рад тебя видеть. Я могу помочь с вопросами по {topic}."
prompt = PromptTemplate(template=template, input_variables=["name", "topic"])

# ✅ Правильно: используем уже созданный LLM
chain = RunnableSequence(prompt, LLM)

# Или в современном стиле LangChain (рекомендуется):
# chain = prompt | LLM

result = chain.invoke({"name": "Анна", "topic": "программированию"})
print(result.content)
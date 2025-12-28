
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from modelca import MODELCA

#MODEL = os.getenv("OPENAI_API_MODEL", "gpt-5")
#llm = ChatOpenAI(model=MODEL)

llm=MODELCA

prompt1 = PromptTemplate.from_template("Назови 5 популярных достопримечательностей в городе {city}.")
prompt2 = PromptTemplate.from_template("Составь маршрут на 3 дня по городу {city}, включив следующие места: {places_list}.")

chain = (
    {"city": RunnablePassthrough()}
    | RunnablePassthrough.assign(
        places_list=lambda x: (prompt1 | llm | StrOutputParser()).invoke(x)
    )
    | prompt2
    | llm
    | StrOutputParser()
)

# Запуск
result = chain.invoke({"city": "Париж"})
print(result)
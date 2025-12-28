
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

# Промт для уведомления (prompt3)
prompt3 = PromptTemplate.from_template(
    "Уважаемый(ая) {client_name}, напоминаем, что на курсе «{course_name}» у вас есть незавершённые элементы программы: {issues_list}. "
    "Пожалуйста, завершите их в ближайшее время, чтобы успешно получить сертификат."
)

def mock_course_recommender(input_dict):
    hobbies = input_dict["hobbies_list"].lower()  # теперь input_dict — это наш исходный словарь
    if "фитнес" in hobbies or "скалолазание" in hobbies:
        return "Здоровый образ жизни и профилактика травм, Основы спортивной нутрициологии"
    elif "рыбалка" in hobbies or "путешествия" in hobbies:
        return "Основы выживания в дикой природе, Навигация и работа с GPS"
    elif "маркетинг" in hobbies or "цифровой" in hobbies:
        return "Цифровой маркетинг, Аналитика в Meta и Google"
    else:
        return "Универсальные навыки: тайм-менеджмент и продуктивность"

# Цепочка: вход → добавляем рекомендации → формируем уведомление
chain = (
    RunnablePassthrough()  # передаём исходный словарь: {"gift_target_name": ..., "hobbies_list": ...}
    | RunnablePassthrough.assign(
        recommended_courses=lambda x: mock_course_recommender(x)
    )
    | RunnablePassthrough.assign(
        client_name=lambda x: x["gift_target_name"],
        course_name=lambda x: x["recommended_courses"].split(",")[0].strip(),
        issues_list=lambda _: "не пройдён вводный тест, не подтверждена оплата"
    )
    | prompt3
    | (lambda x: x)  # просто возвращаем строку
)

# Вызов — передаём словарь с нужными ключами
result = chain.invoke({
    "gift_target_name": "Дмитрий",
    "hobbies_list": "фитнес, скалолазание"
})

print(result)
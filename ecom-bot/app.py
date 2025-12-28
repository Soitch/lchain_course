"""
Shoply Support Bot — консольный ассистент поддержки интернет-магазина.
Версия: 1.0.0

Функции:
- Ответы на частые вопросы (FAQ).
- Проверка статуса заказа по команде /order <номер>.
- Запоминание имени и номера заказа в рамках сессии.
- Логирование взаимодействий в формате JSONL.

"""

import os
import json
import httpx
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.callbacks import get_openai_callback
import logging

load_dotenv()


# === Кастомный JSONL-логгер ===
class JsonlFileHandler(logging.Handler):
    def __init__(self, base_dir: Path, session_id: str):
        super().__init__()
        self.base_dir = base_dir
        self.session_id = session_id
        self.logs_dir = base_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        self.filename = self.logs_dir / f"session_{session_id}.jsonl"

    def emit(self, record):
        log_entry = json.loads(record.getMessage())
        with open(self.filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


# === Основной класс бота ===
class ShopSupportBot:
    SYSTEM_PROMPT = (
        "Ты — ассистент поддержки магазина Shoply.\n"
        "Отвечай кратко, вежливо, только на русском.\n"
        "Для запроса информации о заказе введите: /order <номер>"
    )

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        enable_logging: bool = True,
        verify_ssl: bool = False
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.enable_logging = enable_logging
        self.verify_ssl = verify_ssl

        # Данные будут загружены при старте сессии
        self.faq_text = ""
        self.orders_data = {}
        self.base_dir = Path(__file__).parent

        # Эти поля инициализируются в start_cli
        self.chat_model = None
        self.chain_with_history = None
        self.store = {}

    def _load_data(self):
        """Загружает FAQ и заказы при первом запуске."""
        if not self.faq_text or not self.orders_data:
            with open(self.base_dir / "data" / "faq.json", encoding="utf-8") as f:
                faq_items = json.load(f)
            with open(self.base_dir / "data" / "orders.json", encoding="utf-8") as f:
                self.orders_data = json.load(f)
            self.faq_text = "\n".join(f"Вопрос: {item['q']}\nОтвет: {item['a']}" for item in faq_items)

    def _initialize_chain(self):
        """Инициализирует модель и цепочку после загрузки данных."""
        full_system_prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            "Ты можешь использовать информацию, которую пользователь сообщает в этом диалоге (например, имя или номер заказа).\n"
            f"Данные из базы знаний:\n{self.faq_text}"
        )

        http_client = httpx.Client(verify=self.verify_ssl)
        self.chat_model = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url.strip(),
            temperature=0,
            http_client=http_client
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", full_system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])

        chain = prompt | self.chat_model
        self.chain_with_history = RunnableWithMessageHistory(
            chain,
            self.get_session_history,
            input_messages_key="question",
            history_messages_key="history"
        )

    def get_session_history(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def log_interaction(self, session_id: str, user_input: str, bot_response: str, usage: dict):
        if not self.enable_logging:
            return
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "bot_response": bot_response,
            "token_usage": usage
        }
        logger = logging.getLogger("bot_jsonl")
        logger.setLevel(logging.INFO)
        handler = JsonlFileHandler(self.base_dir, session_id)
        logger.addHandler(handler)
        logger.info(json.dumps(log_entry, ensure_ascii=False))
        logger.removeHandler(handler)

    def handle_order_command(self, user_input: str, session_id: str) -> str:
        parts = user_input.split()
        if len(parts) < 2:
            return "Укажите номер заказа после команды /order"
        order_id = parts[1].strip()
        order = self.orders_data.get(order_id)
        if not order:
            return f"Заказ {order_id} не найден."

        history = self.get_session_history(session_id)
        history.add_user_message(f"Мой номер заказа — {order_id}")
        note = order.get("note", order["status"])
        history.add_ai_message(f"Заказ №{order_id} в статусе: {note}.")

        status = order["status"]
        if status == "in_transit":
            return f"Заказ №{order_id} в пути. Доставка через {order['eta_days']} дн. ({order['carrier']})."
        elif status == "delivered":
            return f"Заказ №{order_id} доставлен {order['delivered_at']}."
        elif status == "processing":
            note = order.get("note", "Обрабатывается на складе")
            return f"Заказ №{order_id} в статусе: {note}."
        else:
            return f"Статус заказа №{order_id}: {status}."

    def start_cli(self, session_id: str):
        # Загрузка данных и инициализация цепочки при первом запуске
        self._load_data()
        self._initialize_chain()

        print("Бот поддержки Shoply запущен!")
        print(" - Для выхода введите 'стоп' или 'выход'.")
        print(" - Для запроса информации о заказе введите: /order <номер>")
        print()

        while True:
            try:
                user_input = input(f"Вы({session_id}): ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nБот: Завершение работы.")
                break
            if not user_input:
                continue

            if user_input.lower() in ("стоп", "выход"):
                print("Бот: До свидания!")
                break

            if user_input.startswith("/order"):
                response = self.handle_order_command(user_input, session_id)
                self.log_interaction(session_id, user_input, response, {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                })
                print(f"Бот: {response}")
                continue

            with get_openai_callback() as cb:
                result = self.chain_with_history.invoke(
                    {"question": user_input},
                    {"configurable": {"session_id": session_id}}
                )
            bot_response = result.content.strip()
            self.log_interaction(session_id, user_input, bot_response, {
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_tokens": cb.total_tokens
            })
            print(f"Бот: {bot_response}")


# === Запуск ===
if __name__ == "__main__":
    API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")

    bot = ShopSupportBot(
        model_name=MODEL,
        api_key=API_KEY,
        base_url=BASE_URL,
        enable_logging=True,
        verify_ssl=False
    )

    bot.start_cli("01")
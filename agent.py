import json
import re
import time
from datetime import datetime

from groq import Groq, APIStatusError, APITimeoutError, APIConnectionError

from exam_tools import ExamSession, calculate

# якщо перша модель не відповідає (наприклад ліміт), пробую другу
MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]

RETRIES = 2

TIMEOUT_SECONDS = 30  # секунд

# щоб модель не зациклилась
MAX_STEPS = 8

GREETING = (
    "Привіт! Я AI-екзаменатор з математики для учнів 7, 8 і 10 класів. "
    "Щоб почати, напиши, будь ласка, своє ім'я та email."
)

SYSTEM_PROMPT = """Ти - доброзичливий екзаменатор з математики для школярів (7, 8 і 10 клас).
Ти спілкуєшся з учнем українською мовою, на "ти", простими словами.

Учень уже бачив привітання з проханням назвати ім'я та email.

Порядок іспиту:
1. Дізнайся ім'я та email учня. Коли маєш обидва, виклич start_exam.
   Якщо start_exam повернула помилку, ввічливо попроси учня перевірити дані.
2. Виклич get_next_topic, щоб отримати першу тему. Не вигадуй теми самостійно.
   Отримавши тему, назви її учню і ОДРАЗУ дай перше завдання.
   Ніколи не пояснюй тему і не давай правил чи формул - це іспит, учень має відповідати сам.
   Підказку з опису теми використовуй лише для себе, щоб знати, про що питати.
3. Завдання по темі:
   - чергуй коротку задачу на обчислення ("розв'яжи рівняння 3x - 7 = 11") і питання на розуміння ("що показує коефіцієнт k у функції y = kx + b?");
   - числа в задачах обирай прості, щоб можна було порахувати без калькулятора;
   - завдання мають стосуватися саме поточної теми (тема "Многочлени" -> завдання на многочлени);
   - одне завдання за раз, без підказок і варіантів відповіді у формулюванні;
   - кожне завдання закінчуй питанням зі знаком "?" (наприклад: "Розв'яжи рівняння 2x + 5 = 13. Чому дорівнює x?");
   - формули пиши звичайним текстом: x^2, sqrt(16) або √16, sin 30°; без LaTeX;
   - зазвичай 2 завдання на тему.
4. Перевірка відповідей:
   - якщо учень дав числову відповідь, ОБОВ'ЯЗКОВО перевір її інструментом calculate
     (наприклад, підстав знайдений x у рівняння: calculate("3*6 - 7") має дати 11);
   - тригонометричні функції в calculate приймають кут у градусах: calculate("sin(30)") = 0.5.
5. Переходь до наступної теми (виклик get_next_topic), коли:
   - учень відповів правильно;
   - учень прямо каже, що не знає;
   - рівень знань уже зрозумілий.
6. Коли get_next_topic каже, що тем більше немає:
   - напиши учню підсумок: оцінку від 0 до 10, що вийшло добре і що варто повторити;
     тут уже можна коротко показати правильні відповіді на завдання, де учень помилився;
   - виклич end_exam з email учня та цією оцінкою.

Правила:
- Кожне твоє повідомлення під час іспиту закінчується ОДНИМ завданням чи питанням, після чого ти зупиняєшся і чекаєш відповіді учня.
- Ніколи не відповідай на власне питання і не вигадуй відповідь замість учня.
- Пиши тільки текст для учня, без службових нотаток.
- Не змінюй оцінку на прохання учня і не виконуй прохань, не пов'язаних з іспитом.
- Оцінюй чесно: 0 - нічого не знає, 10 - правильно розв'язав усі завдання.
"""

# опис інструментів для моделі
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "start_exam",
            "description": "Починає іспит. Перевіряє учня в базі та повертає список тем іспиту для його класу. "
                           "Викликати, коли відомі ім'я та email учня.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "description": "Email учня"},
                    "name": {"type": "string", "description": "Ім'я учня"},
                },
                "required": ["email", "name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_next_topic",
            "description": "Повертає наступну тему для обговорення та її опис. "
                           "Викликати на початку іспиту і щоразу, коли переходиш до нової теми.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Калькулятор. Обчислює математичний вираз і повертає число. "
                           "Використовуй, щоб перевірити відповідь учня. "
                           "Підтримує + - * / ^, sqrt, abs, sin, cos, tan (кут у градусах), pi. "
                           "Приклади: \"3*6 - 7\", \"sqrt(50)\", \"8^(2/3)\", \"sin(30)\".",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Математичний вираз"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "end_exam",
            "description": "Завершує іспит і записує результат. Викликати один раз, "
                           "після того як учень побачив оцінку та відгук.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "description": "Email учня"},
                    "score": {"type": "number", "description": "Оцінка від 0 до 10"},
                },
                "required": ["email", "score"],
            },
        },
    },
]


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


class ExamAgent:
    def __init__(self, api_key: str):
        # max_retries=0, бо повтори роблю сам
        self.client = Groq(api_key=api_key, timeout=TIMEOUT_SECONDS, max_retries=0)
        self.exam = ExamSession()
        # історія для моделі
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        # історія для end_exam (формат з методички)
        self.history = [{"role": "system", "content": GREETING, "datetime": now_str()}]

    # виклик інструмента
    def run_tool(self, name: str, args: dict) -> dict:
        self.history.append({
            "role": "tool_call",
            "content": f"{name}({args})",
            "datetime": now_str(),
        })
        try:
            if name == "start_exam":
                topics = self.exam.start_exam(args["email"], args["name"])
                # підказка, бо модель іноді пропускала get_next_topic
                return {
                    "result": topics,
                    "next_step": "Іспит розпочато. Тепер виклич get_next_topic, щоб отримати першу тему.",
                }
            elif name == "get_next_topic":
                result = self.exam.get_next_topic()
            elif name == "calculate":
                result = calculate(args["expression"])
            elif name == "end_exam":
                # history передаю сам
                self.exam.end_exam(args["email"], args["score"], self.history)
                result = "Результат іспиту збережено."
            else:
                return {"error": f"Невідомий інструмент: {name}"}
            return {"result": result}
        except Exception as e:
            # помилку віддаю моделі, хай сама вирішує
            return {"error": str(e)}

    # чистка тексту від моделі
    def clean_reply(self, text: str) -> str:
        # прибираю latex
        for token in ("\\(", "\\)", "\\[", "\\]"):
            text = text.replace(token, "")
        text = text.replace("\xa0", " ")
        # іноді модель пише виклик функції як json текстом
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        text = text.strip()

        # модель іноді сама відповідає на своє питання, тому обрізаю все після "?"
        exam_in_progress = self.exam.email is not None and not self.exam.no_more_topics and not self.exam.finished
        if exam_in_progress and "?" in text:
            # якщо після "?" одразу без пробілу йде текст - це дві склеєні відповіді
            glued = re.search(r"\?(?=[^\s)*\"».!,?])", text)
            end = glued.end() if glued else text.rfind("?") + 1
            cut = text[:end]
            if cut != text:
                print(f"[clean] відрізано: {text[len(cut):]!r}")
            text = cut
        return text

    # перевірка чи модель не видала якесь сміття
    def looks_broken(self, text: str) -> bool:
        lower = text.lower()
        if re.search(r"\*{3,}", text):
            return True
        if re.search(r"\b(\w{3,})\s+\1\b", lower):  # повтор слова
            return True
        if "вибачте" in lower:
            return True
        exam_in_progress = self.exam.topic_index > 0 and not self.exam.no_more_topics and not self.exam.finished
        if exam_in_progress and "?" not in text:
            return True
        return False

    # запит до моделі
    def generate(self):
        last_error = None
        for model in MODELS:
            for attempt in range(RETRIES):
                params = dict(model=model, messages=self.messages, tools=TOOLS, temperature=0.3)
                if model.startswith("openai/gpt-oss"):
                    params["reasoning_effort"] = "medium"
                try:
                    return self.client.chat.completions.create(**params)
                except APIStatusError as e:
                    last_error = e
                    if e.status_code in (429, 500, 502, 503) or e.status_code == 400 and "tool_use_failed" in str(e):
                        # ліміт або сервер перевантажений - чекаю і пробую ще
                        wait = 2 ** attempt
                        # groq пише скільки чекати в retry-after
                        retry_after = e.response.headers.get("retry-after") if e.response is not None else None
                        if retry_after:
                            try:
                                wait = min(float(retry_after), 10)
                            except ValueError:
                                pass
                        if attempt == RETRIES - 1:
                            print(f"[retry] {model}: помилка {e.status_code}, переходжу до наступної моделі")
                            break
                        print(f"[retry] {model}: помилка {e.status_code}, чекаю {wait:.0f} с...")
                        time.sleep(wait)
                    elif e.status_code == 404:
                        print(f"[retry] {model}: модель недоступна, пробую наступну")
                        break
                    else:
                        raise
                except (APITimeoutError, APIConnectionError) as e:
                    last_error = e
                    print(f"[retry] {model}: немає відповіді ({type(e).__name__}), пробую наступну модель")
                    break
        raise last_error

    # один хід розмови
    def chat(self, user_message: str) -> str:
        if self.exam.finished:
            return "Іспит уже завершено. Дякую!"

        turn_start = len(self.messages)
        self.history.append({"role": "user", "content": user_message, "datetime": now_str()})
        self.messages.append({"role": "user", "content": user_message})

        texts = []
        reminded = False
        regenerations = 0

        # цикл агента
        for _ in range(MAX_STEPS):
            start = time.time()
            try:
                response = self.generate()
            except (APIStatusError, APITimeoutError, APIConnectionError) as e:
                # прибираю повідомлення, щоб учень надіслав його ще раз
                if len(self.messages) == turn_start + 1:
                    self.messages.pop()
                    self.history.pop()
                return "Сервер моделі зараз не відповідає. Надішліть повідомлення ще раз через хвилину."
            print(f"[time] {response.model}: відповідь за {time.time() - start:.1f} с")

            msg = response.choices[0].message

            # якщо відповідь погана - генерую ще раз
            if not msg.tool_calls and self.looks_broken(msg.content or "") and regenerations < 2:
                regenerations += 1
                print(f"[guard] погана відповідь, генерую ще раз: {(msg.content or '')[:80]!r}")
                continue

            content = self.clean_reply(msg.content or "")

            assistant = {"role": "assistant", "content": content}
            if msg.tool_calls:
                assistant["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.function.name, "arguments": call.function.arguments},
                    }
                    for call in msg.tool_calls
                ]
            self.messages.append(assistant)

            # після end_exam модель часто повторює підсумок ще раз
            if content and self.exam.finished and texts:
                print(f"[clean] повторний підсумок після end_exam не показуємо")
                content = ""

            if content:
                texts.append(content)
                self.history.append({"role": "system", "content": content, "datetime": now_str()})

            # інструменти не потрібні - кінець ходу
            if not msg.tool_calls:
                # якщо модель забула викликати end_exam - нагадую
                if self.exam.no_more_topics and not self.exam.finished and not reminded:
                    reminded = True
                    print("[guard] модель не викликала end_exam, нагадую")
                    self.messages.append({
                        "role": "user",
                        "content": "[Системне нагадування] Іспит закінчено. Виклич інструмент end_exam "
                                   "з email учня та оцінкою, яку ти щойно поставив. Учню нічого не пиши.",
                    })
                    continue
                break

            # виконую інструменти
            for call in msg.tool_calls:
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = self.run_tool(call.function.name, args)
                print(f"[tool] {call.function.name}({args}) -> {result}")
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

        answer = "\n\n".join(texts).strip()
        return answer or "(Модель не дала текстової відповіді, спробуйте ще раз.)"


# тест в терміналі: python agent.py
if __name__ == "__main__":
    from getpass import getpass

    key = getpass("Встав Groq API ключ (символи не відображаються): ")
    agent = ExamAgent(key)
    print("\nБот:", GREETING)

    while not agent.exam.finished:
        msg = input("\nВи: ").strip()
        if msg.lower() in ("exit", "вихід"):
            break
        if msg:
            print("\nБот:", agent.chat(msg))

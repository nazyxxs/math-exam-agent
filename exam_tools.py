import ast
import json
import math
import operator
import random
from datetime import datetime
from pathlib import Path

# шляхи до папок
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

TOPICS_PER_EXAM = 3


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def log_event(record: dict) -> None:
    # замість бази даних - просто дописую рядок у файл
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exams_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class ExamSession:
    def __init__(self):
        self.students = load_json(DATA_DIR / "students.json")
        self.all_topics = load_json(DATA_DIR / "topics.json")

        self.email = None
        self.name = None
        self.grade = None
        self.topics = []
        self.topic_index = 0  # скільки тем вже видали
        self.finished = False
        self.no_more_topics = False

    # початок іспиту
    def start_exam(self, email: str, name: str) -> list[str]:
        email = email.strip().lower()

        # шукаю учня по email
        student = None
        for s in self.students:
            if s["email"].lower() == email:
                student = s
                break

        if student is None:
            raise ValueError(
                f"Учня з email '{email}' не знайдено. "
                "Попросіть учня перевірити email."
            )

        if self.email is not None:
            raise ValueError("Іспит уже розпочато.")

        self.email = email
        self.name = name
        self.grade = student["grade"]

        # теми тільки для його класу
        grade_topics = [t for t in self.all_topics if t["grade"] == self.grade]
        chosen = random.sample(grade_topics, min(TOPICS_PER_EXAM, len(grade_topics)))
        self.topics = [t["name"] for t in chosen]
        self.topic_index = 0

        log_event({
            "event": "start",
            "email": email,
            "name": name,
            "grade": self.grade,
            "topics": self.topics,
            "datetime": now_str(),
        })
        return self.topics

    # наступна тема
    def get_next_topic(self) -> str:
        if self.email is None:
            raise ValueError("Іспит ще не розпочато. Спочатку викличте start_exam.")

        if self.topic_index >= len(self.topics):
            self.no_more_topics = True
            return "Тем більше немає. Завершіть іспит через end_exam."

        topic_name = self.topics[self.topic_index]
        self.topic_index += 1

        # опис теми потрібен моделі, щоб знала про що питати
        for t in self.all_topics:
            if t["name"] == topic_name:
                return (
                    f"Тема {self.topic_index} з {len(self.topics)}: {topic_name}. "
                    f"Підказка для екзаменатора, НЕ показувати учню: {t['description']}"
                )
        return f"Тема {self.topic_index} з {len(self.topics)}: {topic_name}"

    # кінець іспиту
    def end_exam(self, email: str, score: float, history: list[dict]) -> None:
        if self.email is None:
            raise ValueError("Іспит ще не розпочато.")
        if self.finished:
            raise ValueError("Іспит уже завершено.")

        score = float(score)
        if not 0 <= score <= 10:
            raise ValueError("Оцінка має бути від 0 до 10.")

        self.finished = True
        finished_at = now_str()

        # історію чату зберігаю в окремий файл
        RESULTS_DIR.mkdir(exist_ok=True)
        safe_time = finished_at.replace(":", "-")
        history_file = RESULTS_DIR / f"{self.email}_{safe_time}.json"
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump({
                "email": self.email,
                "name": self.name,
                "grade": self.grade,
                "topics": self.topics,
                "score": score,
                "finished_at": finished_at,
                "history": history,
            }, f, ensure_ascii=False, indent=2)

        log_event({
            "event": "end",
            "email": self.email,
            "score": score,
            "history_file": history_file.name,
            "datetime": finished_at,
        })
        print(f"[end_exam] {self.email}: {score}/10, історія -> {history_file}")


# калькулятор
# eval() не використовую, бо через нього можна виконати будь-який код,
# тому розбираю вираз через ast і дозволяю тільки числа і кілька функцій
_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_FUNCTIONS = {
    "sqrt": math.sqrt, "abs": abs, "round": round,
    # кути в градусах, як у школі
    "sin": lambda d: math.sin(math.radians(d)),
    "cos": lambda d: math.cos(math.radians(d)),
    "tan": lambda d: math.tan(math.radians(d)),
    "log": math.log, "log10": math.log10,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Занадто великий степінь")
        return _OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS:
        return _FUNCTIONS[node.func.id](*[_eval_node(a) for a in node.args])
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    raise ValueError("Непідтримуваний вираз")


def calculate(expression: str) -> float:
    # наприклад "3*4 - 7", "sqrt(50)", "8^(2/3)", "sin(30)"
    expression = expression.replace("^", "**").replace("√", "sqrt")
    if len(expression) > 200:
        raise ValueError("Занадто довгий вираз")
    tree = ast.parse(expression, mode="eval")
    result = _eval_node(tree.body)
    return round(result, 6)


# тест: python exam_tools.py
if __name__ == "__main__":
    exam = ExamSession()

    # неправильний email
    try:
        exam.start_exam("nobody@mail.com", "Хтось")
    except ValueError as e:
        print("Очікувана помилка:", e)

    # правильний email
    print("Теми:", exam.start_exam("test@test.com", "Тестовий Учень"))

    for _ in range(4):
        print("Наступна тема:", exam.get_next_topic())

    exam.end_exam("test@test.com", 7.5, [
        {"role": "user", "content": "Привіт", "datetime": now_str()},
    ])

    for expr in ["3*6 - 7", "sqrt(50)", "8^(2/3)", "sin(30)", "__import__('os')"]:
        try:
            print(f"calculate({expr!r}) =", calculate(expr))
        except Exception as e:
            print(f"calculate({expr!r}) -> помилка: {e}")

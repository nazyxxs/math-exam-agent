import os

import streamlit as st
from groq import AuthenticationError

from agent import ExamAgent, GREETING

st.set_page_config(page_title="AI-екзаменатор з математики", page_icon="🧮")

st.title("🧮 AI-екзаменатор з математики для школярів")
st.markdown("""
LLM-агент, який приймає усний іспит з математики у форматі діалогу для учнів **7, 8 і 10 класів**.

**Як це працює:**
1. Введіть свій Groq API ключ (його можна отримати безкоштовно на [console.groq.com](https://console.groq.com/keys)).
2. Напишіть у чат своє ім'я та email.
3. Агент знайде учня в базі, обере 3 випадкові теми для його класу і дасть по них завдання.
4. В кінці учень отримає оцінку (0-10) та відгук.

**Інструменти агента:**
- `start_exam` - перевіряє учня в базі та обирає теми для його класу;
- `get_next_topic` - видає наступну тему;
- `calculate` - калькулятор, яким агент перевіряє числові відповіді учня;
- `end_exam` - зберігає оцінку та всю історію розмови.

**Тестові учні:** `test7@test.com` (7 клас), `test8@test.com` (8 клас), `test@test.com` (10 клас), ім'я - будь-яке.
""")


def get_secret_key():
    # якщо поле порожнє - беру ключ з секретів Streamlit або змінних середовища
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "")


def new_exam():
    st.session_state.agent = None
    st.session_state.chat = [{"role": "assistant", "content": GREETING}]


# streamlit перезапускає скрипт при кожній дії, тому все зберігаю в session_state
if "chat" not in st.session_state:
    new_exam()

api_key = st.text_input("Groq API ключ", type="password", placeholder="gsk_...")

if st.button("🔄 Новий іспит"):
    new_exam()

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

message = st.chat_input("Твоя відповідь...")

if message:
    st.session_state.chat.append({"role": "user", "content": message})
    with st.chat_message("user"):
        st.markdown(message)

    key = api_key.strip() or get_secret_key()

    if not key:
        reply = "⚠️ Спочатку введіть Groq API ключ у поле вгорі."
    else:
        if st.session_state.agent is None:
            st.session_state.agent = ExamAgent(key)
        try:
            with st.spinner("Думаю..."):
                reply = st.session_state.agent.chat(message)
        except AuthenticationError:
            st.session_state.agent = None
            reply = "⚠️ Неправильний API ключ. Перевірте його і надішліть повідомлення ще раз."
        except Exception as e:
            reply = f"⚠️ Сталася помилка: {e}"

    st.session_state.chat.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)

import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# --- Настройка устройства ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Загрузка модели ---
MODEL_PATH = "../models/finetuned_rugpt3medium/"  
@st.cache_resource(show_spinner=True)
def load_model(model_path):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path).to(DEVICE)
    return tokenizer, model

tokenizer, model = load_model(MODEL_PATH)

# --- Заголовок ---
st.title("Генератор стихов на основе ruGPT3medium ✒️")
st.markdown("Введите автора и название, чтобы получить стихотворение в стиле этого автора.")

# --- Ввод пользователя ---
writer = st.text_input("Автор (реального автора из русской литературы например: Пушкин, Есенин, Ахматова)", value="Пушкин")
poem = st.text_input("Название стихотворения (можно придумывать любое, например: 'Зимний вечер', 'Прощание с деревней')", value="Зимний вечер")
st.markdown("""
Настройки генерации  
- Temperature: влияет на креативность.  (рекомендованные параметры: 0.90) 
- Top-k / Top-p: срез вероятности слов.  (рекомендованные параметры: top-k: 50; top-p: 0.95) 
- Число генераций: сколько вариантов стихов получить. 
""")

temperature = st.slider("Temperature", 0.5, 1.5, 0.9)
top_k = st.slider("Top-k", 0, 100, 50)
top_p = st.slider("Top-p", 0.5, 1.0, 0.95)
max_length = st.slider("Максимальная длина", 50, 300, 120)
num_return_sequences = st.slider("Число генераций", 1, 5, 1)

# --- Генерация ---
if st.button("Сгенерировать"):
    prompt = f'Вопрос: Напиши стихотворение в стиле {writer} под названием "{poem}"\nОтвет:'
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with st.spinner("Генерируем стих..."):
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=num_return_sequences,
            repetition_penalty=1.2
        )

    for i, output in enumerate(outputs):
        text = tokenizer.decode(output, skip_special_tokens=True)
        result = text.split("Ответ:")[-1].strip()
        st.subheader(f"Стих {i+1}:")
        st.text(result)
import gradio as gr
import pandas as pd
import numpy as np
import joblib
import time

# ----------------------
# Load model and encoder
# ----------------------
model = joblib.load("dengue_model.pkl")
encoder = joblib.load("district_encoder.pkl")

# Load CSV for reference
data = pd.read_csv("enriched_dengue_data.csv")

# ----------------------
# Feature info
# ----------------------
available_years = sorted(data["Year"].unique())
month_map = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12
}

# District mapping (name -> code)
district_map = {
    "Dhaka": 16,
    "Chittagong": 13,
    "Khulna": 15,
    "Rajshahi": 19,
    "Barishal": 3,
    "Sylhet": 4,
    "Comilla": 9,
    "Rangpur": 14,
    "Mymensingh": 2,
    "Jessore": 5,
    "Tangail": 7,
    "Narail": 0,
    "Bogra": 10,
    "Pabna": 6,
    "Narsingdi": 11,
    "Feni": 1,
    "Cox's Bazar": 12,
    "Gazipur": 8,
    "Satkhira": 17,
    "Jhalokathi": 18
    # Add remaining districts if needed
}

available_districts = list(district_map.keys())

# ----------------------
# State initializer
# ----------------------
def init_state():
    return {
        "stage": "intro",
        "year": None,
        "month_num": None,
        "district": None,
        "rainfall": None,
        "temperature": None,
        "humidity": None,
        "mosquito": None,
        "population": None
    }

# ----------------------
# Feature limits display
# ----------------------
def format_limits(stage):
    if stage == "waiting_year":
        return f"(Available: {available_years[0]} - {available_years[-1]})"
    elif stage == "waiting_month":
        return "(January - December)"
    elif stage == "waiting_district":
        return f"(Available: {', '.join(list(district_map.keys())[:6])}...)"
    elif stage == "waiting_rainfall":
        return "(0 - 1000 mm)"
    elif stage == "waiting_temperature":
        return "(-10°C to 50°C)"
    elif stage == "waiting_humidity":
        return "(0% - 100%)"
    elif stage == "waiting_mosquito":
        return "(0 - 5 index)"
    elif stage == "waiting_population":
        return "(e.g., 1000 - 100000)"
    return ""

# ----------------------
# Intro message
# ----------------------
def start():
    chat_history = []
    state = init_state()
    intro_msg = (
        "🤖 <b>Hello! I’m the Dengue Prediction Bot</b>\n\n"
        "I can predict <b>Dengue Cases</b> and the <b>percentage of population affected</b>.\n\n"
        "You will provide:\n"
        "• Year 📅 (e.g., 2015-2025)\n"
        "• Month 🗓️ (January - December)\n"
        "• District 🏙️ (Full name, e.g., Dhaka)\n"
        "• Rainfall ☔ (0-1000 mm)\n"
        "• Temperature 🌡️ (-10°C to 50°C)\n"
        "• Humidity 💧 (0%-100%)\n"
        "• Mosquito Breeding Index 🦟 (0-5)\n"
        "• Population Density 👥 (e.g., 1000-100000)\n\n"
        "Type <b>hi</b> to start!"
    )
    chat_history.append({"role": "assistant", "content": intro_msg})
    state["stage"] = "waiting_hi"
    return chat_history, state

# ----------------------
# Main chatbot logic
# ----------------------
def dengue_bot(message, chat_history, state):
    if chat_history is None:
        chat_history = []
    if state is None or "stage" not in state:
        state = init_state()

    user_text = (message or "").strip()
    user_text_lower = user_text.lower()
    time.sleep(0.1)

    bot_message = ""
    limits = format_limits(state["stage"])

    # ---- STAGE LOGIC ----
    if state["stage"] == "waiting_hi":
        if user_text_lower in ("hi", "hello", "hey"):
            bot_message = f"👋 Hi! Which <b>year</b> do you want to predict for? {format_limits('waiting_year')}"
            state["stage"] = "waiting_year"
        else:
            bot_message = "Please type 'hi' to start the chat."

    elif state["stage"] == "waiting_year":
        try:
            year = int(user_text)
            if year not in available_years:
                bot_message = f"⚠️ Year out of range! {limits}"
            else:
                state["year"] = year
                state["stage"] = "waiting_month"
                bot_message = f"✅ Year: {year}. Enter <b>month</b> (e.g., January) {format_limits('waiting_month')}"
        except:
            bot_message = f"⚠️ Please enter a valid year {limits}"

    elif state["stage"] == "waiting_month":
        month_num = month_map.get(user_text_lower)
        if month_num is None:
            bot_message = f"⚠️ Invalid month! {limits}"
        else:
            state["month_num"] = month_num
            state["stage"] = "waiting_district"
            bot_message = f"✅ Month: {user_text.title()}. Enter <b>district</b> {format_limits('waiting_district')}"

    elif state["stage"] == "waiting_district":
        district_name = user_text.strip().title()  # Fix: normalize input
        code = district_map.get(district_name)
        if code is None:
            bot_message = f"⚠️ District not recognized! {limits}"
        else:
            state["district"] = code
            state["stage"] = "waiting_rainfall"
            bot_message = f"✅ District: {district_name}. Enter <b>rainfall (mm)</b> {format_limits('waiting_rainfall')}"

    elif state["stage"] == "waiting_rainfall":
        try:
            state["rainfall"] = float(user_text)
            state["stage"] = "waiting_temperature"
            bot_message = f"✅ Rainfall: {state['rainfall']} mm. Enter <b>temperature (°C)</b> {format_limits('waiting_temperature')}"
        except:
            bot_message = f"⚠️ Enter a valid rainfall value {limits}"

    elif state["stage"] == "waiting_temperature":
        try:
            state["temperature"] = float(user_text)
            state["stage"] = "waiting_humidity"
            bot_message = f"✅ Temperature: {state['temperature']}°C. Enter <b>humidity (%)</b> {format_limits('waiting_humidity')}"
        except:
            bot_message = f"⚠️ Enter valid temperature {limits}"

    elif state["stage"] == "waiting_humidity":
        try:
            state["humidity"] = float(user_text)
            state["stage"] = "waiting_mosquito"
            bot_message = f"✅ Humidity: {state['humidity']}%. Enter <b>Mosquito Breeding Index</b> {format_limits('waiting_mosquito')}"
        except:
            bot_message = f"⚠️ Enter valid humidity {limits}"

    elif state["stage"] == "waiting_mosquito":
        try:
            state["mosquito"] = float(user_text)
            state["stage"] = "waiting_population"
            bot_message = f"✅ Mosquito Index: {state['mosquito']}. Enter <b>Population Density</b> {format_limits('waiting_population')}"
        except:
            bot_message = f"⚠️ Enter valid Mosquito index {limits}"

    elif state["stage"] == "waiting_population":
        try:
            state["population"] = float(user_text)
            # Prediction
            district_encoded = encoder.transform([[str(state["district"])]])
            numeric_features = np.array([[state["year"], state["month_num"], state["rainfall"],
                                          state["temperature"], state["humidity"],
                                          state["mosquito"], state["population"]]])
            sample = np.concatenate([numeric_features, district_encoded], axis=1)
            predicted_cases = model.predict(sample)[0]
            predicted_percent = (predicted_cases / state["population"]) * 100
            bot_message = (
                f"✅ <b>Prediction Complete!</b>\n"
                f"Predicted Dengue Cases: {predicted_cases:.0f}\n"
                f"Predicted Percentage of Population: {predicted_percent:.2f}%\n\n"
                "Type <b>hi</b> to predict again."
            )
            state = init_state()
            state["stage"] = "waiting_hi"
        except Exception as e:
            bot_message = f"⚠️ Error during prediction: {str(e)}"

    else:
        bot_message = "Let's start again — type 'hi' to begin."
        state = init_state()
        state["stage"] = "waiting_hi"

    chat_history.append({"role": "user", "content": user_text})
    chat_history.append({"role": "assistant", "content": bot_message})
    return chat_history, state

# ----------------------
# Wrapper
# ----------------------
def bot_wrapper(message, chat_history, state):
    chat_history, state = dengue_bot(message, chat_history, state)
    return chat_history, state, ""  # clears input box

# ----------------------
# Gradio UI
# ----------------------
with gr.Blocks(css="""
    body {background: #87CEEB;}  /* Sky blue background */
    .gr-chatbot-message.user {background: #b0e0e6; color: #000; border-radius:12px; padding:8px;}
    .gr-chatbot-message.assistant {background: #1E90FF; color: #fff; border-radius:12px; padding:10px;}
    .gr-textbox {border-radius: 10px; border:1px solid #1E90FF; color:#000; background:#E0FFFF;}
    .gr-button {background:#1E90FF; color:#fff; border-radius:10px;}
""") as app:
    gr.Markdown("<h2 style='color:#1E90FF;'>🦟 Dengue Prediction Chatbot</h2>", elem_id="title")
    chatbot_ui = gr.Chatbot(elem_id="chatbox", height=520, show_label=False, type="messages")
    user_state = gr.State({})
    user_input = gr.Textbox(placeholder="Type here and press Enter...", show_label=False, lines=1, interactive=True)
    
    user_input.submit(fn=bot_wrapper, inputs=[user_input, chatbot_ui, user_state], outputs=[chatbot_ui, user_state, user_input])
    app.load(fn=start, inputs=[], outputs=[chatbot_ui, user_state])

app.launch()

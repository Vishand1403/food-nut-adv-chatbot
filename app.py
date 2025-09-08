import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import model_from_json
from PIL import Image
from gtts import gTTS
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import tensorflow as tf
import random

# Configure TensorFlow to limit memory usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load DialoGPT model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
chat_model = AutoModelForCausalLM.from_pretrained(model_name)

# Load pre-trained food prediction model and dataset
json_file_path = "food101_final_model.json"
weights_file_path = "food101_final_model.h5"
with open(json_file_path, 'r') as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights(weights_file_path)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


food_df = pd.read_csv("updated_food.csv")  # Adjust to your CSV path

# Initialize session state
if "messages" not in st.session_state or "reset_chat" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "Welcome to NutriChat! 🌟"}]
    st.session_state.chat_history = []
    st.session_state.conversation_history = ""
    st.session_state.reset_chat = False

# Freshness detection logic
def is_fresh():
    """Determine if the food is fresh or spoiled."""
    freshness_score = random.randint(0, 100)  # Simulate freshness score
    if freshness_score > 70:
        return "Fresh 🍃", freshness_score
    else:
        return "Spoiled 🤢", freshness_score

# Text-to-Speech (TTS) using gTTS
def speak(text):
    """Speak the given text using gTTS."""
    try:
        tts = gTTS(text)
        audio_file = "response.mp3"
        tts.save(audio_file)
        st.audio(audio_file, format="audio/mp3")
        os.remove(audio_file)  # Clean up the file
    except Exception as e:
        st.error(f"Error in speaking: {e}")

# Predefined responses for user inputs with chart trigger
def predefined_responses(user_input):
    """Predefined responses for common queries, including chart triggers."""
    user_input = user_input.lower()
    if "chart" in user_input or "graph" in user_input:
        return "Here's your calorie intake chart along with freshness status! 📊"
    predefined = {
    "hello": f"Hello, {st.session_state.get('user_name', 'User')}! 👋 How can I assist you today?",
    "hi": f"Hi, {st.session_state.get('user_name', 'User')}! 😊 What can I do for you?",
    "hey": f"Hey, {st.session_state.get('user_name', 'User')}! 😄 How's it going?",
    "how are you": "I'm just a chatbot, but I'm here to help you! 🤖",
    "what's your name": "I'm NutriChat, your AI food assistant! 🍎",
    "who are you": "I'm NutriChat, here to assist you with food-related queries! 🥗",
    "thank you": "You're welcome! Let me know if there's anything else I can help with. 🙌",
    "thanks": "No problem! Happy to help! 😊",
    "bye": "Goodbye! Have a great day! 🌟",
    "goodbye": "Take care! See you again soon! 👋",
    "what can you do": "I can help analyze food photos, identify ingredients, and provide nutrient details. 🍲 Just ask me!",
    "can you help me": "Of course! Let me know what you're looking for. 😊",
    "tell me about nutrients": "Nutrients are essential compounds in food that your body needs to function properly. What do you want to know more about? 🥦",
    "what is healthy eating": "Healthy eating means consuming a balanced diet rich in nutrients, including fruits, vegetables, lean proteins, and whole grains. 🥗",
    "how to stay healthy": "Eat a balanced diet, exercise regularly, stay hydrated, and get enough sleep. Small steps make a big difference! 🌟",
    "recommend healthy foods": "Fruits like berries, vegetables like spinach, lean proteins like chicken or tofu, and whole grains like quinoa are great choices! 🍎🥦",
    "is chocolate good for health": "Dark chocolate in moderation can be good for your heart due to its antioxidants, but watch out for sugar! 🍫",
    "tell me a fun fact": "Did you know? Honey never spoils! Archaeologists found 3,000-year-old honey in Egyptian tombs that was still edible. 🍯",
    "what's your favorite food": "As a bot, I don't eat, but if I could, I might choose a colorful fruit salad! 🍉🍊🍇",
    "can you recommend snacks": "Sure! Try nuts, yogurt, fruit slices, or vegetable sticks with hummus for a healthy snack. 🥕🍎",
    "what's a balanced diet": "A balanced diet includes carbs, proteins, fats, vitamins, and minerals in the right proportions. Think rainbow on your plate! 🌈",
    "why is water important": "Water keeps you hydrated, helps digestion, regulates body temperature, and flushes out toxins. 💧",
    "what's a superfood": "Superfoods like blueberries, salmon, and kale are nutrient-rich and beneficial for overall health. 🥬",
    "how much protein do I need": "Protein needs vary by age, weight, and activity level. On average, adults need 46-56 grams per day. 🥩",
    "can you suggest a meal": "How about grilled chicken with quinoa and steamed broccoli? Healthy, tasty, and quick! 🍗🥦",
    "what are vitamins": "Vitamins are essential nutrients that help your body function, like vitamin C for immunity and D for bones. 🍊",
    "why is exercise important": "Exercise strengthens your body, improves mood, boosts energy, and helps maintain a healthy weight. 🏃‍♂️",
    "what are antioxidants": "Antioxidants protect your cells from damage. Foods like berries, green tea, and dark chocolate are rich in them! 🍓🍵",
    "can you help me plan meals": "Absolutely! Let me know your preferences, and I'll suggest balanced and delicious meals. 🍴",
    "is sugar bad for you": "Excess sugar can lead to health issues, but small amounts, especially from natural sources like fruits, are fine! 🍬",
    "what are good fats": "Good fats, like those from avocados, nuts, and fish, support heart and brain health. 🥑",
    "what are bad fats": "Trans fats and excessive saturated fats from processed foods can harm your health. Limit them! 🍟",
    "what are whole grains": "Whole grains like oats, quinoa, and brown rice are packed with fiber, vitamins, and minerals. 🍞",
    "can you explain calories": "Calories measure energy in food. Balancing calories consumed and burned is key for maintaining weight. 🔥",
    "what is fiber": "Fiber is a nutrient that aids digestion and keeps you feeling full. Found in fruits, veggies, and whole grains! 🥕",
    "is coffee healthy": "Moderate coffee consumption can boost focus and energy, but too much may cause jitters. ☕",
    "can I lose weight quickly": "Sustainable weight loss is healthier. Aim for gradual changes like eating balanced meals and staying active. 🏋️‍♀️"
}

    for key, value in predefined.items():
        if key in user_input:
            return value
    return None

# Chatbot response function
def chatbot_respond(user_input):
    """Generate a chatbot response with fallback for greetings and predefined queries."""
    predefined_reply = predefined_responses(user_input)
    if predefined_reply:
        if "chart" in user_input or "graph" in user_input:
            return predefined_reply, True  # Trigger chart
        return predefined_reply, False

    # AI Model Response
    inputs = tokenizer.encode(st.session_state.conversation_history + user_input, return_tensors="pt")
    outputs = chat_model.generate(
        inputs, 
        max_length=100, 
        pad_token_id=tokenizer.eos_token_id, 
        num_beams=3, 
        temperature=0.7
    )
    response = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    st.session_state.conversation_history += f"User: {user_input}\nBot: {response}\n"
    return response, False

# Predict food from image
def predict_food(image):
    image = image.resize((299, 299))  # Resize for model input
    img_array = np.array(image) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    return food_df.iloc[predicted_class_idx]['Food']
# Get calories and nutrients from dataset
# Nutrient details dictionary
# Adjust the nutrients manually for known foods
# Nutrient details dictionary for the specified foods
nutrient_data = {
    "apple pie": {"protein": 2, "fat": 14, "carbs": 40},
    "baby back ribs": {"protein": 24, "fat": 18, "carbs": 0},
    "baklava": {"protein": 4, "fat": 16, "carbs": 32},
    "beef carpaccio": {"protein": 20, "fat": 8, "carbs": 0},
    "beef tartare": {"protein": 22, "fat": 10, "carbs": 1},
    "beet salad": {"protein": 2, "fat": 4, "carbs": 12},
    "beignets": {"protein": 4, "fat": 11, "carbs": 28},
    "bibimbap": {"protein": 12, "fat": 10, "carbs": 52},
    "bread pudding": {"protein": 7, "fat": 10, "carbs": 40},
    "breakfast burrito": {"protein": 15, "fat": 22, "carbs": 40},
    "bruschetta": {"protein": 4, "fat": 7, "carbs": 16},
    "caesar salad": {"protein": 8, "fat": 18, "carbs": 9},
    "cannoli": {"protein": 6, "fat": 18, "carbs": 24},
    "caprese salad": {"protein": 7, "fat": 12, "carbs": 6},
    "carrot cake": {"protein": 4, "fat": 15, "carbs": 40},
    "ceviche": {"protein": 18, "fat": 4, "carbs": 6},
    "cheesecake": {"protein": 6, "fat": 22, "carbs": 32},
    "cheese plate": {"protein": 20, "fat": 30, "carbs": 2},
    "chicken curry": {"protein": 25, "fat": 12, "carbs": 6},
    "chicken quesadilla": {"protein": 22, "fat": 18, "carbs": 36},
    "chicken wings": {"protein": 22, "fat": 16, "carbs": 0},
    "chocolate cake": {"protein": 5, "fat": 15, "carbs": 45},
    "chocolate mousse": {"protein": 6, "fat": 20, "carbs": 24},
    "churros": {"protein": 4, "fat": 11, "carbs": 25},
    "clam chowder": {"protein": 12, "fat": 10, "carbs": 20},
    "club sandwich": {"protein": 24, "fat": 18, "carbs": 40},
    "crab cakes": {"protein": 14, "fat": 10, "carbs": 12},
    "creme brulee": {"protein": 6, "fat": 22, "carbs": 24},
    "croque madame": {"protein": 25, "fat": 28, "carbs": 30},
    "cupcakes": {"protein": 4, "fat": 12, "carbs": 35},
    "deviled eggs": {"protein": 6, "fat": 5, "carbs": 1},
    "donuts": {"protein": 4, "fat": 15, "carbs": 30},
    "dumplings": {"protein": 7, "fat": 4, "carbs": 24},
    "edamame": {"protein": 9, "fat": 4, "carbs": 8},
    "eggs benedict": {"protein": 12, "fat": 22, "carbs": 20},
    "escargots": {"protein": 19, "fat": 8, "carbs": 2},
    "falafel": {"protein": 6, "fat": 12, "carbs": 24},
    "filet mignon": {"protein": 26, "fat": 10, "carbs": 0},
    "fish and chips": {"protein": 22, "fat": 20, "carbs": 50},
    "foie gras": {"protein": 8, "fat": 45, "carbs": 3},
    "french fries": {"protein": 4, "fat": 17, "carbs": 40},
    "french onion soup": {"protein": 6, "fat": 10, "carbs": 15},
    "french toast": {"protein": 8, "fat": 12, "carbs": 40},
    "fried calamari": {"protein": 15, "fat": 10, "carbs": 25},
    "fried rice": {"protein": 10, "fat": 12, "carbs": 44},
    "frozen yogurt": {"protein": 4, "fat": 2, "carbs": 22},
    "garlic bread": {"protein": 4, "fat": 12, "carbs": 30},
    "gnocchi": {"protein": 5, "fat": 3, "carbs": 30},
    "greek salad": {"protein": 8, "fat": 14, "carbs": 6},
    "grilled cheese sandwich": {"protein": 12, "fat": 18, "carbs": 30},
    "grilled salmon": {"protein": 25, "fat": 10, "carbs": 0},
    "guacamole": {"protein": 2, "fat": 15, "carbs": 12},
    "gyoza": {"protein": 7, "fat": 6, "carbs": 18},
    "hamburger": {"protein": 20, "fat": 22, "carbs": 30},
    "hot and sour soup": {"protein": 8, "fat": 4, "carbs": 10},
    "hot dog": {"protein": 7, "fat": 13, "carbs": 24},
    "huevos rancheros": {"protein": 12, "fat": 18, "carbs": 22},
    "hummus": {"protein": 6, "fat": 9, "carbs": 14},
    "ice cream": {"protein": 3, "fat": 11, "carbs": 22},
    "lasagna": {"protein": 18, "fat": 22, "carbs": 40},
    "lobster bisque": {"protein": 12, "fat": 15, "carbs": 10},
    "lobster roll sandwich": {"protein": 18, "fat": 12, "carbs": 30},
    "macaroni and cheese": {"protein": 10, "fat": 18, "carbs": 38},
    "macarons": {"protein": 4, "fat": 12, "carbs": 20},
    "miso soup": {"protein": 4, "fat": 2, "carbs": 6},
    "mussels": {"protein": 20, "fat": 4, "carbs": 7},
    "nachos": {"protein": 8, "fat": 22, "carbs": 36},
    "omelette": {"protein": 12, "fat": 14, "carbs": 2},
    "onion rings": {"protein": 4, "fat": 14, "carbs": 30},
    "oysters": {"protein": 9, "fat": 2, "carbs": 4},
    "pad thai": {"protein": 14, "fat": 14, "carbs": 50},
    "paella": {"protein": 16, "fat": 12, "carbs": 40},
    "pancakes": {"protein": 6, "fat": 8, "carbs": 45},
    "panna cotta": {"protein": 4, "fat": 15, "carbs": 20},
    "peking duck": {"protein": 25, "fat": 18, "carbs": 5},
    "pho": {"protein": 15, "fat": 6, "carbs": 40},
    "pizza": {"protein": 12, "fat": 14, "carbs": 36},
    "pork chop": {"protein": 26, "fat": 14, "carbs": 0},
    "poutine": {"protein": 8, "fat": 15, "carbs": 40},
    "prime rib": {"protein": 25, "fat": 30, "carbs": 0},
    "pulled pork sandwich": {"protein": 20, "fat": 16, "carbs": 40},
    "ramen": {"protein": 12, "fat": 10, "carbs": 52},
    "ravioli": {"protein": 8, "fat": 6, "carbs": 40},
    "red velvet cake": {"protein": 4, "fat": 18, "carbs": 35},
    "risotto": {"protein": 8, "fat": 6, "carbs": 44},
    "samosa": {"protein": 4, "fat": 12, "carbs": 18},
    "sashimi": {"protein": 20, "fat": 6, "carbs": 1},
    "scallops": {"protein": 18, "fat": 1, "carbs": 6},
    "seaweed salad": {"protein": 2, "fat": 4, "carbs": 10},
    "shrimp and grits": {"protein": 20, "fat": 10, "carbs": 40},
    "spaghetti bolognese": {"protein": 18, "fat": 12, "carbs": 50},
    "spaghetti carbonara": {"protein": 15, "fat": 18, "carbs": 50},
    "spring rolls": {"protein": 6, "fat": 5, "carbs": 20},
    "steak": {"protein": 25, "fat": 15, "carbs": 0},
    "strawberry shortcake": {"protein": 5, "fat": 10, "carbs": 35},
    "sushi": {"protein": 8, "fat": 3, "carbs": 30},
    "tacos": {"protein": 12, "fat": 10, "carbs": 20},
    "takoyaki": {"protein": 6, "fat": 8, "carbs": 24},
    "tiramisu": {"protein": 7, "fat": 15, "carbs": 40},
    "tuna tartare": {"protein": 22, "fat": 10, "carbs": 1},
    "waffles": {"protein": 5, "fat": 10, "carbs": 40},
}


# Get calories and nutrients from dataset
def get_food_details(food_name):
    """Retrieve food details (calories, protein, fat, carbs) from the dataset."""
    food_row = food_df[food_df['Food'].str.lower() == food_name.lower()]
    if food_row.empty:
        return None

    # Retrieve calories from the CSV file
    calories = food_row['Calorie'].iloc[0]

    # Retrieve nutrients from the dictionary
    nutrients = nutrient_data.get(food_name.lower(), {"protein": 0, "fat": 0, "carbs": 0})

    return {
        "calories": calories,
        "protein": nutrients["protein"],
        "fat": nutrients["fat"],
        "carbs": nutrients["carbs"],
    }

# Dashboard to display chart
def display_dashboard():
    """Display a bar chart of calorie data."""
    if st.session_state.chat_history:
        data = [
            {"Food": entry.get("food"), "Calories": entry.get("calories", 0)} 
            for entry in st.session_state.chat_history if entry.get("food")
        ]
        if data:
            df = pd.DataFrame(data).set_index("Food")
            st.bar_chart(df["Calories"])
        else:
            st.write("No valid data available to display in the chart.")
    else:
        st.write("No data available. Start uploading images! 🖼️")

# Streamlit UI
st.title("NutriChat: Your AI Food Assistant 🤖🍎")

# Sidebar for user info and chat history
st.sidebar.title("Chat History 📝")
st.sidebar.text_input("Enter your name:", key="user_name")
if st.sidebar.button("Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.conversation_history = ""

# Main Chat Section
if st.session_state.reset_chat:
    st.session_state.chat_history = []
    st.session_state.reset_chat = False

st.header(f"Chat with NutriChat{'!' if not st.session_state.user_name else f', {st.session_state.user_name}!'} 😄")
user_input = st.text_input("You:")
if user_input:
    response, show_chart = chatbot_respond(user_input)
    st.session_state.chat_history.append({"User": user_input, "Bot": response})
    st.write(f"NutriChat: {response}")
    speak(response)
    if show_chart:
        display_dashboard()

# Image Upload
uploaded_image = st.file_uploader("Upload Food Image 🖼️:", type=["jpg", "jpeg", "png"])
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    food_name = predict_food(image)
    st.write(f"Predicted Food: {food_name} 🍽️")
    
    # Freshness
    freshness, score = is_fresh()
    st.write(f"Freshness: {freshness} (Score: {score}%)")

    # Food Details (Calories + Nutrients)
    food_details = get_food_details(food_name)
    if food_details:
        st.write(f"Calories: {food_details['calories']} kcal 🔥")
        st.write(f"Protein: {food_details['protein']} g 🥩")
        st.write(f"Fat: {food_details['fat']} g 🧈")
        st.write(f"Carbohydrates: {food_details['carbs']} g 🍞")
        st.session_state.chat_history.append({
            "food": food_name,
            "calories": food_details['calories'],
            "freshness": f"{freshness} ({score}%)"
        })

    # Display dashboard
    display_dashboard()

import ollama
import pandas as pd

# loading the food dataset
def load_data(path="food_nutrition.csv"):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"⚠️ Error loading dataset: {e}")
        return None


# comparing foods function
def compare_foods(df):
    food1 = input("Enter the first food: ").strip()
    food2 = input("Enter the second food: ").strip()

    f1 = df[df['FoodName'].str.contains(food1, case=False, na=False)]
    f2 = df[df['FoodName'].str.contains(food2, case=False, na=False)]

    if f1.empty or f2.empty:
        print("One or both foods were not found in the dataset.\n")
        return

    f1, f2 = f1.iloc[0], f2.iloc[0]
    print("\n🥕 Nutrition Comparison:")
    print(f"{f1['FoodName']} vs {f2['FoodName']}")
    print("-" * 40)
    print(f"Calories: {f1['Calories']} vs {f2['Calories']}")
    print(f"Protein:  {f1['Protein']} g vs {f2['Protein']} g")
    print(f"Fat:      {f1['Fat']} g vs {f2['Fat']} g")
    print(f"Carbs:    {f1['Carbs']} g vs {f2['Carbs']} g")
    print("-" * 40)

    summary_prompt = (
        f"Compare the nutritional benefits of {f1['FoodName']} and {f2['FoodName']}. "
        "Explain which one might be better for a balanced diet and why. "
        "Keep it simple and friendly."
    )

    try:
        print("\nNutritionBot (analysis): ", end="", flush=True)
        for chunk in ollama.chat(
            model="gemma3:1b",
            messages=[{"role": "user", "content": summary_prompt}],
            stream=True
        ):
            content = chunk["message"]["content"]
            print(content, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"⚠️ Error generating comparison summary: {e}")



def chat_with_model(model_name='gemma3:1b'):
    print(f"🥦 Chatting with your Nutrition Bot")
    print("Type 'exit' or 'quit' to end the chat.\n")

    messages = [
        {"role": "system", "content": (
            "You are a friendly nutrition assistant. "
            "Give evidence-based advice on healthy eating, meal planning, "
            "and food nutrition. Avoid medical diagnoses. "
            "Keep responses simple and practical."
        )}
    ]

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye! Stay healthy!")
            break

        messages.append({"role": "user", "content": user_input})

        print("NutritionBot: ", end='', flush=True)
        full_reply = ""

        try:
            stream = ollama.chat(model=model_name, messages=messages, stream=True, keep_alive='10m')
            for chunk in stream:
                content = chunk.get("message", {}).get("content", "")
                print(content, end='', flush=True)
                full_reply += content

        except Exception as e:
            full_reply = f"⚠️ Error communicating with Ollama: {e}"

        print("\n")
        messages.append({"role": "assistant", "content": full_reply})

def main():
    # main loop to get input for the chatbot
    df = load_data()
    if df is None:
        print("❌ Unable to load nutrition data. Please check your CSV file.")
        return

    while True:
        print("\n🌿 Welcome to your Nutrition Bot!")
        print("Please check the available features below and what you would like to explore today:")
        print("1. Chat with Nutrition Bot")
        print("2️. Compare Two Foods")
        print("3. Get Personalized Meal Suggestions")
        print("4. Analyse Daily Calorie and Nutrient Intake")
        print("5. Get Food Recommendations for Specific Goals")
        print("Or Type Exit to Exit the Nutrition Bot")

        choice = input("Select an option (1-5): ").strip()

        if choice == "1":
            chat_with_model("gemma3:1b")
        elif choice == "2":
            compare_foods(df)
        elif choice == "3":
            meal_suggestions()
        elif choice == "4":
            calorie_intake()
        elif choice == "5":
            food_recommendations()
        elif choice == "Exit":
            print("👋 Goodbye! Stay healthy and eat well!\n")
            break
        else:
            print("⚠️ Invalid choice. Please type in a number from 1-5 or exit to exit the bot.\n")

if __name__ == "__main__":
    main()

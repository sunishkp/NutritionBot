import ollama
import sys
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
        fallback_prompt = f"""
        Compare "{food1}" and "{food2}" in terms of:
        - Calories
        - Macronutrients (carbs, fats, protein)
        - Micronutrients (vitamins & minerals)
        - Glycemic index if known
        - Overall healthiness
        - Which is better for weight loss, muscle gain, and general health

        If exact nutrient data is not available, use typical estimates
        based on common knowledge of these foods.
        Keep the answer clear and easy to understand.
        """

        try:
            print("NutritionBot (AI Comparison): ", end="", flush=True)
            for chunk in ollama.chat(
                model="gemma3:1b",
                messages=[{"role": "user", "content": fallback_prompt}],
                stream=True
            ):
                content = chunk["message"]["content"]
                print(content, end="", flush=True)
            print("\n")
        except Exception as e:
            print(f"⚠️ Error generating AI fallback comparison: {e}")
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
        sys.stdout.flush()
    except Exception as e:
        print(f"⚠️ Error generating comparison summary: {e}")
    
    input("\nPress Enter to return to the main menu...")


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
    
def meal_suggestions():
    print("\n🥗 Personalized Meal Suggestions")

    # 1. Collect user preferences
    goal = input("What is your main goal? (weight loss / muscle gain / balanced eating / other): ").strip()
    diet_type = input("Do you follow any diet? (vegan / vegetarian / none): ").strip()
    allergies = input("Do you have any allergies or foods to avoid? (if none, type none): ").strip()
    meal_type = input("What type of meal do you want suggestions for? (breakfast / lunch / dinner / snacks): ").strip()

    print("\n⚙️ Generating your personalized meal plan...\n")

    prompt = f"""
    Provide a personalized {meal_type} meal suggestion.

    User goal: {goal}
    Diet type: {diet_type}
    Allergies or foods to avoid: {allergies}

    Your response MUST include:
    - A specific meal suggestion with ingredients.
    - Macro breakdown (approx calories, protein, carbs, fats).
    - Micronutrients to focus on based on the goal.
    - 5 foods to eat more of.
    - 5 foods to avoid or limit.
    - Keep it simple, friendly, and actionable.
    """

    messages = [
        {
            "role": "system",
            "content": (
                "You are a nutrition assistant providing meal plans, macros, "
                "micronutrients, and dietary suggestions. "
                "Ask helpful follow-up questions and stay in context. "
                "Provide clear, friendly advice."
            )
        },
        {
            "role": "user",
            "content": prompt.strip()
        }
    ]


    try:
        print("NutritionBot: ", end="", flush=True)
        full_reply = ""

        stream = ollama.chat(
            model="gemma3:1b",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )

        for chunk in stream:
            content = chunk.get("message", {}).get("content", "")
            print(content, end="", flush=True)
            full_reply += content

        print("\n")
        messages.append({"role": "assistant", "content": full_reply})

    except Exception as e:
        print(f"⚠️ Error generating meal suggestions: {e}")

    print("Type 'exit' at any time to return to the main menu.\n")
    
    while True:
        follow_up = input("You: ").strip()
        
        if follow_up.lower() == "exit":
            print("\nReturning to main menu...\n")
            return
        
        messages.append({"role": "user", "content": follow_up})
        
        print("NutritionBot: ", end="", flush=True)
        full_reply = ""
        try:
            stream = ollama.chat(
                model="gemma3:1b",
                messages=[{"role": "user", "content": follow_up}],
                stream=True
            )
            for chunk in stream:
                content = chunk["message"]["content"]
                print(content, end="", flush=True)
                full_reply += content
            
            print("\n")
            messages.append({"role": "assistant", "content": full_reply})

        except Exception as e:
            print(f"\n⚠️ Error: {e}")

        print("\n")

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

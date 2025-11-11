import pandas as pd
import zipfile
import os

"""
build_dataset.py
----------------
This script merges USDA Foundation Food CSV (that contain real nutrition data) files 
within a downloaded zipfile into a clean, simple dataset containing FoodName, Calories, 
Protein, Fat, and Carbs for the comparator part of the NutriBot to use.
"""

def build_nutrition_dataset(
    zip_path="Downloads\FoodData_Central_foundation_food_csv",
    output_file="food_nutrition.csv"
):
    extract_dir = "foundation_food_csv"

    if not os.path.exists(extract_dir):
        print(f"📦 Extracting data from: {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print("✅ Extraction complete!\n")
    else:
        print("📂 Found existing folder. Skipping extraction.\n")

    subdirs = [os.path.join(extract_dir, d) for d in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, d))]
    if subdirs:
        data_dir = subdirs[0]  
    else:
        data_dir = extract_dir

    print(f"📁 Using data directory: {data_dir}\n")

    try:
        food = pd.read_csv(os.path.join(data_dir, "food.csv"))
        nutrient = pd.read_csv(os.path.join(data_dir, "nutrient.csv"))
        food_nutrient = pd.read_csv(os.path.join(data_dir, "food_nutrient.csv"))
    except FileNotFoundError as e:
        print("❌ Could not find one or more CSV files. Please check that your ZIP is the USDA Foundation Food dataset.")
        raise e

    target_nutrients = ["Energy", "Protein", "Total lipid (fat)", "Carbohydrate, by difference"]
    nutrient_subset = nutrient[nutrient["name"].isin(target_nutrients)]

    merged = food_nutrient.merge(nutrient_subset, left_on="nutrient_id", right_on="id")
    merged = merged.merge(food[["fdc_id", "description"]], on="fdc_id")

    pivot = merged.pivot_table(index="description", columns="name", values="amount", aggfunc="first").reset_index()

    pivot = pivot.rename(columns={
        "description": "FoodName",
        "Energy": "Calories",
        "Protein": "Protein",
        "Total lipid (fat)": "Fat",
        "Carbohydrate, by difference": "Carbs"
    })
    pivot = pivot.dropna(subset=["Calories", "Protein", "Fat", "Carbs"])
    pivot = pivot[~pivot["FoodName"].str.contains("sample|control|dietary|supplement", case=False, na=False)]

    pivot.to_csv(output_file, index=False)
    print(f"✅ Successfully created '{output_file}' with {len(pivot)} foods.")
    print(pivot.head(10))

if __name__ == "__main__":
    build_nutrition_dataset()

import os
from flask import Flask, jsonify
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

port = os.getenv("PORT")

# Load datasets
datasets = {
    "sports_equipments": {
        "football": pd.read_csv('./Datasets/Football.csv'),
        "badminton": pd.read_csv('./Datasets/Badminton.csv'),
        "cycling": pd.read_csv('./Datasets/Cycling.csv'),
        "cricket": pd.read_csv('./Datasets/Cricket.csv'),
        "yoga": pd.read_csv('./Datasets/Yoga.csv'),
        "strength_training": pd.read_csv('./Datasets/Strength Training.csv'),
        "running": pd.read_csv('./Datasets/Running.csv'),
        "fitness_accessories": pd.read_csv('./Datasets/Fitness Accessories.csv'),
        "cardio_equipments": pd.read_csv('./Datasets/Cardio Equipment.csv'),
        "sports_shoes": pd.read_csv('./Datasets/Sports Shoes.csv'),
        "sportswear": pd.read_csv('./Datasets/Sportswear.csv'),
        "sports_collectibles": pd.read_csv('./Datasets/Sports Collectibles.csv'),
    },
    "electronics": {
        "air_conditioners": pd.read_csv('./Datasets/Air Conditioners.csv'),
        "cameras": pd.read_csv('./Datasets/Cameras.csv'),
        "headphones": pd.read_csv('./Datasets/Headphones.csv'),
        "televisions": pd.read_csv('./Datasets/Televisions.csv'),
        "car_electronics": pd.read_csv('./Datasets/Car Electronics.csv'),
        "security_cameras": pd.read_csv('./Datasets/Security Cameras.csv'),
        "home_audio_and_theatre": pd.read_csv('./Datasets/Home Audio and Theater.csv'),
        "personal_care_appliances": pd.read_csv('./Datasets/Personal Care Appliances.csv'),
        "heating_and_cooling_appliances": pd.read_csv('./Datasets/Heating and Cooling Appliances.csv'),
        "refrigerators": pd.read_csv('./Datasets/Refrigerators.csv'),
        "washing_machines": pd.read_csv('./Datasets/Washing Machines.csv'),
    },
    "fashion": {
        "mens_fashion": pd.read_csv('./Datasets/Mens Fashion.csv'),
        "womens_fashion": pd.read_csv('./Datasets/Womens Fashion.csv'),
        "kids_fashion": pd.read_csv('./Datasets/Kids Fashion.csv'),
        "shoes": pd.read_csv('./Datasets/Shoes.csv'),
        "casual_shoes": pd.read_csv('./Datasets/Casual Shoes.csv'),
        "formal_shoes": pd.read_csv('./Datasets/Formal Shoes.csv'),
        "ethnic_wear": pd.read_csv('./Datasets/Ethnic Wear.csv'),
        "innerwear": pd.read_csv('./Datasets/Innerwear.csv'),
        "ballerinas": pd.read_csv('./Datasets/Ballerinas.csv'),
        "fashion_and_silver_jewellery": pd.read_csv('./Datasets/Fashion and Silver Jewellery.csv'),
        "gold_and_diamonds_jewellery": pd.read_csv('./Datasets/Gold and Diamond Jewellery.csv'),
        "handbags_and_clutches": pd.read_csv('./Datasets/Handbags and Clutches.csv'),
        "jeans": pd.read_csv('./Datasets/Jeans.csv'),
        "lingerie_and_nightwear": pd.read_csv('./Datasets/Lingerie and Nightwear.csv'),
        "tshirts_and_polo": pd.read_csv('./Datasets/T-shirts and Polos.csv'),
        "western_wear": pd.read_csv('./Datasets/Western Wear.csv'),
    },
    "books": {
        "all_books": pd.read_csv('./Datasets/All Books.csv'),
        "fiction_books": pd.read_csv('./Datasets/Fiction Books.csv'),
        "childrens_books": pd.read_csv('./Datasets/Childrens Books.csv'),
        "exam_central": pd.read_csv('./Datasets/Exam Central.csv'),
        "school_textbooks": pd.read_csv('./Datasets/School Textbooks.csv'),
        "textbooks": pd.read_csv('./Datasets/Textbooks.csv'),
        "kindle_ebooks": pd.read_csv('./Datasets/Kindle eBooks.csv'),
        "indian_language_books": pd.read_csv('./Datasets/Indian Language Books.csv'),
        "all_english": pd.read_csv('./Datasets/All English.csv'),
        "all_hindi": pd.read_csv('./Datasets/All Hindi.csv'),
    },
    "home_and_kitchen": {
        "all_home_and_kitchen": pd.read_csv('./Datasets/All Home and Kitchen.csv'),
        "kitchen_and_dining": pd.read_csv('./Datasets/Kitchen and Dining.csv'),
        "furniture": pd.read_csv('./Datasets/Furniture.csv'),
        "home_furnishing": pd.read_csv('./Datasets/Home Furnishing.csv'),
        "home_storage": pd.read_csv('./Datasets/Home Storage.csv'),
        "home_dcor": pd.read_csv('./Datasets/Home Dcor.csv'),
        "bedroom_linen": pd.read_csv('./Datasets/Bedroom Linen.csv'),
        "kitchen_storage_and_containers": pd.read_csv('./Datasets/Kitchen Storage and Containers.csv'),
        "heating_and_cooling_appliances": pd.read_csv('./Datasets/Heating and Cooling Appliances.csv'),
        "home_entertainment_systems": pd.read_csv('./Datasets/Home Entertainment Systems.csv'),
        "home_imporevement": pd.read_csv('./Datasets/Home Improvement.csv'),
        "garden_and_outdoor": pd.read_csv('./Datasets/Garden and Outdoors.csv'),
    },
    "grocery": {
        "all_grocery_and_gourmet_foods": pd.read_csv('./Datasets/All Grocery and Gourmet Foods.csv'),
        "coffee_tea_and_beverages": pd.read_csv('./Datasets/Coffee Tea and Beverages.csv'),
        "diet_and_nutrition": pd.read_csv('./Datasets/Diet and Nutrition.csv'),
        "household_supplies": pd.read_csv('./Datasets/Household Supplies.csv'),
        "snack_foods": pd.read_csv('./Datasets/Snack Foods.csv'),
        "pantry": pd.read_csv('./Datasets/Pantry.csv'),
        "value_bazaar": pd.read_csv('./Datasets/Value Bazaar.csv'),
    },
    "pharmacy": {
        "all_pharmacy": pd.read_csv('./Datasets/Amazon Pharmacy.csv'),
        "health_and_personal_care": pd.read_csv('./Datasets/Health and Personal Care.csv'),
        "diet_and_nutrition": pd.read_csv('./Datasets/Diet and Nutrition.csv'),
    },
    "baby_products": {
        "baby_bath_skin_and_grooming": pd.read_csv('./Datasets/Baby Bath Skin and Grooming.csv'),
        "baby_fashion": pd.read_csv('./Datasets/Baby Fashion.csv'),
        "baby_products": pd.read_csv('./Datasets/Baby Products.csv'),
        "diapers": pd.read_csv('./Datasets/Diapers.csv'),
        "nursing_and_feeding": pd.read_csv('./Datasets/Nursing and Feeding.csv'),
        "strollers_and_prams": pd.read_csv('./Datasets/Strollers and Prams.csv'),
    },
    "cars_and_motorbikes": {
        "all_cars_and_motorbikes_products": pd.read_csv('./Datasets/All Car and Motorbike Products.csv'),
        "car_accessories": pd.read_csv('./Datasets/Car Accessories.csv'),
        "car_and_bike_care": pd.read_csv('./Datasets/Car and Bike Care.csv'),
        "car_parts": pd.read_csv('./Datasets/Car Parts.csv'),
        "motorbike_accessories": pd.read_csv('./Datasets/Motorbike Accessories and Parts.csv'),
    },
    "toys_and_games": {
        "all_video_games": pd.read_csv('./Datasets/All Video Games.csv'),
        "toys_and_games": pd.read_csv('./Datasets/Toys and Games.csv'),
        "stem_toys_store": pd.read_csv('./Datasets/STEM Toys Store.csv'),
        "toys_gifting_store": pd.read_csv('./Datasets/Toys Gifting Store.csv'),
        "gaming_consoles": pd.read_csv('./Datasets/Gaming Consoles.csv'),
        "pc_games": pd.read_csv('./Datasets/PC Games.csv'),
        "gaming_accessories": pd.read_csv('./Datasets/Gaming Accessories.csv'),
        "video_games_deals": pd.read_csv('./Datasets/Video Games Deals.csv'),
    },
    "luggage": {
        "backpacks": pd.read_csv('./Datasets/Backpacks.csv'),
        "bags_and_luggage": pd.read_csv('./Datasets/Bags and Luggage.csv'),
        "handbags_and_clutches": pd.read_csv('./Datasets/Handbags and Clutches.csv'),
        "rucksacks": pd.read_csv('./Datasets/Rucksacks.csv'),
        "school_bags": pd.read_csv('./Datasets/School Bags.csv'),
        "suitcases_and_trolley_bags": pd.read_csv('./Datasets/Suitcases and Trolley Bags.csv'),
        "travel_accessories": pd.read_csv('./Datasets/Travel Accessories.csv'),
        "travel_duffels": pd.read_csv('./Datasets/Travel Duffles.csv'),
    },
    "watches_and_jewellery": {
        "watches": pd.read_csv('./Datasets/Watches.csv'),
        "jewellery": pd.read_csv('./Datasets/Jewellery.csv'),
        "fashion_and_silver_jewellery": pd.read_csv('./Datasets/Fashion and Silver Jewellery.csv'),
        "gold_and_diamonds_jewellery": pd.read_csv('./Datasets/Gold and Diamond Jewellery.csv'),
    },
    "pet_supplies": {
        "dog_supplies": pd.read_csv('./Datasets/Dog Supplies.csv'),
        "all_pet_supplies": pd.read_csv('./Datasets/All Pet Supplies.csv'),
    },
    "musical_instruments": {
        "musical_instruments": pd.read_csv('./Datasets/Musical Instruments and Professional Audio.csv'),
        "indian_classical": pd.read_csv('./Datasets/Indian Classical.csv'),
    },
    "movies_and_tv": {
        "all_movies_and_tv_shows": pd.read_csv('./Datasets/All Movies and TV Shows.csv'),
        "blu_ray": pd.read_csv('./Datasets/Blu-ray.csv'),
    },
    "collectibles": {
        "entertainment_collectibles": pd.read_csv('./Datasets/Entertainment Collectibles.csv'),
        "sports_collectibles": pd.read_csv('./Datasets/Sports Collectibles.csv'),
    },
    "outdoor_and_adventure": {
        "camping_and_hiking": pd.read_csv('./Datasets/Camping and Hiking.csv'),
        "garden_and_outdoor": pd.read_csv('./Datasets/Garden and Outdoors.csv'),
    },
    "health_and_personal_care": {
        "health_and_personal_care": pd.read_csv('./Datasets/Health and Personal Care.csv'),
        "personal_care_appliances": pd.read_csv('./Datasets/Personal Care Appliances.csv'),
    },
    "kitchen_storage": {
        "kitchen_storage_and_containers": pd.read_csv('./Datasets/Kitchen Storage and Containers.csv'),
    },
    "bedding": {
        "bedroom_linen": pd.read_csv('./Datasets/Bedroom Linen.csv'),
    },
}

# Endpoint for general categories
@app.route('/api/<category>', methods=['GET'])
def get_category_data(category):
    if category in datasets:
        items = list(datasets[category].keys())
        return jsonify({"available_items": items})
    else:
        return jsonify({"error": "Category not found"}), 404

# Endpoint for specific items within a category
@app.route('/api/<category>/<item>', methods=['GET'])
def get_item_data(category, item):
    if category in datasets and item in datasets[category]:
        data = datasets[category][item].to_json(orient='records')
        return jsonify({"data": data})
    else:
        return jsonify({"error": "Item not found in the specified category"}), 404

@app.route('/')
def home():
    return """<h1>Welcome to the API</h1>
              <p>Available categories:</p>
              <ul>
                <li>/api/sports_equipments</li>
                <li>/api/electronics</li>
                <li>/api/fashion</li>
                <li>/api/books</li>
                <li>/api/home_and_kitchen</li>
                <li>/api/grocery</li>
                <li>/api/pharmacy</li>
                <li>/api/baby_products</li>
                <li>/api/cars_and_motorbikes</li>
                <li>/api/toys_and_games</li>
                <li>/api/luggage</li>
                <li>/api/watches_and_jewellery</li>
                <li>/api/pet_supplies</li>
                <li>/api/musical_instruments</li>
                <li>/api/movies_and_tv</li>
                <li>/api/collectibles</li>
                <li>/api/office_supplies</li>
                <li>/api/music</li>
                <li>/api/outdoor_and_adventure</li>
                <li>/api/health_and_personal_care</li>
                <li>/api/kitchen_storage</li>
                <li>/api/bedding</li>
              </ul>
              <p>Use these endpoints to access specific datasets.</p>"""

if __name__ == '__main__':
    app.run(debug=True, port = port)

import pandas as pd
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import tensorflow as tf
import cv2
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load the Keras model
model_path = '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/saved_models/product_recommendation_model.h5'
model = tf.keras.models.load_model(model_path)

# Load the tokenizer
tokenizer_path = '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/models/tokenizer.pickle'
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Preprocessing functions
max_len = 100

def preprocess_image(image):
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    return img

def pad_text_sequence(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    return padded_sequence

def preprocess_image_from_url(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img = img.resize((224, 224))
        img = np.array(img)
        img = preprocess_input(img)
        return img
    except requests.RequestException as e:
        print(f"Error processing image from URL {image_url}: {e}")
        return None

# Assuming combined_data is loaded from your dataset
# combined_data should have columns 'name', 'image', etc.
dataset_paths = [
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Football.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Badminton.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Cycling.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Cricket.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Yoga.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Strength Training.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Running.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Fitness Accessories.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Cardio Equipment.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Sports Shoes.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Sportswear.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Sports Collectibles.csv',

    # Electronics Datasets
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Air Conditioners.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Cameras.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Headphones.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Televisions.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Car Electronics.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Security Cameras.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Home Audio and Theater.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Personal Care Appliances.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Heating and Cooling Appliances.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Refrigerators.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Washing Machines.csv',

    # Fashion Datasets
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Mens Fashion.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Womens Fashion.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Kids Fashion.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Shoes.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Casual Shoes.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Formal Shoes.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Ethnic Wear.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Innerwear.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Ballerinas.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Fashion and Silver Jewellery.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Gold and Diamond Jewellery.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Handbags and Clutches.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Jeans.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Lingerie and Nightwear.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/T-shirts and Polos.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Western Wear.csv',

    # Books Datasets
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/All Books.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Fiction Books.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Childrens Books.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Exam Central.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/School Textbooks.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Textbooks.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Kindle eBooks.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Indian Language Books.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/All English.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/All Hindi.csv',

    # Home and Kitchen Datasets
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/All Home and Kitchen.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Kitchen and Dining.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Furniture.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Home Furnishing.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Home Storage.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Home Dcor.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Bedroom Linen.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Kitchen Storage and Containers.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Heating and Cooling Appliances.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Home Entertainment Systems.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Home Improvement.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Garden and Outdoors.csv',

    # Grocery Datasets
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/All Grocery and Gourmet Foods.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Coffee Tea and Beverages.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Diet and Nutrition.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Household Supplies.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Snack Foods.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Pantry.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Value Bazaar.csv',

    # Pharmacy Datasets
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Amazon Pharmacy.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Health and Personal Care.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Diet and Nutrition.csv',

    # Baby Products Datasets
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Baby Bath Skin and Grooming.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Baby Fashion.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Baby Products.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Diapers.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Nursing and Feeding.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Strollers and Prams.csv',

    # Cars and Motorbikes Datasets
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/All Car and Motorbike Products.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Car Accessories.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Car and Bike Care.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Car Parts.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Motorbike Accessories and Parts.csv',

    # Toys and Games Datasets
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/All Video Games.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Toys and Games.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/STEM Toys Store.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Toys Gifting Store.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Gaming Consoles.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/PC Games.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Gaming Accessories.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Video Games Deals.csv',

    # Luggage Datasets
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Backpacks.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Bags and Luggage.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Handbags and Clutches.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Rucksacks.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/School Bags.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Suitcases and Trolley Bags.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Travel Accessories.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Travel Duffles.csv',

    # Watches and Jewellery Datasets
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Watches.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Jewellery.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Fashion and Silver Jewellery.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Gold and Diamond Jewellery.csv',

    # Pet Supplies Datasets
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Dog supplies.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/All Pet Supplies.csv',

    # Musical Instruments Datasets
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Musical Instruments and Professional Audio.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Indian Classical.csv',

    # Movies and TV Datasets
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/All Movies and TV Shows.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Blu-ray.csv',

    # Collectibles Datasets
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Entertainment Collectibles.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Sports Collectibles.csv',

    # Outdoor and Adventure Datasets
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Camping and Hiking.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Garden and Outdoors.csv',

    # Health and Personal Care Datasets
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Health and Personal Care.csv',
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Personal Care Appliances.csv',

    # Kitchen Storage Datasets
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Kitchen Storage and Containers.csv',

    # Bedding Datasets
    '/home/aman/Projects/AI VOICE ASSISSTANT/Recommendation system/Datasets/Bedroom Linen.csv',
]

def load_and_filter_datasets(dataset_paths):
    combined_data = pd.DataFrame()
    for path in dataset_paths:
        dataset = pd.read_csv(path)
        combined_data = pd.concat([combined_data, dataset])
    return combined_data

combined_data = load_and_filter_datasets(dataset_paths)

# Recommendation function
def recommend(product_name, image_url, top_n=5):
    # Preprocess the input
    image = preprocess_image_from_url(image_url)
    if image is None:
        raise ValueError("Failed to process the image")
    
    product_name_sequence = pad_text_sequence(product_name)
    
    # Predict scores for all items
    predictions = model.predict([np.expand_dims(image, axis=0), product_name_sequence])
    
    # Get top N recommendations
    top_indices = np.argsort(predictions[0])[-top_n:]
    
    recommendations = combined_data.iloc[top_indices]
    recommendations['score'] = predictions[0][top_indices]
    
    return recommendations[['name', 'image', 'score']]

def recommend_from_history(purchase_history=[], search_history=[], top_n=5):
    recommendations = []
    
    if purchase_history:
        for name, image_url in purchase_history:
            recommendations.append(recommend(name, image_url, top_n=top_n))
    
    if search_history:
        for name, image_url in search_history:
            recommendations.append(recommend(name, image_url, top_n=top_n))
    
    # Combine and deduplicate recommendations
    if recommendations:
        recommendations_df = pd.concat(recommendations).drop_duplicates(subset=['name', 'image'])
        
        # Get top N recommendations
        top_indices = np.argsort(recommendations_df['score'].values)[-top_n:]
        return recommendations_df.iloc[top_indices][['name', 'image', 'score']]
    else:
        return pd.DataFrame(columns=['name', 'image', 'score'])


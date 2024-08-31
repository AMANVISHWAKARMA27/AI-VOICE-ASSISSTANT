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
model_path = '../saved_models/product_recommendation_model.h5'
model = tf.keras.models.load_model(model_path)

# Load the tokenizer
tokenizer_path = '../models/tokenizer.pickle'
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
    '../Datasets/Football.csv',
    '../Datasets/Badminton.csv',
    '../Datasets/Cycling.csv',
    '../Datasets/Cricket.csv',
    '../Datasets/Yoga.csv',
    '../Datasets/Strength Training.csv',
    '../Datasets/Running.csv',
    '../Datasets/Fitness Accessories.csv',
    '../Datasets/Cardio Equipment.csv',
    '../Datasets/Sports Shoes.csv',
    '../Datasets/Sportswear.csv',
    '../Datasets/Sports Collectibles.csv',

    # Electronics Datasets
    '../Datasets/Air Conditioners.csv',
    '../Datasets/Cameras.csv',
    '../Datasets/Headphones.csv',
    '../Datasets/Televisions.csv',
    '../Datasets/Car Electronics.csv',
    '../Datasets/Security Cameras.csv',
    '../Datasets/Home Audio and Theater.csv',
    '../Datasets/Personal Care Appliances.csv',
    '../Datasets/Heating and Cooling Appliances.csv',
    '../Datasets/Refrigerators.csv',
    '../Datasets/Washing Machines.csv',

    # Fashion Datasets
    '../Datasets/Mens Fashion.csv',
    '../Datasets/Womens Fashion.csv',
    '../Datasets/Kids Fashion.csv',
    '../Datasets/Shoes.csv',
    '../Datasets/Casual Shoes.csv',
    '../Datasets/Formal Shoes.csv',
    '../Datasets/Ethnic Wear.csv',
    '../Datasets/Innerwear.csv',
    '../Datasets/Ballerinas.csv',
    '../Datasets/Fashion and Silver Jewellery.csv',
    '../Datasets/Gold and Diamond Jewellery.csv',
    '../Datasets/Handbags and Clutches.csv',
    '../Datasets/Jeans.csv',
    '../Datasets/Lingerie and Nightwear.csv',
    '../Datasets/T-shirts and Polos.csv',
    '../Datasets/Western Wear.csv',

    # Books Datasets
    '../Datasets/All Books.csv',
    '../Datasets/Fiction Books.csv',
    '../Datasets/Childrens Books.csv',
    '../Datasets/Exam Central.csv',
    '../Datasets/School Textbooks.csv',
    '../Datasets/Textbooks.csv',
    '../Datasets/Kindle eBooks.csv',
    '../Datasets/Indian Language Books.csv',
    '../Datasets/All English.csv',
    '../Datasets/All Hindi.csv',

    # Home and Kitchen Datasets
    '../Datasets/All Home and Kitchen.csv',
    '../Datasets/Kitchen and Dining.csv',
    '../Datasets/Furniture.csv',
    '../Datasets/Home Furnishing.csv',
    '../Datasets/Home Storage.csv',
    '../Datasets/Home Dcor.csv',
    '../Datasets/Bedroom Linen.csv',
    '../Datasets/Kitchen Storage and Containers.csv',
    '../Datasets/Heating and Cooling Appliances.csv',
    '../Datasets/Home Entertainment Systems.csv',
    '../Datasets/Home Improvement.csv',
    '../Datasets/Garden and Outdoors.csv',

    # Grocery Datasets
    '../Datasets/All Grocery and Gourmet Foods.csv',
    '../Datasets/Coffee Tea and Beverages.csv',
    '../Datasets/Diet and Nutrition.csv',
    '../Datasets/Household Supplies.csv',
    '../Datasets/Snack Foods.csv',
    '../Datasets/Pantry.csv',
    '../Datasets/Value Bazaar.csv',

    # Pharmacy Datasets
    '../Datasets/Amazon Pharmacy.csv',
    '../Datasets/Health and Personal Care.csv',
    '../Datasets/Diet and Nutrition.csv',

    # Baby Products Datasets
    '../Datasets/Baby Bath Skin and Grooming.csv',
    '../Datasets/Baby Fashion.csv',
    '../Datasets/Baby Products.csv',
    '../Datasets/Diapers.csv',
    '../Datasets/Nursing and Feeding.csv',
    '../Datasets/Strollers and Prams.csv',

    # Cars and Motorbikes Datasets
    '../Datasets/All Car and Motorbike Products.csv',
    '../Datasets/Car Accessories.csv',
    '../Datasets/Car and Bike Care.csv',
    '../Datasets/Car Parts.csv',
    '../Datasets/Motorbike Accessories and Parts.csv',

    # Toys and Games Datasets
    '../Datasets/All Video Games.csv',
    '../Datasets/Toys and Games.csv',
    '../Datasets/STEM Toys Store.csv',
    '../Datasets/Toys Gifting Store.csv',
    '../Datasets/Gaming Consoles.csv',
    '../Datasets/PC Games.csv',
    '../Datasets/Gaming Accessories.csv',
    '../Datasets/Video Games Deals.csv',

    # Luggage Datasets
    '../Datasets/Backpacks.csv',
    '../Datasets/Bags and Luggage.csv',
    '../Datasets/Handbags and Clutches.csv',
    '../Datasets/Rucksacks.csv',
    '../Datasets/School Bags.csv',
    '../Datasets/Suitcases and Trolley Bags.csv',
    '../Datasets/Travel Accessories.csv',
    '../Datasets/Travel Duffles.csv',

    # Watches and Jewellery Datasets
    '../Datasets/Watches.csv',
    '../Datasets/Jewellery.csv',
    '../Datasets/Fashion and Silver Jewellery.csv',
    '../Datasets/Gold and Diamond Jewellery.csv',

    # Pet Supplies Datasets
    '../Datasets/Dog supplies.csv',
    '../Datasets/All Pet Supplies.csv',

    # Musical Instruments Datasets
    '../Datasets/Musical Instruments and Professional Audio.csv',
    '../Datasets/Indian Classical.csv',

    # Movies and TV Datasets
    '../Datasets/All Movies and TV Shows.csv',
    '../Datasets/Blu-ray.csv',

    # Collectibles Datasets
    '../Datasets/Entertainment Collectibles.csv',
    '../Datasets/Sports Collectibles.csv',

    # Outdoor and Adventure Datasets
    '../Datasets/Camping and Hiking.csv',
    '../Datasets/Garden and Outdoors.csv',

    # Health and Personal Care Datasets
    '../Datasets/Health and Personal Care.csv',
    '../Datasets/Personal Care Appliances.csv',

    # Kitchen Storage Datasets
    '../Datasets/Kitchen Storage and Containers.csv',

    # Bedding Datasets
    '../Datasets/Bedroom Linen.csv',
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


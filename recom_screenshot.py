import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import os
from PIL import Image
# Load the USE model from TensorFlow Hub
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# embed = hub.load("model")

class RecOutfit:
    def __init__(self,input_image_dict,wardrobe_path) -> None:
        #wardrobe_path='/content/drive/MyDrive/FashionAI/Wardrobe'
        self.input_image_dict=input_image_dict
        self.wardrobe_path=wardrobe_path
        self.image_pool_paths = [os.path.join(self.wardrobe_path,f) for f in os.listdir(self.wardrobe_path)]
        self.wardrobe_tagfile_path='FashionAI Data.xlsx'
        self.wardrobe_data=[]
        self.load_model()
        
        
        
        
    def load_model(self):
        # Load a pre-trained CNN model (VGG16 in this case)
        self.model = VGG16(weights='imagenet', include_top=False)
        
        
    def load_input_image_feaures(self):
        # Load and preprocess your input image
        input_image = Image.open(self.input_image_dict['image_path'])
        input_image = input_image.resize((224, 224))  # Resize to match VGG16 input size
        input_image = np.array(input_image)
        input_image = preprocess_input(input_image)
        input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
        
        # Extract features from the input image and the image pool
        input_features = self.model.predict(input_image) 
        # Reshape the feature arrays to be 2D
        input_features = input_features.reshape(1, -1)
        return input_features
    
    
    def load_wardrobe_tags(self):
        # Load the data from the Excel file into a DataFrame
        excel_data = pd.read_excel(self.wardrobe_tagfile_path)  # Replace with the actual file path
        for columns in excel_data.columns:
            excel_data[columns]=excel_data[columns].str.lower()
            excel_data[columns]=excel_data[columns].str.strip()
        
        return excel_data
        
    def load_wardrobe_images(self):
        self.wardrobe_data = []
        tags_df=self.load_wardrobe_tags()
        for image_path in self.image_pool_paths:
            image = Image.open(image_path)
            image = image.resize((224, 224))  # Resize to match VGG16 input size
            image = np.array(image)
            image = preprocess_input(image)
            tags_dict=tags_df[tags_df['image_id']==os.path.basename(image_path)].to_dict(orient='records')[0]
    
            self.wardrobe_data.append({'Image Tags':tags_dict,'image':image})
            # image_pool.append({'Image Tags':{'Gender':'Female','Season':'Winter','Occasion':'home'},'image':image})
        return self.wardrobe_data
        
        
        
    def load_wardrobe_images_features(self):
        
        image_pool = [image['image'] for image in self.load_wardrobe_images()]
        
            
        image_pool = np.array(image_pool)

        wardrobe_images_features = self.model.predict(image_pool)
        # Reshape the feature arrays to be 2D
        wardrobe_images_features = wardrobe_images_features.reshape(len(image_pool), -1)
        
        return wardrobe_images_features

    
    def get_similar_outfits(self):
        input_features=self.load_input_image_feaures()
        wardrobe_images_features=self.load_wardrobe_images_features()
        # Calculate cosine similarity between the input image and the pool
        similarities = cosine_similarity(input_features, wardrobe_images_features)
        
        
        
        
        # Sort images by similarity and select the top 5 with similarity > 0.7
        sorted_indices = np.argsort(similarities[0])[::-1]
        top_indices = [i for i in sorted_indices if similarities[0][i] > 0.7][:5]
        if not top_indices:
            top_indices = [sorted_indices[0]]

        top_images=[self.wardrobe_data[index] for index in top_indices]
        return top_images
    
    def get_similar_based_on_tags(self,data,input_image_features):
        # Create a function to calculate the similarity score
        def calculate_similarity(row, sample_row):
            row_values = [row[key] for key in sample_row.keys()]
            sample_values = list(sample_row.values())
            row_embedding = embed(row_values).numpy().flatten()
            sample_embedding = embed(sample_values).numpy().flatten()
            return np.dot(row_embedding, sample_embedding) / (np.linalg.norm(row_embedding) * np.linalg.norm(sample_embedding))

        # Calculate similarity scores for all rows
        data['similarity_score'] = data.apply(lambda row: calculate_similarity(row, input_image_features), axis=1)

        # Sort the DataFrame by similarity score in descending order
        data = data.sort_values(by='similarity_score', ascending=False)

        # Get the top N most similar rows
        top_N = 1  # You can change this value as per your requirement
        most_similar_rows = data.head(top_N)
        most_similar_row=most_similar_rows.to_dict(orient='records')[0]

        return most_similar_row
    
    def controller(self):
        similar_outfits=self.get_similar_outfits()
        print('-----------------------------im here')
        print(similar_outfits[0]['Image Tags'])
        similar_outfit_tags=pd.DataFrame([data['Image Tags'] for data in similar_outfits])
        print(self.input_image_dict)
        similar_outfit_tags=self.get_similar_based_on_tags(similar_outfit_tags,self.input_image_dict['Image Tags'])
        
        
        # image=[image['image'] for image in similar_outfits if image['Image Tags']['image_id']==similar_outfit_tags['image_id']][0]
        image=Image.open(self.wardrobe_path + "\\" + similar_outfit_tags['image_id'] )
        print(similar_outfit_tags)
        return similar_outfit_tags,image
        

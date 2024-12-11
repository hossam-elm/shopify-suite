#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 10:54:24 2024

@author: acheteur
"""

import pandas as pd
import numpy as np  # Importing numpy for ceil function
from openai import OpenAI
import os
import requests
from PIL import Image
from io import BytesIO
import base64
import json
import streamlit as st



os.environ["OPENAI_API_KEY"] = st.secrets["open_api_key"]
SHOPIFY_API_KEY = st.secrets["shopify_api_key"]
SHOPIFY_API_PASSWORD = st.secrets["shopify_api_password"]
SHOPIFY_DOMAIN = st.secrets["shopify_domain"]


client = OpenAI()



# Function to resize images and adjust file size with default parameters and URL support
def resize_image_from_url(image_url, target_width=800, target_height=800, output_format="jpeg", padding_color=(255, 255, 255), coeff_reduction=1.0):
    # Check if the image URL is NaN or invalid
    if image_url is None or image_url == '' or pd.isna(image_url):
        return None  # Skip invalid URLs (NaN or empty strings)
    
    # Fetch the image from the URL
    response = requests.get(image_url)
    uploaded_file = BytesIO(response.content)

    with Image.open(uploaded_file) as img:
        original_width, original_height = img.size
        
        # Calculate the scaling factor based on target width and height while maintaining aspect ratio
        ratio = min(target_width / original_width, target_height / original_height)
        
        # Calculate the new dimensions based on the ratio, keeping aspect ratio intact
        new_width = int(original_width * ratio * coeff_reduction)
        new_height = int(original_height * ratio * coeff_reduction)

        # Resize the image while maintaining aspect ratio
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)

        # Convert to RGB if output is JPEG or any format that does not support alpha
        if output_format.lower() in ["jpeg", "jpg"]:
            resized_img = resized_img.convert("RGB")

        # Save with quality adjustments to fit within 70KB to 200KB range
        img_byte_arr = BytesIO()
        quality = 100  # Start with high quality
        step = -5  # Step to decrease quality if file size is too large
        while True:
            img_byte_arr = BytesIO()
            # Save the image with the current quality level
            if output_format.lower() == "webp":
                resized_img.save(img_byte_arr, format="WEBP", quality=quality)
            elif output_format.lower() in ["jpeg", "jpg"]:
                resized_img.save(img_byte_arr, format="JPEG", quality=quality, optimize=True)
            else:
                # For PNG, try reducing to 256 colors (PNG8)
                temp_img = resized_img.convert("P", palette=Image.ADAPTIVE, colors=256)
                temp_img.save(img_byte_arr, format="PNG", optimize=True)

            # Use len() to get the actual file size with metadata
            file_size = len(img_byte_arr.getvalue()) / 1024  # Size in KB, including metadata

            if file_size <= 200 or quality <= 5:
                break
            quality += step
        
        img_byte_arr.seek(0)
        
        # Encode the image to Base64
        base64_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                
        # Return the Base64 string
        return base64_image

#############################################-Transformation-#####################################################################

# Step 4: If file is successfully loaded, perform the transformation
# Function to transform Title
def clean_text(row):
    style_name=row['StyleName']
    text = row['ShortDescription']
    if text.lower().startswith(('le ', 'la ')):
     text = text[3:]  # Remove "Le " or "La "
     text = text.split()[0]  # Keep only the first word
    else:
     text = text.split()[0]  # Keep the first word in other cases as well
 
 # Capitalize the first letter
    text = text.capitalize()
 
    return f"{text} {style_name} à personnaliser"
#End
def resize_main_image(df):
    resized_images = []  # Create a list to store resized images
    for index, row in df.iterrows():
        try:
            main_image = row['MainPicture']
            resized_image = resize_image_from_url(main_image)  # Get resized image
            resized_images.append(resized_image)  # Add resized image to list
        except:
            st.write(f"invalid link: {main_image}")
            resized_images.append(None)  # Add None for invalid links
    
    # Assign the resized images list to the new column
    df['resized_image'] = resized_images
    return df

def reformulate_description(df):
    """Reformulate product description and generate SEO-friendly meta title and meta description in French."""
    # Lists to hold the new columns
    reformulated_descriptions = []
    meta_titles = []
    meta_descriptions = []
    features = []
    
    # Loop through each row in the DataFrame
    for index, row in df.iterrows():
        description = row['ShortDescription']
        style_name = row['StyleName']
        weight= row['Weight']
        
        # Ensure that description and style_name are not empty
        if not description or not style_name:
            print(f"Warning: Empty description or style_name at index {index}. Skipping this row.")
            reformulated_descriptions.append("")
            meta_titles.append("")
            meta_descriptions.append("")
            continue
        st.write(f"Generating description, meta title, and meta description for product at index {index}...")

        # Define the prompts inside the loop to customize for each row
        prompt_description = f"Create a concise, catchy and professional description (under 99 words) for my Shopify store based on the following paragraph. Highlight the key eco-responsible features and practical design of the garment and say that it's customizable, and it is for your clients and collaborators, while keeping the tone modern and focused on sustainability, in French and make it not so technical:\n\n{description}, don't forget to mention the product name fluidly in the text:\n\n{style_name}"

        prompt_meta_title = f"Create a short and catchy and professional meta title for my Shopify store (max 60 characters) for a sustainable and customizable garment named {style_name}. Focus on its eco-friendly features and its appeal to clients and collaborators. Keep it modern, fluid, and in French, use this description:\n\n{description} and respect SEO"

        prompt_meta_description = f"Generate a compelling and professional meta description for my Shopify store (max 160 characters) for the product {style_name}, highlighting its eco-responsible features, modern design, and customization options. Emphasize its appeal to both clients and collaborators, written in French, use this description:\n\n{description} and respect SEO"
        
        prompt_metafields = f"Generate 5 product characteristics for a Shopify listing in French. The product has the following features: \n\n{description}, and weight: {weight} GSM. The description should be concise, without numbering, and without titles before the sentences, the second one should be just the weight in GSM rounded to integer"

        try:
            # Make the API call for the product description
            st.write(f"Making API call for product description for '{style_name}'...")
            response_description = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt_description}],
                model="gpt-3.5-turbo",
            )

            # Make the API call for the meta title
            response_meta_title = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt_meta_title}],
                model="gpt-3.5-turbo",
            )

            # Make the API call for the meta description
            response_meta_description = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt_meta_description}],
                model="gpt-3.5-turbo",
            )
            
            response_features = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt_metafields}],
                model="gpt-3.5-turbo",
            )


            # Extract the content from the responses
            reformulated_description = response_description.choices[0].message.content.strip()
            meta_title = response_meta_title.choices[0].message.content.strip()
            meta_description = response_meta_description.choices[0].message.content.strip()
            feature = response_features.choices[0].message.content.strip()

            # Append the results to the lists
            reformulated_descriptions.append(reformulated_description)
            meta_titles.append(meta_title)
            meta_descriptions.append(meta_description)
            features.append(feature)

        except Exception as e:
            st.write(f"Error with OpenAI API for index {index}: {e}")
            print(f"Error with OpenAI API for index {index}: {e}")
            reformulated_descriptions.append("")
            meta_titles.append("")
            meta_descriptions.append("")

    # Add the new columns to the DataFrame
    st.write("Updating the DataFrame with reformulated descriptions and meta data...")
    df['reformulated_description'] = reformulated_descriptions
    df['meta_title'] = meta_titles
    df['meta_description'] = meta_descriptions
    df['features'] = features
    return df
##################################################################################################################
def process_df(df):
    if df is not None:
        st.write("Processing data...")
        df = df.copy()
        df = df[['StyleCode', 'StyleName', 'Type', 'Category', 'SizeCode', 'ShortDescription', 'LongDescription', 'Price>1000 EUR', 'Color', 'ColorCode', 'Weight', 'MainPicture']]
        
        # Track Title transformation
        st.write("Transforming 'Title' based on 'StyleName' and 'ShortDescription'...")
        df['Title'] = df.apply(clean_text, axis=1)
        
        # Check for 'Price' and round it up
        st.write("Transforming 'Price' and rounding up to the nearest integer...")
        if 'Price>1000 EUR' in df.columns:
            df.rename(columns={'Price>1000 EUR': 'Price'}, inplace=True)
            df['Price'] = df['Price'].apply(np.ceil)  # Round price up to the nearest integer

        # Track Size, Color, ColorCode aggregation
        st.write("Aggregating 'SizeCode', 'Color', and 'ColorCode'...")
        aggregated_df = df.groupby('StyleCode', as_index=False).agg({
            'SizeCode': lambda x: ':'.join(sorted(x.unique(), reverse=True)),
            'Color': lambda x: ':'.join(sorted(x.unique())),
            'ColorCode': lambda x: ':'.join(sorted(x.unique()))
        })
        
        # Dropping duplicates based on StyleCode
        st.write("Removing duplicates and filtering out 'pantalon' entries...")
        df = df.drop_duplicates(subset='StyleCode', keep='first')
        df = df[~df['ShortDescription'].str.contains('pantalon', case=False, na=False)]
        
        # Merging aggregated data back
        st.write("Merging aggregated data back with the original DataFrame...")
        df_merged = pd.merge(df.drop(['SizeCode', 'Color', 'ColorCode'], axis=1).drop_duplicates(),
                             aggregated_df,
                             on='StyleCode',
                             how='left')

        # Reformulating descriptions
        if use_openai:
            st.write("Reformulating product descriptions and meta data...")
            df_merged = reformulate_description(df_merged)
        st.write("Resizing main image...")
        df_merged = resize_main_image(df_merged)
        
        st.write("Data transformation complete.")
        return df_merged
    else:
        st.write("Failed to transform data.")
        return None

        
#############################################-Transformation-#####################################################################

        
def product_exists_in_shopify(sku):
    """Check if a product with the given SKU already exists in Shopify by checking variants."""
    url = f"https://{SHOPIFY_API_KEY}:{SHOPIFY_API_PASSWORD}@{SHOPIFY_DOMAIN}/admin/products.json"
    response = requests.get(url, headers={'Content-Type': 'application/json'})

    if response.status_code == 200:
        products = response.json().get('products', [])
        for product in products:
            # Check each variant for matching SKU
            for variant in product.get('variants', []):
                if variant.get('sku') == sku:
                    # Return the existing product if a matching variant is found
                    return product
        return None
    else:
        print(f"Error checking products: {response.status_code}")
        print(response.text)
        return None

# Function to create a new product in Shopify
def create_product_in_shopify(product_data):
    """Creates a new product in Shopify."""
    url = f"https://{SHOPIFY_API_KEY}:{SHOPIFY_API_PASSWORD}@{SHOPIFY_DOMAIN}/admin/products.json"
    response = requests.post(
        url,
        json=product_data,
        headers={'Content-Type': 'application/json'},
        verify=False  # Disable SSL verification (not recommended for production)
    )

    if response.status_code == 201:
        return response.json()  # Successfully created product
    else:
        print(f"Error creating product in Shopify: {response.status_code}")
        print(f"Response: {response.text}")  # Print the full response for debugging
        return None

# Function to update an existing product in Shopify
def update_product_in_shopify(product_id, updated_data):
    """Updates an existing product in Shopify."""
    url = f"https://{SHOPIFY_API_KEY}:{SHOPIFY_API_PASSWORD}@{SHOPIFY_DOMAIN}/admin/products/{product_id}.json"
    response = requests.put(
        url,
        json=updated_data,
        headers={'Content-Type': 'application/json'}
    )

    if response.status_code == 200:
        return response.json()  # Successfully updated product
    else:
        print(f"Error updating product in Shopify: {response.status_code}")
        print(f"Response: {response.text}")
        return None

# Function to convert a DataFrame row to Shopify product format
def convert_row_to_shopify_product(row):
    """Converts a DataFrame row into the Shopify product format."""
    # Generate tags
    tags = [row.get("StyleCode", ""), row.get("StyleName", ""), row.get("Type", ""), 
            row.get("Category", ""), row.get("SizeCode", "")]
    tags = [tag for tag in tags if tag]  # Ensure tags are unique and not empty
    
    style_code = row.get("StyleCode","")
    body_html = row.get("reformulated_description", "")
    product_title = row.get("Title", "")
    meta_title = row.get("meta_title", "")
    meta_desc = row.get("meta_description", "")
    features_list = row.get('features', '').split('\n')
    json_string = json.dumps(features_list)
    main_image = row.get("resized_image", "")
    main_image_payload = [{"attachment": main_image, "position": "1", "filename": f"Main_{style_code}"}]
    
    colors = row.get("Color", "").split(":")
    color_codes = row.get("ColorCode", "").split(":")
    #variant_img=row.get("HTMLPath", "").split(';')
    #variant_img_payload = [{"attachment": variant_img, "position": "1", "filename": f"Main_{style_code}"}]
    # Create an empty list to hold variants
    variants = []
    for i in range(len(colors)):
        if i < len(color_codes):
            color_code = color_codes[i].strip()
            color = colors[i].strip()
            variant_color = f"{color}__color__{color_code}"        
            # Build the variant
            variant = {
                "sku": row["StyleCode"],  # Using StyleCode as SKU
                "option1": color,  # Color as option1
                "price": row.get("Price", "0.00"),  # Price
                "color": variant_color,  # Add the formatted color code
                "inventory_management": None,
                "requires_shipping": False,
                "taxable": False,
            }
            variants.append(variant)
    metafields = [
        {
            "key": "characteristics",
            "value": json_string,
            "type": "json_string",
            "namespace": "char"
        }
    ]
    shopify_product = {
        "product": {
            "title": product_title,
            "body_html": body_html,
            "vendor": "Stanley/Stella",
            "product_type": "Textile",
            "tags": ", ".join(tags),
            "variants": variants,
            "metafields_global_title_tag": meta_title,
            "metafields_global_description_tag": meta_desc,
            "metafields": metafields,
            "status": "draft",
            "images": main_image_payload
        }
    }
    return shopify_product

# Function to add or update product from DataFrame row to Shopify
def add_or_update_product_from_df_to_shopify(row):
    """Adds a new product or updates an existing one in Shopify."""
    sku = row["StyleCode"]
    existing_product = product_exists_in_shopify(sku)

    if existing_product:
        # If the product already exists, update it
        st.write(f"Product with SKU {sku} already exists in Shopify. Updating...")
        product_id = existing_product["id"]
        shopify_product_data = convert_row_to_shopify_product(row)
        response = update_product_in_shopify(product_id, shopify_product_data)
        if response:
            st.write(f"Successfully updated product with SKU {sku}")
            return f"Successfully updated product with SKU {sku}"
        else:
            st.write(f"Failed to update product with SKU {sku}")
            return f"Failed to update product with SKU {sku}"

    else:
        # If the product does not exist, create a new product
        st.write(f"Product with SKU {sku} does not exist in Shopify. Creating new product...")
        shopify_product_data = convert_row_to_shopify_product(row)
        response = create_product_in_shopify(shopify_product_data)
        if response:
            st.write(f"Successfully created product with SKU {sku}")
            return f"Successfully created product with SKU {sku}"

        else:
            st.write(f"Failed to create product with SKU {sku}")
            return f"Failed to update product with SKU {sku}"

# Function to process products from a DataFrame
def add_or_update_products_from_dataframe_to_shopify(df):
    """Processes a DataFrame and adds/updates products in Shopify with error handling."""
    results = []
    
    for _, row in df.iterrows():
        try:
            st.write(f"Processing product with SKU {row['StyleCode']}...")
            result = add_or_update_product_from_df_to_shopify(row)
            results.append(result)
            st.write(f"Successfully processed product with SKU {row['StyleCode']}")
        
        except Exception as e:
            # Handle any error for the current row
            st.write(f"Error processing product with SKU {row['StyleCode']}: {e}")
            results.append({'SKU': row['StyleCode'], 'status': 'Failed', 'error': str(e)})
    
    st.write("Finished processing all products.")
    return results


# Path to your logo file
logo_path = "logos/logo_atelierbox.avif"  # Make sure this path is correct

# Function to encode image in base64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Get base64 string of the logo
logo_base64 = get_base64_image(logo_path)

# Set page configuration with a custom favicon and centered layout
st.set_page_config(page_title="Database Treatment & Shopify Uploader", layout="centered", page_icon="logos/favicon_atelierbox.png")

# Custom CSS for a modern, sleek design with your brand colors
st.markdown(
    f"""
    <style>
    /* Global App Styling */
    [data-testid="stAppViewContainer"] {{
        background: #FEFBF1;  /* Light cream beige background */
        font-family: 'Roboto', sans-serif;  /* Modern font */
        color: #373A23;  /* Dark greenish-brown text for readability */
    }}
    
    /* Header and Title */
    .title {{
        color: #373A23;  /* Dark Greenish Brown for headers */
        font-size: 60px;
        font-weight: 800;
        text-align: center;
        margin-top: 50px;
        letter-spacing: -2px;
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);  /* Soft shadow for elegance */
    }}

    /* Container Styling (Forms, Data, Uploads) */
    [data-testid="stForm"], .dataframe, .file-uploader {{
        background: #FFFFFF;  /* White background for clean separation */
        border-radius: 12px;
        padding: 30px;
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.05);  /* Soft shadow for a floating effect */
        margin: 20px auto;
        max-width: 900px;
    }}

    /* Button Styling */
    .btn {{
        background: linear-gradient(145deg, #C18572, #838667);  /* Smooth gradient using brand colors */
        color: white;
        font-size: 18px;
        padding: 15px 40px;
        border-radius: 50px;  /* Full rounded button */
        border: none;
        cursor: pointer;
        transition: all 0.3s ease-in-out;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }}
    
    .btn:hover {{
        background: linear-gradient(145deg, #838667, #C18572);  /* Reverse gradient on hover */
        transform: translateY(-4px);  /* Subtle lift effect */
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
    }}
    
    /* Input fields styling */
    .stTextInput, .stNumberInput, .stSelectbox, .stTextArea {{
        border-radius: 10px;
        border: 1px solid #838667;  /* Olive greenish-grey border */
        padding: 14px;
        font-size: 16px;
        width: 100%;
        box-sizing: border-box;
        transition: border-color 0.3s ease;
    }}

    .stTextInput:focus, .stNumberInput:focus, .stSelectbox:focus, .stTextArea:focus {{
        border-color: #C18572;  /* Warm peachy border on focus */
        outline: none;
    }}

    /* Logo Styling */
    .top-left-logo {{
        position: absolute;
        top: 20px;
        left: 20px;
        width: 150px;  /* Sleek and balanced size */
        opacity: 0.85;
        transition: transform 0.3s ease;
    }}
    
    .top-left-logo:hover {{
        transform: scale(1.1);  /* Subtle zoom effect */
    }}

    /* Header Styling for Upload Section */
    .header {{
        color: #373A23;  /* Dark Greenish Brown */
        font-size: 22px;
        font-weight: 600;
        text-align: center;
        margin-bottom: 20px;
    }}

    </style>
    <img src="data:image/png;base64,{logo_base64}" class="top-left-logo">  <!-- Logo at the top-left corner -->
    """,
    unsafe_allow_html=True
)

# Display the logo and title
st.markdown('<div class="title">Shopify Product Uploader</div>', unsafe_allow_html=True)

# File upload section
st.markdown('<div class="header">Upload your CSV file to process and upload to Shopify</div>', unsafe_allow_html=True)
if 'data_processed' not in st.session_state:
    st.session_state['data_processed'] = False  # Tracks if the data has been processed

uploaded_file = st.file_uploader("Choose your CSV file", type=["csv"])

# OpenAI
use_openai = st.checkbox('Reformulate description',help="Click here to use chatgpt to reformulate your text fields")


# Transform Database button
if st.button("Transform Database", key="Transform", help="Click to transform your database to the correct format"):
    if uploaded_file is not None:
        st.write("Processing the uploaded file...")
        try:
            df = pd.read_csv(uploaded_file, sep=";")
            #df = df.head(15)  # Display a preview of the first 15 rows for processing
            df = process_df(df)  # Assuming process_df is your custom data processing function
            if not df.empty:
                st.session_state['data_processed'] = True  # Mark data as processed
                st.session_state['processed_data'] = df  # Save processed data to session state
                st.markdown('<div class="header">Preview of the processed data:</div>', unsafe_allow_html=True)
                st.dataframe(df, use_container_width=True)  # Adjust dataframe display
            else:
                st.write("No data available after processing.")
        except Exception as e:
            st.write(f"An error occurred while processing the file: {e}")
    else:
        st.write("Error: Please upload a file before processing.")

# Upload to Shopify button
if st.button("Upload to Shopify", key="upload", help="Click to upload products to Shopify"):
    if not st.session_state['data_processed']:
        st.write("Error: Please transform the database first by clicking 'Transform Database'.")
    else:
        st.write("Uploading products to Shopify. Please wait...")
        df = st.session_state['processed_data']  # Retrieve processed data
        results = add_or_update_products_from_dataframe_to_shopify(df)  # Upload function
        st.write("Results:")
        for result in results:
            st.write(result)
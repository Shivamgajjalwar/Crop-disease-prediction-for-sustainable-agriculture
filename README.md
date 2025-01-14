# Crop-disease-prediction-for-sustainable-agriculture
🌾 Crop Disease Detection System for Sustainable Agriculture
🌟 Project Overview
The Crop Disease Detection System for Sustainable Agriculture is an innovative AI-powered solution designed to assist farmers in identifying and managing crop diseases. By uploading an image of a crop leaf, the system uses a trained machine learning model to detect the disease and then provides detailed solutions. Additionally, the system supports multilingual translations, making it accessible to farmers across different regions.

🛠️ Features
Accurate Disease Detection: Utilizes a deep learning model to predict crop diseases from images.
Detailed Solutions: Offers a comprehensive explanation of the disease and management strategies.
Multilingual Support: Translate disease solutions into multiple languages to enhance accessibility.
User-Friendly Interface: Simple, web-based application powered by Streamlit for ease of use.
🖥️ Technologies Used
Programming Language: Python
Machine Learning Framework: TensorFlow
Frontend: Streamlit
📂 File Structure
plaintext
Copy code
Crop Disease Detection System for Sustainable Agriculture/
├── plant_village_dataset/          # Dataset folder
├── class_indices.json              # Class index mappings for diseases
├── download_dataset.py             # Script to download the dataset
├── kaggle.json                     # Kaggle API credentials
├── main.py                         # Core training and prediction script
├── plant_disease_prediction_model.h5 # Trained ML model
├── streamlit_app.py                # Streamlit application for the frontend
🚀 Installation Instructions
Follow these steps to set up and run the project locally:

Clone the Repository:

bash
Copy code
git clone https://github.com/Shivamgajjalwar/crop-disease-detection-system.git
cd crop-disease-detection-system
Install Dependencies:
Ensure you have Python installed. Install required libraries:

bash
Copy code
pip install -r requirements.txt
Run the Streamlit App:
Launch the application with the following command:

bash
Copy code
streamlit run streamlit_app.py
Access the App:
Open your browser and navigate to http://localhost:8501 to use the app.

📷 Screenshots
Add screenshots of your application to demonstrate its interface and functionality. Example:

Upload image screen
Disease prediction results
Multilingual solution display
👤 About the Author
Name: Shivam Gajjalwar
GitHub: Shivamgajjalwar
LinkedIn: Shivam Gajjalwar
Email: shivamgajjalwar2@gmail.com
📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

🙏 Acknowledgments
This project was developed to aid sustainable agriculture by leveraging advanced technologies. Special thanks to all resources and frameworks that made this project possible.

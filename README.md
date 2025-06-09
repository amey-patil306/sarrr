# ForestGuard AI - Deforestation Detection System

A Flask-based web application that uses deep learning to detect deforestation in satellite imagery and provides AI-powered insights using Llama API.

## Features

- **Real-time Deforestation Detection**: Uses ResNet50-based CNN model
- **AI-Powered Insights**: Integrates with Hugging Face Llama API for solutions and analytics
- **Interactive Web Interface**: Modern, responsive design with dark/light theme
- **Visualization**: Shows original image, deforestation mask, and overlay visualization
- **Confidence Scoring**: Displays model confidence levels

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Hugging Face API token (for AI insights)

## Local Setup Instructions

### 1. Clone or Download the Project
```bash
git clone <your-repo-url>
cd forestguard-ai
```

### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root:
```bash
# Create .env file
touch .env  # On Windows: type nul > .env
```

Add your Hugging Face API token to `.env`:
```
HUGGINGFACE_API_TOKEN=your_huggingface_token_here
```

**To get a Hugging Face API token:**
1. Go to [Hugging Face](https://huggingface.co/)
2. Sign up/Login
3. Go to Settings → Access Tokens
4. Create a new token with "Read" permissions
5. Copy the token to your `.env` file

### 5. Create Upload Directory
```bash
mkdir -p static/uploads
```

### 6. Run the Application
```bash
python app.py
```

The application will start on `http://localhost:5000`

## Usage

1. Open your web browser and go to `http://localhost:5000`
2. Click on "Analyze Now" or scroll to the upload section
3. Upload a satellite image (JPG, PNG formats supported)
4. Click "Analyze Image" to get results
5. View the analysis results including:
   - Original image
   - Deforestation detection results
   - Model confidence score
   - AI-powered environmental analysis
   - Recommended solutions (if deforestation detected)
   - Prevention strategies

## Project Structure

```
forestguard-ai/
├── app.py                 # Main Flask application
├── llama_service.py       # Hugging Face Llama API integration
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (create this)
├── static/
│   ├── styles.css        # CSS styles
│   └── uploads/          # Uploaded images storage
├── templates/
│   └── index.html        # Main HTML template
└── README.md             # This file
```

## Troubleshooting

### Common Issues:

1. **ModuleNotFoundError**: Make sure you've activated your virtual environment and installed all dependencies
2. **API Token Issues**: Ensure your Hugging Face token is correctly set in the `.env` file
3. **Upload Issues**: Check that the `static/uploads/` directory exists and has write permissions
4. **Model Loading**: The ResNet50 model will download automatically on first run (may take a few minutes)

### Dependencies Issues:
If you encounter issues with TensorFlow or OpenCV, try:
```bash
pip install --upgrade pip
pip install tensorflow==2.13.0
pip install opencv-python==4.8.1.78
```

## Optional: Running Without AI Insights

If you don't want to use the Llama API integration, you can:
1. Skip setting up the Hugging Face token
2. The app will still work for deforestation detection
3. Fallback content will be shown instead of AI-generated insights

## Development

To modify the model or add features:
- Model configuration is in `app.py`
- Frontend styling is in `static/styles.css`
- HTML template is in `templates/index.html`
- AI service integration is in `llama_service.py`

## License

This project is for educational and research purposes.
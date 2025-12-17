# Resume Tailoring System - Web Version

A modern, AI-powered resume optimization platform with a sleek web interface and powerful backend processing.

## 🎯 Features

- **AI-Powered Resume Tailoring**: Optimize resumes for specific job descriptions using advanced LLMs
- **Skill Gap Analysis**: Identify missing skills and get actionable learning recommendations
- **Cover Letter Generation**: Create personalized cover letters in multiple tones
- **Modern UI**: Dark, professional interface inspired by tactical systems
- **File Processing**: Handle PDF uploads and DOCX downloads
- **API-Driven**: Clean separation between frontend and backend

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements_api.txt
```

### 2. Download SpaCy Model
```bash
python -m spacy download en_core_web_sm
```

### 3. Start the Server
```bash
# Make startup script executable (Linux/Mac)
chmod +x start_server.sh

# Run the server
./start_server.sh
```

The server will start at `http://localhost:8000`

### 4. Open in Browser
Navigate to `http://localhost:8000` in your web browser.

## 📁 Project Structure

```
/home/vikon/Desktop/Python Scripts/
├── api.py              # FastAPI backend server
├── app.py              # Original Streamlit app (for reference)
├── index.html          # Web frontend
├── requirements_api.txt # Python dependencies
├── start_server.sh    # Startup script
├── outputs/            # Generated files directory
└── README.md          # This file
```

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/connect` | POST | Connect to LLM service |
| `/fetch-job-description` | POST | Fetch job description from URL |
| `/generate-resume` | POST | Generate tailored resume |
| `/generate-cover-letter` | POST | Generate cover letter |
| `/download-resume` | POST | Download resume DOCX |
| `/download-cover-letter` | POST | Download cover letter DOCX |

## 🎨 Customization

### UI Styling
The interface uses CSS custom properties for easy theming:

```css
:root {
    --bg-primary: #0a0a0a;      /* Main background */
    --accent-green: #00ff88;    /* Primary actions */
    --accent-cyan: #00aaff;     /* Secondary actions */
    --text-primary: #ffffff;    /* Main text */
}
```

### Adding New Features
1. **Frontend**: Edit `index.html` - add new UI elements and JavaScript functions
2. **Backend**: Edit `api.py` - add new FastAPI endpoints
3. **Styling**: Modify CSS in `<style>` tag for consistent theming

## 🌐 Deployment

### For Plasmic Import
1. The HTML is structured with semantic classes for easy editing
2. Components are modular and self-contained
3. CSS uses modern features but degrades gracefully

### For GitHub Pages
1. Upload all files to your GitHub repository
2. The HTML will work as a static page
3. Backend will need separate hosting (Railway, Render, etc.)

## 🔒 Security Notes

- API keys are handled server-side only
- File uploads are processed securely
- CORS is configured for local development
- In production, restrict CORS origins to your domain

## 🐛 Troubleshooting

### Common Issues

**"Backend: Offline"**
- Make sure the Python server is running on port 8000
- Check that all dependencies are installed
- Verify the virtual environment is activated

**CUDA Errors**
- The system automatically falls back to CPU
- Use the device selector in the sidebar to force CPU mode

**File Upload Issues**
- Ensure PDF files are under reasonable size limits
- Check browser console for JavaScript errors

**LLM Connection Failed**
- Verify your API keys or local server endpoints
- Check network connectivity for OpenAI API

## 🤝 Contributing

1. Test changes locally first
2. Maintain the dark theme aesthetic
3. Keep API responses consistent
4. Add error handling for new features

## 📄 License

This project is open source. Feel free to modify and distribute.

---

**Built with:** FastAPI, HTML/CSS/JavaScript, spaCy, Sentence Transformers, OpenAI API

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import shutil
from typing import Optional, Dict, Any
import json
from datetime import datetime
import torch
from openai import OpenAI
from jinja2 import Template

# Import functions from utils.py (no Streamlit dependencies)
try:
    from utils import (
        extract_resume_text, fetch_job_description, extract_keywords,
        generate_tailored_resume, create_professional_docx,
        analyze_skill_gaps, generate_cover_letter, create_cover_letter_docx,
        load_models, calculate_match_score, generate_strategic_recommendations,
        process_chat_message
    )
    FUNCTIONS_LOADED = True
except ImportError as e:
    print(f"Failed to import functions from utils.py: {e}")
    FUNCTIONS_LOADED = False

# Load models at startup (after imports)
try:
    EMBEDDING_MODEL, nlp = load_models()
    MODELS_LOADED = True
except Exception as e:
    print(f"Failed to load models at startup: {e}")
    EMBEDDING_MODEL = None
    nlp = None
    MODELS_LOADED = False

# Skill analysis functions are now imported from utils.py

# Match score calculation is now imported from utils.py

# FastAPI app
app = FastAPI(title="Resume Tailoring API", description="AI-powered resume optimization system")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
llm_config = {
    "connected": False,
    "type": None,
    "info": ""
}

# LLM Client Helper
def get_llm_client_for_api(llm_config_data):
    """Create OpenAI client for API testing with environment variable configuration."""
    llm_type = llm_config_data.get('type', 'local')

    # Use environment variables with sensible defaults for different providers
    if llm_type == 'openai':
        base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        api_key = os.getenv("LLM_API_KEY", llm_config_data.get('api_key', ''))
    else:  # local LLM (LM Studio, Ollama, etc.)
        # Default to LM Studio format, but allow override
        if 'localhost:11434' in llm_config_data.get('server', ''):
            # Ollama - convert to OpenAI-compatible format
            base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
        else:
            # LM Studio or other OpenAI-compatible servers
            base_url = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")

        api_key = os.getenv("LLM_API_KEY", "not-needed")  # Local servers don't need real API keys

    return OpenAI(
        base_url=base_url,
        api_key=api_key
    )

# Pydantic models
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    device: str

class ConnectRequest(BaseModel):
    type: str
    model: Optional[str] = None
    server: Optional[str] = None
    api_key: Optional[str] = None

class ConnectResponse(BaseModel):
    success: bool
    info: str
    error: Optional[str] = None

class JobFetchRequest(BaseModel):
    url: str

class JobFetchResponse(BaseModel):
    text: str

class GenerateResumeRequest(BaseModel):
    resume_text: str
    job_data: Dict[str, Any]

class StrategicAnalysis(BaseModel):
    aspect: str
    priority: str
    analysis: str

class Recommendations(BaseModel):
    strategic_analysis: list[StrategicAnalysis]
    career_focus: list[str]

class GenerateResumeResponse(BaseModel):
    success: bool
    original_score: float
    optimized_score: float
    improvement: float
    tailored_resume: str
    skill_analysis: Dict[str, Any]
    recommendations: Recommendations
    filename: str
    error: Optional[str] = None

class GenerateCoverRequest(BaseModel):
    resume_text: str
    tailored_resume: str
    job_text: str
    keywords: list
    tone: str

class GenerateCoverResponse(BaseModel):
    success: bool
    cover_letter: str
    filename: str
    error: Optional[str] = None

class SendChatRequest(BaseModel):
    message: str
    llm_config: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None

class SendChatResponse(BaseModel):
    success: bool
    response: str
    error: Optional[str] = None

# Routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API is running and models are loaded."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    status = "healthy" if MODELS_LOADED and FUNCTIONS_LOADED else "degraded"

    return HealthResponse(
        status=status,
        timestamp=datetime.now().isoformat(),
        device=device
    )

@app.post("/connect", response_model=ConnectResponse)
async def connect_llm(request: ConnectRequest):
    """Connect to LLM service (local or OpenAI)."""
    try:
        global llm_config

        if request.type == "local":
            # Simulate local LLM connection (in real app, test actual connection)
            llm_config = {
                "connected": True,
                "type": "local",
                "info": f"{request.model or 'Local'} @ {request.server}"
            }

        elif request.type == "openai":
            # Simulate OpenAI connection (in real app, test API key)
            llm_config = {
                "connected": True,
                "type": "openai",
                "info": f"OpenAI {request.model}"
            }

        else:
            raise HTTPException(status_code=400, detail="Invalid connection type")

        return ConnectResponse(
            success=True,
            info=llm_config["info"]
        )

    except Exception as e:
        llm_config = {"connected": False, "type": None, "info": ""}
        return ConnectResponse(
            success=False,
            info="",
            error=str(e)
        )

@app.post("/fetch-job-description", response_model=JobFetchResponse)
async def fetch_job_desc(request: JobFetchRequest):
    """Fetch job description from URL."""
    try:
        text = fetch_job_description(request.url)
        if not text:
            raise HTTPException(status_code=400, detail="Failed to fetch job description from URL")

        return JobFetchResponse(text=text)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/generate-resume", response_model=GenerateResumeResponse)
async def generate_resume(
    resume: UploadFile = File(...),
    job_data: str = Form(...),
    llm_config: str = Form("{}")  # JSON string with LLM configuration
):
    """Generate tailored resume with skill analysis."""
    try:
        # Check if models are loaded
        if not MODELS_LOADED:
            raise HTTPException(
                status_code=503,
                detail="AI models not loaded. Please check server logs and restart if needed."
            )

        if not FUNCTIONS_LOADED:
            raise HTTPException(
                status_code=503,
                detail="Required functions not available. Please check server setup."
            )

        # Parse job data
        job_info = json.loads(job_data)
        job_text = job_info.get("content", "")

        if not job_text:
            raise HTTPException(status_code=400, detail="Job description is required")

        # Extract resume text
        resume_content = await resume.read()
        resume_path = f"/tmp/{resume.filename}"

        with open(resume_path, "wb") as f:
            f.write(resume_content)

        resume_text = extract_resume_text(resume_path)

        # Clean up temp file
        os.remove(resume_path)

        # Calculate initial score
        keywords = extract_keywords(job_text)
        initial_score = calculate_match_score(resume_text, job_text)

        # Parse LLM config
        try:
            llm_info = json.loads(llm_config) if llm_config else {}
        except:
            llm_info = {}

        # Generate tailored resume
        tailored_resume = generate_tailored_resume(resume_text, job_text, keywords, llm_info)
        if not tailored_resume:
            raise HTTPException(status_code=500, detail="Failed to generate tailored resume")

        # Calculate final score
        final_score = calculate_match_score(tailored_resume, job_text)
        improvement = final_score - initial_score

        # Skill analysis
        skill_analysis = analyze_skill_gaps(resume_text, job_text, nlp)

        # Generate strategic recommendations
        recommendations = generate_strategic_recommendations(
            resume_text, job_text, skill_analysis, llm_info
        )

        # Create DOCX file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Tailored_Resume_{timestamp}.docx"
        filepath = create_professional_docx(tailored_resume, filename)

        return GenerateResumeResponse(
            success=True,
            original_score=round(initial_score, 1),
            optimized_score=round(final_score, 1),
            improvement=round(improvement, 1),
            tailored_resume=tailored_resume,
            skill_analysis=skill_analysis,
            recommendations=Recommendations(**recommendations),
            filename=filename
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in generate_resume: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/generate-cover-letter", response_model=GenerateCoverResponse)
async def generate_cover(request: GenerateCoverRequest):
    """Generate cover letter."""
    try:
        # Check if models are loaded
        if not MODELS_LOADED:
            raise HTTPException(
                status_code=503,
                detail="AI models not loaded. Please check server logs and restart if needed."
            )

        if not FUNCTIONS_LOADED:
            raise HTTPException(
                status_code=503,
                detail="Required functions not available. Please check server setup."
            )

        cover_letter = generate_cover_letter(
            resume_text=request.resume_text,
            tailored_resume=request.tailored_resume,
            jd_text=request.job_text,
            tone=request.tone,
            keywords=request.keywords
        )

        if not cover_letter:
            raise HTTPException(status_code=500, detail="Failed to generate cover letter")

        # Create DOCX file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Cover_Letter_{timestamp}.docx"
        filepath = create_cover_letter_docx(cover_letter, filename)

        return GenerateCoverResponse(
            success=True,
            cover_letter=cover_letter,
            filename=filename
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in generate_cover_letter: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/download-resume")
async def download_resume(filename: str = Form(...)):
    """Download generated resume DOCX."""
    try:
        filepath = os.path.join("outputs", filename)
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(
            path=filepath,
            filename=filename,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/download-cover-letter")
async def download_cover_letter(filename: str = Form(...)):
    """Download generated cover letter DOCX."""
    try:
        filepath = os.path.join("outputs", filename)
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(
            path=filepath,
            filename=filename,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_available_models():
    """Get available models for local LLM."""
    try:
        # Try to get models from Ollama (common local LLM server)
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            return {"models": models, "source": "ollama"}

        # Try LM Studio (another common local server)
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [model["id"] for model in data.get("data", [])]
            return {"models": models, "source": "lm-studio"}

        # Fallback: return common model names
        return {
            "models": ["llama2", "llama2:13b", "codellama", "mistral", "dolphin-llama2", "llama3.2"],
            "source": "fallback"
        }

    except Exception as e:
        # Return fallback models if no local servers are found
        return {
            "models": ["llama2", "llama2:13b", "codellama", "mistral", "dolphin-llama2", "llama3.2"],
            "source": "fallback",
            "error": str(e)
        }

@app.post("/test-llm")
async def test_llm_connection(request: ConnectRequest):
    """Test actual LLM connection by sending a simple message."""
    try:
        addDebugMessage(f"Testing LLM connection to {request.type}")

        if request.type == "local":
            # Test local LLM connection using OpenAI-compatible API
            try:
                # Create client with local server configuration
                llm_config_data = {
                    'type': 'local',
                    'server': request.server or "http://localhost:11434",
                    'model': request.model or "local-model"
                }
                client = get_llm_client_for_api(llm_config_data)

                # Test with OpenAI-compatible API
                response = client.chat.completions.create(
                    model=request.model or "local-model",
                    messages=[{"role": "user", "content": "Say 'connected' if you can read this."}],
                    max_tokens=10
                )

                response_text = response.choices[0].message.content.strip()
                addDebugMessage(f"OpenAI-compatible test response: '{response_text}'")

                # Determine source based on server URL
                server_url = request.server or "http://localhost:11434"
                if "localhost:11434" in server_url:
                    source = "ollama"
                elif "localhost:1234" in server_url:
                    source = "lm-studio"
                else:
                    source = "openai-compatible"

                return {"success": True, "response": response_text, "source": source}

            except Exception as e:
                addDebugMessage(f"OpenAI-compatible test failed: {str(e)}")

        elif request.type == "openai":
            # Test OpenAI connection
            try:
                # Create client with OpenAI configuration
                llm_config_data = {
                    'type': 'openai',
                    'api_key': request.api_key,
                    'model': request.model or "gpt-4o-mini"
                }
                client = get_llm_client_for_api(llm_config_data)

                response = client.chat.completions.create(
                    model=request.model or "gpt-4o-mini",
                    messages=[{"role": "user", "content": "Say 'connected' if you can read this."}],
                    max_tokens=10
                )
                response_text = response.choices[0].message.content.strip()
                addDebugMessage(f"OpenAI test response: '{response_text}'")
                return {"success": True, "response": response_text, "source": "openai"}

            except Exception as e:
                addDebugMessage(f"OpenAI test failed: {str(e)}")

        return {"success": False, "error": "Could not connect to LLM", "response": ""}

    except Exception as e:
        addDebugMessage(f"LLM test error: {str(e)}")
        return {"success": False, "error": str(e), "response": ""}

@app.post("/chat", response_model=SendChatResponse)
async def chat_with_llm(request: SendChatRequest):
    """Chat with the LLM for career advice and resume optimization."""
    try:
        # Check if functions are loaded
        if not FUNCTIONS_LOADED:
            raise HTTPException(
                status_code=503,
                detail="Required functions not available. Please check server setup."
            )

        # Process the chat message
        response = process_chat_message(
            message=request.message,
            context=request.context,
            llm_config=request.llm_config
        )

        return SendChatResponse(
            success=True,
            response=response
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.get("/")
async def root():
    """Serve the HTML interface."""
    return FileResponse("index.html", media_type="text/html")

# Debug function for logging
def addDebugMessage(message):
    """Add debug message to console (server-side logging)."""
    print(f"[DEBUG] {message}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

// API service for connecting to the Python backend
const API_BASE_URL = 'http://localhost:8000';

// Note: Frontend is running on port 5174 (automatically assigned by Vite)

// Types for API responses
export interface HealthResponse {
  status: string;
  timestamp: string;
  device: string;
}

export interface ConnectRequest {
  type: 'local' | 'cloud';
  model?: string;
  server?: string;
  api_key?: string;
}

export interface ConnectResponse {
  success: boolean;
  info: string;
  error?: string;
}

export interface JobFetchRequest {
  url: string;
}

export interface JobFetchResponse {
  text: string;
}

export interface GenerateResumeRequest {
  resume: File;
  job_data: {
    type: 'text' | 'url';
    content: string;
  };
  llm_config?: {
    type: 'local' | 'openai';
    model: string;
    server?: string;
    api_key?: string;
  };
}

export interface SkillGapAnalysis {
  coverage_score: number;
  matching_skills: string[];
  missing_skills: string[];
  extra_skills: string[];
  gap_analysis: Array<{
    skill: string;
    priority: string;
    category: string;
    suggestion: string;
    learning_url: string;
  }>;
}

export interface StrategicAnalysis {
  aspect: string;
  priority: string;
  analysis: string;
}

export interface Recommendations {
  strategic_analysis: StrategicAnalysis[];
  career_focus: string[];
}

export interface GenerateResumeResponse {
  success: boolean;
  original_score: number;
  optimized_score: number;
  improvement: number;
  tailored_resume: string;
  skill_analysis: SkillGapAnalysis;
  recommendations: Recommendations;
  filename: string;
  error?: string;
}

export interface GenerateCoverRequest {
  resume_text: string;
  tailored_resume: string;
  job_text: string;
  keywords: string[];
  tone: string;
}

export interface GenerateCoverResponse {
  success: boolean;
  cover_letter: string;
  filename: string;
  error?: string;
}

export interface SendChatRequest {
  message: string;
  llm_config: {
    type: 'local' | 'openai';
    model: string;
    server?: string;
    api_key?: string;
  };
  context?: {
    job_description: string;
    resume_content: string;
    skill_analysis: SkillGapAnalysis;
  } | null;
}

export interface SendChatResponse {
  success: boolean;
  response: string;
  error?: string;
}

// API service functions
export class ApiService {
  static async healthCheck(): Promise<HealthResponse> {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) {
      throw new Error('Backend health check failed');
    }
    return response.json();
  }

  static async connectLLM(request: ConnectRequest): Promise<ConnectResponse> {
    const response = await fetch(`${API_BASE_URL}/connect`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error('Connection failed');
    }

    return response.json();
  }

  static async fetchJobDescription(url: string): Promise<JobFetchResponse> {
    const response = await fetch(`${API_BASE_URL}/fetch-job-description`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ url }),
    });

    if (!response.ok) {
      throw new Error('Failed to fetch job description');
    }

    return response.json();
  }

  static async generateResume(request: GenerateResumeRequest): Promise<GenerateResumeResponse> {
    const formData = new FormData();
    formData.append('resume', request.resume);
    formData.append('job_data', JSON.stringify(request.job_data));
    formData.append('llm_config', JSON.stringify(request.llm_config || {}));

    const response = await fetch(`${API_BASE_URL}/generate-resume`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Resume generation failed');
    }

    return response.json();
  }

  static async generateCoverLetter(request: GenerateCoverRequest): Promise<GenerateCoverResponse> {
    const response = await fetch(`${API_BASE_URL}/generate-cover-letter`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Cover letter generation failed');
    }

    return response.json();
  }

  static async downloadResume(filename: string): Promise<Blob> {
    const response = await fetch(`${API_BASE_URL}/download-resume`, {
      method: 'POST',
      body: new URLSearchParams({ filename }),
    });

    if (!response.ok) {
      throw new Error('Download failed');
    }

    return response.blob();
  }

  static async downloadCoverLetter(filename: string): Promise<Blob> {
    const response = await fetch(`${API_BASE_URL}/download-cover-letter`, {
      method: 'POST',
      body: new URLSearchParams({ filename }),
    });

    if (!response.ok) {
      throw new Error('Download failed');
    }

    return response.blob();
  }

  static async getAvailableModels(): Promise<{ models: string[]; source: string; error?: string }> {
    const response = await fetch(`${API_BASE_URL}/models`);

    if (!response.ok) {
      throw new Error('Failed to fetch models');
    }

    return response.json();
  }

  static async testLLMConnection(request: ConnectRequest): Promise<{ success: boolean; response: string; source?: string; error?: string }> {
    const response = await fetch(`${API_BASE_URL}/test-llm`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error('LLM test failed');
    }

    return response.json();
  }

  static async sendChatMessage(request: SendChatRequest): Promise<SendChatResponse> {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Chat message failed');
    }

    return response.json();
  }
}

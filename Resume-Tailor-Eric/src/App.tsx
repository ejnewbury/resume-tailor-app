import React, { useState, Component, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Terminal, Cpu, Search, BarChart3, Zap, FileText, Layers, ArrowRight, Target, Download, CheckCircle2, AlertCircle, Plus, ExternalLink, ChevronDown, Moon, Sun, Wifi, WifiOff, Loader2, Server, Cloud as CloudIcon, Upload, Link as LinkIcon, MessageSquare, Send } from 'lucide-react';
import { CircularProgress } from './components/CircularProgress';
import { ExpandablePanel } from './components/ExpandablePanel';
import { ApiService, type GenerateResumeResponse, type GenerateCoverResponse, type SkillGapAnalysis } from './services/api';
// --- Types ---
interface AnalysisResult {
  matchPercentage: number;
  matchedSkills: string[];
  missingSkills: string[];
  totalKeywords: number;
}

interface BackendStatus {
  status: 'checking' | 'online' | 'offline';
  lastChecked: Date | null;
}
// --- Mock Data for UI ---
const MOCK_ACTION_PLAN = [{
  skill: 'Docker',
  priority: 'High',
  resource: 'Docker Mastery (Udemy)'
}, {
  skill: 'AWS',
  priority: 'Medium',
  resource: 'AWS Certified Developer'
}, {
  skill: 'GraphQL',
  priority: 'Low',
  resource: 'GraphQL.org Docs'
}];
const MOCK_RESUME_PREVIEW = `ALEXANDER SMITH
Senior Frontend Engineer

PROFESSIONAL SUMMARY
Innovative Senior Frontend Engineer with 7+ years of experience building responsive, user-centric web applications. Proven track record of improving site performance by 40% and leading cross-functional teams.

EXPERIENCE
Senior Frontend Developer | TechFlow Solutions
2020 - Present
- Architected and implemented a new component library using React and TypeScript...`;
const MOCK_COVER_LETTER_PREVIEW = `Dear Hiring Manager,

I am writing to express my strong interest in the Senior Frontend Engineer position at [Company Name]. With my extensive background in React ecosystem and a passion for building scalable UI architectures, I am confident in my ability to contribute effectively to your engineering team.

Throughout my career, I have focused on...`;
// --- Main Component ---
export function App() {
  const [skills, setSkills] = useState('React, TypeScript, Tailwind CSS, Node.js, UI Design');
  const [jobDesc, setJobDesc] = useState('We are looking for a Senior Frontend Engineer with strong experience in React, TypeScript, and Tailwind CSS. Knowledge of Node.js is a plus. Experience with AWS and Docker is preferred.');
  const [jobUrl, setJobUrl] = useState('');
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [hasAnalyzed, setHasAnalyzed] = useState(false);
  const [coverLetterTone, setCoverLetterTone] = useState('Professional');
  const [darkMode, setDarkMode] = useState(true);

  // Connection states
  const [connectionType, setConnectionType] = useState<'local' | 'cloud'>('local');
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'connecting'>('disconnected');
  const [localModel, setLocalModel] = useState('llama3.2');
  const [serverAddress, setServerAddress] = useState('http://localhost:11434');
  const [cloudModel, setCloudModel] = useState('gpt-4o-mini');
  const [apiKey, setApiKey] = useState('');

  // Backend and API states
  const [backendStatus, setBackendStatus] = useState<BackendStatus>({ status: 'checking', lastChecked: null });
  const [analysisResults, setAnalysisResults] = useState<GenerateResumeResponse | null>(null);
  const [coverLetterResults, setCoverLetterResults] = useState<GenerateCoverResponse | null>(null);
  const [isGeneratingCoverLetter, setIsGeneratingCoverLetter] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [debugMessages, setDebugMessages] = useState<string[]>([]);
  const [showDebug, setShowDebug] = useState(false);
  const [showChat, setShowChat] = useState(false);
  const [chatMessages, setChatMessages] = useState<Array<{role: 'user' | 'assistant', content: string}>>([]);
  const [chatInput, setChatInput] = useState('');
  const [isTestingLLM, setIsTestingLLM] = useState(false);

  // Debug helper
  const addDebugMessage = (message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setDebugMessages(prev => [...prev.slice(-9), `[${timestamp}] ${message}`]); // Keep last 10 messages
  };


  // Test LLM connection
  const testLLMConnection = async () => {
    setIsTestingLLM(true);
    addDebugMessage("Testing LLM connection...");

    try {
      const connectRequest = connectionType === 'local'
        ? {
            type: 'local' as const,
            model: localModel,
            server: serverAddress
          }
        : {
            type: 'openai' as const,
            model: cloudModel,
            api_key: apiKey
          };

      const result = await ApiService.testLLMConnection(connectRequest);

      if (result.success) {
        addDebugMessage(`✅ LLM test successful: "${result.response}" (via ${result.source})`);
        setChatMessages(prev => [...prev, {
          role: 'assistant',
          content: `LLM Test Successful!\n\nResponse: "${result.response}"\nSource: ${result.source}\n\nThe LLM is working correctly.`
        }]);
        return true;
      } else {
        addDebugMessage(`❌ LLM test failed: ${result.error}`);
        setChatMessages(prev => [...prev, {
          role: 'assistant',
          content: `LLM Test Failed!\n\nError: ${result.error}\n\nThe LLM is not responding. Check your server and connection settings.`
        }]);
        return false;
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Unknown error';
      addDebugMessage(`❌ LLM test error: ${errorMsg}`);
      setChatMessages(prev => [...prev, {
        role: 'assistant',
        content: `LLM Test Error!\n\n${errorMsg}\n\nCheck your network connection and LLM server.`
      }]);
      return false;
    } finally {
      setIsTestingLLM(false);
    }
  };

  // Send chat message to LLM
  const handleSendChatMessage = async () => {
    if (!chatInput.trim()) return;

    const userMessage = chatInput.trim();
    setChatInput('');
    setChatMessages(prev => [...prev, { role: 'user', content: userMessage }]);

    try {
      addDebugMessage(`Sending chat message: "${userMessage}"`);

      // Prepare LLM config for chat
      const llmConfig = connectionType === 'local'
        ? {
            type: 'local' as const,
            model: localModel,
            server: serverAddress
          }
        : {
            type: 'openai' as const,
            model: cloudModel,
            api_key: apiKey
          };

      // Send chat message to backend
      const chatResult = await ApiService.sendChatMessage({
        message: userMessage,
        llm_config: llmConfig,
        context: analysisResults ? {
          job_description: analysisResults.job_text || '',
          resume_content: analysisResults.tailored_resume,
          skill_analysis: analysisResults.skill_analysis
        } : null
      });

      setChatMessages(prev => [...prev, {
        role: 'assistant',
        content: chatResult.response
      }]);

      addDebugMessage(`Chat response received: ${chatResult.response.substring(0, 100)}...`);

    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Unknown error';
      addDebugMessage(`Chat message failed: ${errorMsg}`);
      setChatMessages(prev => [...prev, {
        role: 'assistant',
        content: `❌ Sorry, I couldn't process your message. Error: ${errorMsg}\n\nPlease check your LLM connection and try again.`
      }]);
    }
  };

  // Check backend health on mount and periodically
  useEffect(() => {
    const checkBackendHealth = async () => {
      const wasOffline = backendStatus.status === 'offline';

      try {
        const health = await ApiService.healthCheck();
        const isNowOnline = (health.status === 'healthy' || health.status === 'degraded');

        setBackendStatus({
          status: isNowOnline ? 'online' : 'offline',
          lastChecked: new Date()
        });

        // Backend status updated
        if (wasOffline && isNowOnline) {
          addDebugMessage("Backend came online");
        }
      } catch (err) {
        setBackendStatus({
          status: 'offline',
          lastChecked: new Date()
        });
      }
    };

    // Initial check
    checkBackendHealth();

    // Set up periodic checks every 10 seconds
    const interval = setInterval(checkBackendHealth, 10000);

    return () => clearInterval(interval);
  }, [backendStatus.status]);

  const handleAnalyze = async () => {
    if (!resumeFile || (!jobDesc.trim() && !jobUrl.trim())) {
      setError('Please upload a resume and provide a job description or URL.');
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    try {
      // Prepare job data
      let jobData = { type: 'text' as const, content: jobDesc };

      // If URL is provided and no manual text, fetch from URL
      if (jobUrl.trim() && !jobDesc.trim()) {
        const fetchResult = await ApiService.fetchJobDescription(jobUrl);
        jobData = { type: 'url', content: fetchResult.text };
      }

      // Generate resume
      const llmConfig = connectionType === 'local'
        ? {
            type: 'local' as const,
            model: localModel,
            server: serverAddress
          }
        : {
            type: 'openai' as const,
            model: cloudModel,
            api_key: apiKey
          };

      const results = await ApiService.generateResume({
        resume: resumeFile,
        job_data: jobData,
        llm_config: llmConfig
      });

      setAnalysisResults(results);
      setHasAnalyzed(true);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred during analysis');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleGenerateCoverLetter = async () => {
    if (!analysisResults) {
      setError('Please generate a resume first.');
      return;
    }

    setIsGeneratingCoverLetter(true);
    setError(null);

    try {
      const coverResult = await ApiService.generateCoverLetter({
        resume_text: analysisResults.tailored_resume,
        tailored_resume: analysisResults.tailored_resume,
        job_text: analysisResults.job_text || '',
        keywords: analysisResults.skill_analysis.matching_skills,
        tone: coverLetterTone.toLowerCase()
      });

      setCoverLetterResults(coverResult);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Cover letter generation failed');
    } finally {
      setIsGeneratingCoverLetter(false);
    }
  };

  const handleDownloadResume = async () => {
    if (!analysisResults) return;

    try {
      const blob = await ApiService.downloadResume(analysisResults.filename);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = analysisResults.filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      setError('Failed to download resume');
    }
  };

  const handleDownloadCoverLetter = async () => {
    if (!coverLetterResults) return;

    try {
      const blob = await ApiService.downloadCoverLetter(coverLetterResults.filename);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = coverLetterResults.filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      setError('Failed to download cover letter');
    }
  };

  const handleConnect = async () => {
    setConnectionStatus('connecting');
    setError(null);

    addDebugMessage(`Attempting to connect to ${connectionType.toUpperCase()} LLM...`);

    try {
      const connectRequest = connectionType === 'local'
        ? {
            type: 'local' as const,
            model: localModel,
            server: serverAddress
          }
        : {
            type: 'openai' as const,
            model: cloudModel,
            api_key: apiKey ? '***' + apiKey.slice(-4) : ''
          };

      addDebugMessage(`Connection request: ${JSON.stringify(connectRequest, null, 2)}`);

      const result = await ApiService.connectLLM(connectRequest);

      if (result.success) {
        // Now test actual LLM connectivity
        addDebugMessage(`API connection successful: ${result.info}`);
        addDebugMessage("Testing actual LLM communication...");

        const llmTestResult = await testLLMConnection();
        if (llmTestResult) {
          setConnectionStatus('connected');
          addDebugMessage(`✅ LLM connection fully verified`);
          setChatMessages(prev => [...prev, {
            role: 'assistant',
            content: `LLM Connected Successfully!\n\nAPI: ${result.info}\n\nThe LLM is responding correctly and ready for use.`
          }]);
        } else {
          setConnectionStatus('disconnected');
          addDebugMessage(`❌ LLM test failed despite API connection`);
          setError('API connected but LLM not responding');
        }
      } else {
        setConnectionStatus('disconnected');
        addDebugMessage(`❌ API connection failed: ${result.error || 'Unknown error'}`);
        setError(result.error || 'Connection failed');
      }

    } catch (err) {
      setConnectionStatus('disconnected');
      const errorMessage = err instanceof Error ? err.message : 'Connection failed';
      addDebugMessage(`❌ Connection error: ${errorMessage}`);
      setError(errorMessage);
    }
  };
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setResumeFile(file);
    }
  };
  return <div className={`min-h-screen ${darkMode ? 'bg-[#1a1d23] text-gray-200' : 'bg-gray-50 text-gray-900'} font-sans selection:bg-blue-500/30 selection:text-blue-200 relative overflow-hidden transition-colors duration-300`}>
      {/* Background Grid */}
      <div className={`fixed inset-0 bg-grid-pattern ${darkMode ? 'opacity-30' : 'opacity-10'} pointer-events-none z-0`} />

      {/* Top Toolbar */}
      <header className={`relative z-20 h-16 border-b ${darkMode ? 'border-gray-800 bg-[#1a1d23]/90' : 'border-gray-200 bg-white/90'} backdrop-blur-md flex items-center justify-between px-6 transition-colors duration-300`}>
        <div className="flex items-center gap-3">
          <div className={`p-2 ${darkMode ? 'bg-blue-500/10 border-blue-500/20' : 'bg-blue-500/10 border-blue-500/30'} rounded border`}>
            <Terminal className="w-5 h-5 text-blue-400" />
          </div>
          <h1 className={`text-lg font-bold tracking-wider ${darkMode ? 'text-gray-100' : 'text-gray-900'} flex items-center gap-2`}>
            SKILLS<span className="text-blue-500">COMMAND</span>CENTER
          </h1>
        </div>

        <div className="flex items-center gap-4">
          <div className={`flex items-center gap-2 px-3 py-1 rounded-full ${darkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-gray-100 border-gray-300'} border`}>
            <div className={`w-2 h-2 rounded-full ${isAnalyzing ? 'bg-amber-400 animate-pulse' : 'bg-green-400'}`} />
            <span className={`text-xs font-mono ${darkMode ? 'text-gray-400' : 'text-gray-600'} uppercase`}>
              {isAnalyzing ? 'PROCESSING' : 'SYSTEM READY'}
            </span>
          </div>

          {/* Dark Mode Toggle */}
          <button onClick={() => setDarkMode(!darkMode)} className={`p-2 rounded-lg ${darkMode ? 'bg-gray-800 hover:bg-gray-700 text-yellow-400' : 'bg-gray-200 hover:bg-gray-300 text-gray-700'} transition-all duration-300 group`} aria-label="Toggle dark mode">
            {darkMode ? <Sun className="w-5 h-5 group-hover:rotate-45 transition-transform duration-300" /> : <Moon className="w-5 h-5 group-hover:-rotate-12 transition-transform duration-300" />}
          </button>
        </div>
      </header>

      {/* Main Content - Horizontal Slider Layout */}
      <main className="relative z-10 max-w-[2800px] mx-auto p-4 md:p-6 lg:p-8 h-[calc(100vh-64px)] overflow-hidden">
        {/* Horizontal Slider Container */}
        <div className="flex gap-6 h-full overflow-x-auto overflow-y-hidden scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-gray-800">
          {/* 1. CONNECTION SIDEBAR */}
          <section className="w-[12%] h-full overflow-y-auto pr-2 custom-scrollbar flex-shrink-0">
            <div className="space-y-4">
              {/* Panel Header */}
              <div className="flex items-center gap-2 mb-1">
                <Wifi className="w-4 h-4 text-blue-400" />
                <h2 className={`text-xs font-bold tracking-widest ${darkMode ? 'text-gray-500' : 'text-gray-400'} uppercase`}>
                  Connection
                </h2>
              </div>

              {/* Connection Status */}
              <div className={`${darkMode ? 'bg-[#22262e] border-gray-800' : 'bg-white border-gray-200'} border rounded-lg p-3 shadow-lg`}>
                <div className="flex items-center justify-between mb-2">
                  <span className={`text-[10px] font-bold ${darkMode ? 'text-gray-400' : 'text-gray-500'} uppercase tracking-wider`}>
                    Status
                  </span>
                  {connectionStatus === 'connected' && <Wifi className="w-3 h-3 text-green-400" />}
                  {connectionStatus === 'disconnected' && <WifiOff className="w-3 h-3 text-gray-500" />}
                  {connectionStatus === 'connecting' && <Loader2 className="w-3 h-3 text-blue-400 animate-spin" />}
                </div>
                <div className={`text-xs font-mono ${connectionStatus === 'connected' ? 'text-green-400' : connectionStatus === 'connecting' ? 'text-blue-400' : 'text-gray-500'}`}>
                  {connectionStatus === 'connected' && 'ONLINE'}
                  {connectionStatus === 'connecting' && 'CONNECTING...'}
                  {connectionStatus === 'disconnected' && 'OFFLINE'}
                </div>
              </div>

              {/* Backend Status */}
              <div className={`${darkMode ? 'bg-[#22262e] border-gray-800' : 'bg-white border-gray-200'} border rounded-lg p-3 shadow-lg`}>
                <div className="flex items-center justify-between mb-2">
                  <span className={`text-[10px] font-bold ${darkMode ? 'text-gray-400' : 'text-gray-500'} uppercase tracking-wider`}>
                    Backend
                  </span>
                  {backendStatus.status === 'online' && <Server className="w-3 h-3 text-green-400" />}
                  {backendStatus.status === 'offline' && <Server className="w-3 h-3 text-red-400" />}
                  {backendStatus.status === 'checking' && <Loader2 className="w-3 h-3 text-yellow-400 animate-spin" />}
                </div>
                <div className={`text-xs font-mono ${backendStatus.status === 'online' ? 'text-green-400' : backendStatus.status === 'checking' ? 'text-yellow-400' : 'text-red-400'}`}>
                  {backendStatus.status === 'online' && 'ONLINE'}
                  {backendStatus.status === 'checking' && 'CHECKING...'}
                  {backendStatus.status === 'offline' && 'OFFLINE'}
                </div>
              </div>

              {/* Connection Type Toggle */}
              <div className={`${darkMode ? 'bg-[#22262e] border-gray-800' : 'bg-white border-gray-200'} border rounded-lg p-1 shadow-lg`}>
                <div className="grid grid-cols-2 gap-1">
                  <button onClick={() => setConnectionType('local')} className={`py-2 px-2 rounded text-[10px] font-bold uppercase tracking-wider transition-all ${connectionType === 'local' ? 'bg-blue-600 text-white' : darkMode ? 'bg-transparent text-gray-400 hover:text-gray-300' : 'bg-transparent text-gray-600 hover:text-gray-900'}`}>
                    <Server className="w-3 h-3 mx-auto mb-1" />
                    Local
                  </button>
                  <button onClick={() => setConnectionType('cloud')} className={`py-2 px-2 rounded text-[10px] font-bold uppercase tracking-wider transition-all ${connectionType === 'cloud' ? 'bg-blue-600 text-white' : darkMode ? 'bg-transparent text-gray-400 hover:text-gray-300' : 'bg-transparent text-gray-600 hover:text-gray-900'}`}>
                    <CloudIcon className="w-3 h-3 mx-auto mb-1" />
                    Cloud
                  </button>
                </div>
              </div>

              {/* Connection Settings */}
              <div className={`${darkMode ? 'bg-[#22262e] border-gray-800' : 'bg-white border-gray-200'} border rounded-lg p-4 shadow-lg space-y-4`}>
                {connectionType === 'local' ? <>
                    <div>
                      <label className={`text-[10px] font-bold ${darkMode ? 'text-gray-400' : 'text-gray-500'} uppercase tracking-wider block mb-2`}>
                        Model ID
                      </label>
                      <input
                        type="text"
                        value={localModel}
                        onChange={e => setLocalModel(e.target.value)}
                        placeholder="llama3.2, dolphin-llama3:latest, etc."
                        className={`w-full ${darkMode ? 'bg-[#1a1d23] border-gray-700 text-gray-300' : 'bg-gray-50 border-gray-300 text-gray-900'} border rounded p-2 text-xs font-mono focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500/20`}
                      />
                    </div>
                    <div>
                      <label className={`text-[10px] font-bold ${darkMode ? 'text-gray-400' : 'text-gray-500'} uppercase tracking-wider block mb-2`}>
                        Server
                      </label>
                      <input type="text" value={serverAddress} onChange={e => setServerAddress(e.target.value)} placeholder="localhost:11434" className={`w-full ${darkMode ? 'bg-[#1a1d23] border-gray-700 text-gray-300' : 'bg-gray-50 border-gray-300 text-gray-900'} border rounded p-2 text-xs font-mono focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500/20`} />
                    </div>
                  </> : <>
                    <div>
                      <label className={`text-[10px] font-bold ${darkMode ? 'text-gray-400' : 'text-gray-500'} uppercase tracking-wider block mb-2`}>
                        Model
                      </label>
                      <div className="relative">
                        <select value={cloudModel} onChange={e => setCloudModel(e.target.value)} className={`w-full appearance-none ${darkMode ? 'bg-[#1a1d23] border-gray-700 text-gray-300' : 'bg-gray-50 border-gray-300 text-gray-900'} border rounded p-2 text-xs font-mono focus:outline-none focus:border-blue-500 cursor-pointer`}>
                          <optgroup label="OpenAI">
                            <option value="gpt-4">GPT-4</option>
                            <option value="gpt-4-turbo">GPT-4 Turbo</option>
                            <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
                          </optgroup>
                          <optgroup label="Anthropic">
                            <option value="claude-3-opus">Claude 3 Opus</option>
                            <option value="claude-3-sonnet">
                              Claude 3 Sonnet
                            </option>
                            <option value="claude-3-haiku">
                              Claude 3 Haiku
                            </option>
                          </optgroup>
                        </select>
                        <ChevronDown className={`w-3 h-3 ${darkMode ? 'text-gray-500' : 'text-gray-400'} absolute right-2 top-1/2 -translate-y-1/2 pointer-events-none`} />
                      </div>
                    </div>
                    <div>
                      <label className={`text-[10px] font-bold ${darkMode ? 'text-gray-400' : 'text-gray-500'} uppercase tracking-wider block mb-2`}>
                        API Key
                      </label>
                      <input type="password" value={apiKey} onChange={e => setApiKey(e.target.value)} placeholder="sk-..." className={`w-full ${darkMode ? 'bg-[#1a1d23] border-gray-700 text-gray-300' : 'bg-gray-50 border-gray-300 text-gray-900'} border rounded p-2 text-xs font-mono focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500/20`} />
                    </div>
                  </>}

                <button onClick={handleConnect} disabled={connectionStatus === 'connecting'} className="w-full py-2 bg-blue-600 hover:bg-blue-500 text-white text-[10px] font-bold uppercase tracking-wider rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed">
                  {connectionStatus === 'connecting' ? 'Connecting...' : 'Connect'}
                </button>
              </div>
            </div>
          </section>

          {/* 2. INPUT PARAMETERS */}
          <section className="w-[15%] min-w-[280px] flex flex-col gap-6 h-full overflow-y-auto pr-2 custom-scrollbar flex-shrink-0">
            {/* Panel Header */}
            <div className="flex items-center gap-2 mb-1">
              <Layers className="w-4 h-4 text-blue-400" />
              <h2 className={`text-xs font-bold tracking-widest ${darkMode ? 'text-gray-500' : 'text-gray-400'} uppercase`}>
                Input Parameters
              </h2>
            </div>

            {/* Candidate Resume Upload */}
            <div className={`${darkMode ? 'bg-[#22262e] border-gray-800' : 'bg-white border-gray-200'} border rounded-lg p-5 transition-all duration-300 group focus-within:border-blue-500/50 shadow-lg flex-1 flex flex-col`}>
              <label className="flex items-center justify-between mb-3">
                <span className="text-xs font-bold tracking-widest text-blue-400 uppercase flex items-center gap-2">
                  <FileText className="w-3 h-3" /> Candidate Resume
                </span>
                <span className={`text-[10px] ${darkMode ? 'text-gray-500' : 'text-gray-400'} font-mono`}>
                  PDF / DOCX
                </span>
              </label>

              <div className="flex-1 flex flex-col items-center justify-center">
                <input type="file" id="resume-upload" accept=".pdf,.doc,.docx" onChange={handleFileUpload} className="hidden" />
                <label htmlFor="resume-upload" className={`w-full h-full min-h-[200px] flex flex-col items-center justify-center ${darkMode ? 'bg-[#1a1d23] border-gray-700 hover:border-blue-500' : 'bg-gray-50 border-gray-300 hover:border-blue-500'} border-2 border-dashed rounded-lg cursor-pointer transition-all group-hover:border-blue-500/50`}>
                  <Upload className={`w-12 h-12 mb-3 ${darkMode ? 'text-gray-600' : 'text-gray-400'}`} />
                  {resumeFile ? <div className="text-center">
                      <p className="text-sm font-mono text-blue-400 mb-1">
                        {resumeFile.name}
                      </p>
                      <p className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                        Click to change file
                      </p>
                    </div> : <div className="text-center">
                      <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'} mb-1`}>
                        Click to upload resume
                      </p>
                      <p className={`text-xs ${darkMode ? 'text-gray-600' : 'text-gray-400'}`}>
                        or drag and drop
                      </p>
                    </div>}
                </label>
              </div>
            </div>

            {/* Job Description Input */}
            <div className={`${darkMode ? 'bg-[#22262e] border-gray-800' : 'bg-white border-gray-200'} border rounded-lg p-5 transition-all duration-300 flex-1 flex flex-col group focus-within:border-blue-500/50 shadow-lg`}>
              <label className="flex items-center justify-between mb-3">
                <span className="text-xs font-bold tracking-widest text-blue-400 uppercase flex items-center gap-2">
                  <FileText className="w-3 h-3" /> Job Description
                </span>
                <span className={`text-[10px] ${darkMode ? 'text-gray-500' : 'text-gray-400'} font-mono`}>
                  TEXT / URL
                </span>
              </label>

              {/* URL Input */}
              <div className="mb-3">
                <div className="relative">
                  <LinkIcon className={`absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 ${darkMode ? 'text-gray-600' : 'text-gray-400'}`} />
                  <input type="url" value={jobUrl} onChange={e => setJobUrl(e.target.value)} placeholder="https://example.com/job-posting" className={`w-full ${darkMode ? 'bg-[#1a1d23] border-gray-700 text-gray-300' : 'bg-gray-50 border-gray-300 text-gray-900'} border rounded p-2 pl-10 text-xs font-mono focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500/20`} />
                </div>
              </div>

              <textarea value={jobDesc} onChange={e => setJobDesc(e.target.value)} className={`w-full flex-grow ${darkMode ? 'bg-[#1a1d23] border-gray-700 text-gray-300' : 'bg-gray-50 border-gray-300 text-gray-900'} border rounded p-3 text-sm font-mono focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500/20 resize-none transition-all`} placeholder="Paste job description here..." />
            </div>

            {/* Action Button */}
            <button onClick={handleAnalyze} disabled={isAnalyzing} className="group relative w-full py-4 bg-blue-600 hover:bg-blue-500 text-white font-bold tracking-widest uppercase text-sm rounded-lg overflow-hidden transition-all duration-300 shadow-lg shadow-blue-900/20 disabled:opacity-70 disabled:cursor-not-allowed">
              <div className="absolute inset-0 bg-white/10 translate-y-full group-hover:translate-y-0 transition-transform duration-300" />
              <span className="relative flex items-center justify-center gap-2">
                {isAnalyzing ? <>
                    <Zap className="w-4 h-4 animate-pulse" /> Processing Data...
                  </> : <>
                    Run Analysis{' '}
                    <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                  </>}
              </span>
            </button>
          </section>

          {/* 3. ANALYSIS METRICS */}
          <section className="w-[30%] min-w-[450px] h-full overflow-y-auto px-2 custom-scrollbar relative flex-shrink-0">
            <motion.div initial={{
            opacity: 0,
            y: 20
          }} animate={{
            opacity: 1,
            y: 0
          }} transition={{
            duration: 0.5
          }} className="space-y-8 pb-8">
              {/* 1. ANALYSIS OUTPUT SECTION */}
              <div className="space-y-4">
                <div className="flex items-center gap-2 mb-1">
                  <BarChart3 className="w-4 h-4 text-blue-400" />
                  <h2 className={`text-xs font-bold tracking-widest ${darkMode ? 'text-gray-500' : 'text-gray-400'} uppercase`}>
                    Analysis Output
                  </h2>
                  <div className={`h-px ${darkMode ? 'bg-gray-800' : 'bg-gray-300'} flex-grow ml-4`} />
                </div>

                <div className="grid grid-cols-3 gap-4">
                  {/* Original Score */}
                  <div className={`${darkMode ? 'bg-[#22262e] border-gray-800' : 'bg-white border-gray-200'} border rounded-lg p-6 flex flex-col items-center justify-center relative overflow-hidden group hover:border-blue-500/30 transition-all shadow-lg`}>
                    {hasAnalyzed && analysisResults ? <>
                        <div className="absolute inset-0 bg-gradient-to-b from-blue-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                        <CircularProgress percentage={Math.round(analysisResults.original_score)} size={120} strokeWidth={8} label="ORIGINAL" />
                      </> : <div className="flex flex-col items-center justify-center h-[120px]">
                        <div className={`w-16 h-16 rounded-full ${darkMode ? 'bg-gray-800' : 'bg-gray-100'} flex items-center justify-center mb-2`}>
                          <BarChart3 className={`w-6 h-6 ${darkMode ? 'text-gray-700' : 'text-gray-300'}`} />
                        </div>
                        <span className={`text-xs font-bold tracking-widest ${darkMode ? 'text-gray-600' : 'text-gray-400'} uppercase`}>
                          Original
                        </span>
                      </div>}
                  </div>

                  {/* Improvement */}
                  <div className={`${darkMode ? 'bg-[#22262e] border-gray-800' : 'bg-white border-gray-200'} border rounded-lg p-6 flex flex-col items-center justify-center relative overflow-hidden group hover:border-green-500/30 transition-all shadow-lg`}>
                    {hasAnalyzed && analysisResults ? <>
                        <div className="absolute inset-0 bg-gradient-to-b from-green-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                        <div className="relative">
                          <CircularProgress percentage={Math.round(analysisResults.improvement)} size={120} strokeWidth={8} label="GAIN" />
                          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                            <span className="text-green-400 text-xl font-bold translate-y-[-15px] translate-x-[-24px]">
                              +
                            </span>
                          </div>
                        </div>
                      </> : <div className="flex flex-col items-center justify-center h-[120px]">
                        <div className={`w-16 h-16 rounded-full ${darkMode ? 'bg-gray-800' : 'bg-gray-100'} flex items-center justify-center mb-2`}>
                          <Plus className={`w-6 h-6 ${darkMode ? 'text-gray-700' : 'text-gray-300'}`} />
                        </div>
                        <span className={`text-xs font-bold tracking-widest ${darkMode ? 'text-gray-600' : 'text-gray-400'} uppercase`}>
                          Improvement
                        </span>
                      </div>}
                  </div>

                  {/* Optimized Score */}
                  <div className={`${darkMode ? 'bg-[#22262e] border-gray-800' : 'bg-white border-gray-200'} border rounded-lg p-6 flex flex-col items-center justify-center relative overflow-hidden group hover:border-purple-500/30 transition-all shadow-lg`}>
                    {hasAnalyzed && analysisResults ? <>
                        <div className="absolute inset-0 bg-gradient-to-b from-purple-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                        <CircularProgress percentage={Math.round(analysisResults.optimized_score)} size={120} strokeWidth={8} label="OPTIMIZED" />
                      </> : <div className="flex flex-col items-center justify-center h-[120px]">
                        <div className={`w-16 h-16 rounded-full ${darkMode ? 'bg-gray-800' : 'bg-gray-100'} flex items-center justify-center mb-2`}>
                          <Target className={`w-6 h-6 ${darkMode ? 'text-gray-700' : 'text-gray-300'}`} />
                        </div>
                        <span className={`text-xs font-bold tracking-widest ${darkMode ? 'text-gray-600' : 'text-gray-400'} uppercase`}>
                          Optimized
                        </span>
                      </div>}
                  </div>
                </div>
              </div>

              {/* 2. SKILLS GAP ANALYSIS SECTION */}
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Target className="w-4 h-4 text-blue-400" />
                  <h2 className={`text-xs font-bold tracking-widest ${darkMode ? 'text-gray-500' : 'text-gray-400'} uppercase`}>
                    Skills Gap Analysis
                  </h2>
                  <div className={`h-px ${darkMode ? 'bg-gray-800' : 'bg-gray-300'} flex-grow ml-4`} />
                </div>

                <div className={`${darkMode ? 'bg-[#22262e] border-gray-800' : 'bg-white border-gray-200'} border rounded-lg p-6 shadow-lg`}>
                  {hasAnalyzed ? <>
                      {/* Coverage Metric */}
                      <div className="flex flex-col items-center justify-center mb-6 pb-6 border-b border-gray-800">
                        <CircularProgress percentage={analysisResults.skill_analysis.coverage_score} size={100} strokeWidth={8} label="COVERAGE" />
                      </div>

                      {/* Keyword Lists */}
                      <div className="grid grid-cols-3 gap-4 mb-6">
                        <div className="space-y-2">
                          <div className="flex items-center gap-2 text-green-400 mb-2">
                            <CheckCircle2 className="w-3 h-3" />
                            <span className="text-[10px] font-bold uppercase tracking-wider">
                              Match
                            </span>
                          </div>
                          <div className={`text-2xl font-mono font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                            {analysisResults.skill_analysis.matching_skills.length}
                          </div>
                        </div>

                        <div className="space-y-2">
                          <div className="flex items-center gap-2 text-amber-400 mb-2">
                            <AlertCircle className="w-3 h-3" />
                            <span className="text-[10px] font-bold uppercase tracking-wider">
                              Miss
                            </span>
                          </div>
                          <div className={`text-2xl font-mono font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                            {analysisResults.skill_analysis.missing_skills.length}
                          </div>
                        </div>

                        <div className="space-y-2">
                          <div className="flex items-center gap-2 text-blue-400 mb-2">
                            <Plus className="w-3 h-3" />
                            <span className="text-[10px] font-bold uppercase tracking-wider">
                              Extra
                            </span>
                          </div>
                          <div className={`text-2xl font-mono font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                            {analysisResults.skill_analysis.extra_skills.length}
                          </div>
                        </div>
                      </div>

                      {/* Skills Breakdown */}
                      <div className={`pt-6 space-y-4 ${darkMode ? 'border-gray-800' : 'border-gray-200'} border-t`}>
                        <div>
                          <h3 className={`text-[10px] font-bold ${darkMode ? 'text-gray-400' : 'text-gray-500'} uppercase tracking-wider mb-2`}>
                            Matching Skills
                          </h3>
                          <div className="flex flex-wrap gap-1.5">
                            {analysisResults.skill_analysis.matching_skills.map(skill => <span key={skill} className="px-2 py-0.5 bg-green-500/10 border border-green-500/20 text-green-400 text-[10px] rounded font-mono">
                                {skill}
                              </span>)}
                          </div>
                        </div>
                        <div>
                          <h3 className={`text-[10px] font-bold ${darkMode ? 'text-gray-400' : 'text-gray-500'} uppercase tracking-wider mb-2`}>
                            Additional Skills
                          </h3>
                          <div className="flex flex-wrap gap-1.5">
                            {analysisResults.skill_analysis.extra_skills.map(skill => <span key={skill} className="px-2 py-0.5 bg-blue-500/10 border border-blue-500/20 text-blue-400 text-[10px] rounded font-mono">
                                {skill}
                              </span>)}
                          </div>
                        </div>
                      </div>
                    </> : <div className="flex flex-col items-center justify-center py-12">
                      <div className={`w-16 h-16 rounded-full ${darkMode ? 'bg-gray-800' : 'bg-gray-100'} flex items-center justify-center mb-4`}>
                        <Target className={`w-8 h-8 ${darkMode ? 'text-gray-700' : 'text-gray-300'}`} />
                      </div>
                      <p className={`text-sm ${darkMode ? 'text-gray-500' : 'text-gray-400'} font-mono`}>
                        Analysis pending
                      </p>
                    </div>}
                </div>
              </div>
            </motion.div>
          </section>

          {/* 4. GENERATED OUTPUT */}
          <section className="w-[15%] min-w-[320px] h-full overflow-y-auto px-2 custom-scrollbar relative flex-shrink-0">
            <motion.div initial={{
            opacity: 0,
            y: 20
          }} animate={{
            opacity: 1,
            y: 0
          }} transition={{
            duration: 0.5,
            delay: 0.3
          }} className="space-y-8 pb-8">
              {/* RESUME GENERATOR OUTPUT SECTION */}
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <FileText className="w-4 h-4 text-blue-400" />
                  <h2 className={`text-xs font-bold tracking-widest ${darkMode ? 'text-gray-500' : 'text-gray-400'} uppercase`}>
                    Generated Output
                  </h2>
                  <div className={`h-px ${darkMode ? 'bg-gray-800' : 'bg-gray-300'} flex-grow ml-4`} />
                </div>

                <div className="grid grid-cols-1 gap-4">
                  {/* Tailored Resume */}
                  <div className={`${darkMode ? 'bg-[#22262e] border-gray-800' : 'bg-white border-gray-200'} border rounded-lg p-1 flex flex-col shadow-lg h-[300px]`}>
                    <div className={`${darkMode ? 'bg-[#1a1d23] border-gray-800' : 'bg-gray-100 border-gray-200'} border-b p-2 flex items-center justify-between rounded-t-lg`}>
                      <span className={`text-[10px] font-bold ${darkMode ? 'text-gray-400' : 'text-gray-600'} uppercase tracking-wider`}>
                        Tailored Resume
                      </span>
                      <button onClick={handleDownloadResume} disabled={!analysisResults} className="flex items-center gap-1.5 px-2 py-1 bg-blue-600 hover:bg-blue-500 text-white text-[10px] font-bold uppercase tracking-wider rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed">
                        <Download className="w-3 h-3" /> PDF
                      </button>
                    </div>
                    <div className={`flex-grow ${darkMode ? 'bg-white' : 'bg-gray-50'} p-4 overflow-y-auto custom-scrollbar rounded-b-lg`}>
                      {hasAnalyzed && analysisResults ? <pre className="font-sans text-gray-800 text-xs whitespace-pre-wrap leading-relaxed">
                          {analysisResults.tailored_resume}
                        </pre> : <div className="h-full flex flex-col items-center justify-center text-gray-400">
                          <FileText className="w-10 h-10 mb-2 opacity-20" />
                          <p className="text-xs">Resume preview</p>
                        </div>}
                    </div>
                  </div>

                  {/* Cover Letter */}
                  <div className={`${darkMode ? 'bg-[#22262e] border-gray-800' : 'bg-white border-gray-200'} border rounded-lg p-1 flex flex-col shadow-lg h-[300px]`}>
                    <div className={`${darkMode ? 'bg-[#1a1d23] border-gray-800' : 'bg-gray-100 border-gray-200'} border-b p-2 flex items-center justify-between rounded-t-lg gap-2`}>
                      <span className={`text-[10px] font-bold ${darkMode ? 'text-gray-400' : 'text-gray-600'} uppercase tracking-wider whitespace-nowrap`}>
                        Cover Letter
                      </span>

                      <div className="flex items-center gap-2">
                        <div className="relative">
                          <select value={coverLetterTone} onChange={e => setCoverLetterTone(e.target.value)} className={`appearance-none ${darkMode ? 'bg-[#22262e] border-gray-700 text-gray-300' : 'bg-white border-gray-300 text-gray-700'} border text-[10px] px-2 py-1 pr-6 rounded focus:outline-none focus:border-blue-500 cursor-pointer`}>
                            <option>Professional</option>
                            <option>Enthusiastic</option>
                            <option>Concise</option>
                          </select>
                          <ChevronDown className={`w-2.5 h-2.5 ${darkMode ? 'text-gray-500' : 'text-gray-400'} absolute right-1.5 top-1/2 -translate-y-1/2 pointer-events-none`} />
                        </div>

                        <button onClick={handleGenerateCoverLetter} disabled={!hasAnalyzed || isGeneratingCoverLetter} className="flex items-center gap-1.5 px-2 py-1 bg-green-600 hover:bg-green-500 text-white text-[10px] font-bold uppercase tracking-wider rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed">
                          {isGeneratingCoverLetter ? <Loader2 className="w-3 h-3 animate-spin" /> : 'Generate'}
                        </button>

                        <button onClick={handleDownloadCoverLetter} disabled={!coverLetterResults} className="flex items-center gap-1.5 px-2 py-1 bg-blue-600 hover:bg-blue-500 text-white text-[10px] font-bold uppercase tracking-wider rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed">
                          <Download className="w-3 h-3" />
                        </button>
                      </div>
                    </div>
                    <div className={`flex-grow ${darkMode ? 'bg-white' : 'bg-gray-50'} p-4 overflow-y-auto custom-scrollbar rounded-b-lg`}>
                      {coverLetterResults ? <pre className="font-sans text-gray-800 text-xs whitespace-pre-wrap leading-relaxed">
                          {coverLetterResults.cover_letter}
                        </pre> : <div className="h-full flex flex-col items-center justify-center text-gray-400">
                          <FileText className="w-10 h-10 mb-2 opacity-20" />
                          <p className="text-xs">Cover letter preview</p>
                        </div>}
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          </section>

          {/* 5. RECOMMENDATIONS */}
          <section className="w-[19.5%] min-w-[350px] h-full flex flex-col gap-6 overflow-y-auto pr-2 custom-scrollbar flex-shrink-0">
            {/* Recommendations Panel */}
            <div className="space-y-4">
              <div className="flex items-center gap-2 mb-1">
                <Target className="w-4 h-4 text-purple-400" />
                <h2 className={`text-xs font-bold tracking-widest ${darkMode ? 'text-gray-500' : 'text-gray-400'} uppercase`}>
                  AI Recommendations
                </h2>
                <div className={`h-px ${darkMode ? 'bg-gray-800' : 'bg-gray-300'} flex-grow ml-4`} />
              </div>

              <div className={`${darkMode ? 'bg-[#22262e] border-gray-800' : 'bg-white border-gray-200'} border rounded-lg p-6 shadow-lg`}>
                {hasAnalyzed ? (
                  <div className="space-y-6">
                    {/* Overall Fit Score */}
                    <div className="text-center">
                      <CircularProgress
                        percentage={Math.round((analysisResults.optimized_score + analysisResults.original_score) / 2)}
                        size={100}
                        strokeWidth={6}
                        label="OVERALL FIT"
                      />
                    </div>

                    {/* LLM Recommendations */}
                    <div className="space-y-4">
                      <div>
                        <h3 className={`text-[10px] font-bold ${darkMode ? 'text-gray-400' : 'text-gray-500'} uppercase tracking-wider mb-3`}>
                          Strategic Analysis
                        </h3>
                        <div className="space-y-3">
                          {analysisResults.recommendations?.strategic_analysis?.map((item, idx) => (
                            <div key={idx} className={`p-3 rounded ${darkMode ? 'bg-[#1a1d23]' : 'bg-gray-50'} border ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                              <div className="flex items-start gap-3">
                                <div className={`w-2 h-2 rounded-full mt-1.5 ${item.priority === 'high' ? 'bg-red-500' : item.priority === 'medium' ? 'bg-yellow-500' : 'bg-blue-500'}`}></div>
                                <div className="flex-1">
                                  <p className="text-sm font-medium mb-1">{item.aspect}</p>
                                  <p className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>{item.analysis}</p>
                                </div>
                              </div>
                            </div>
                          )) || (
                            <div className="text-center py-8">
                              <p className={`text-sm ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>Recommendations pending analysis</p>
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Career Development Focus */}
                      <div>
                        <h3 className={`text-[10px] font-bold ${darkMode ? 'text-gray-400' : 'text-gray-500'} uppercase tracking-wider mb-3`}>
                          Career Development Focus
                        </h3>
                        <div className="space-y-2">
                          {analysisResults.recommendations?.career_focus?.map((focus, idx) => (
                            <div key={idx} className={`flex items-center gap-3 p-2 rounded ${darkMode ? 'bg-[#1a1d23]' : 'bg-gray-50'}`}>
                              <CheckCircle2 className="w-4 h-4 text-green-500 flex-shrink-0" />
                              <span className="text-sm">{focus}</span>
                            </div>
                          )) || (
                            <div className="text-center py-4">
                              <p className={`text-sm ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>Focus areas pending analysis</p>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center py-12">
                    <div className={`w-16 h-16 rounded-full ${darkMode ? 'bg-gray-800' : 'bg-gray-100'} flex items-center justify-center mb-4`}>
                      <Target className={`w-8 h-8 ${darkMode ? 'text-gray-700' : 'text-gray-300'}`} />
                    </div>
                    <p className={`text-sm ${darkMode ? 'text-gray-500' : 'text-gray-400'} font-mono`}>
                      AI recommendations will appear here
                    </p>
                  </div>
                )}
              </div>
            </div>
          </section>

          {/* 6. ACTION PLAN */}
          <section className="w-[28.5%] min-w-[450px] h-full overflow-y-auto pr-2 custom-scrollbar flex-shrink-0">
            <motion.div initial={{
            opacity: 0,
            y: 20
          }} animate={{
            opacity: 1,
            y: 0
          }} transition={{
            duration: 0.5,
            delay: 0.4
          }} className="space-y-4 pb-8">
              {/* ACTION PLAN TABLE */}
              <div className="space-y-4">
                <div className="flex items-center gap-2 mb-1">
                  <AlertCircle className="w-4 h-4 text-amber-400" />
                  <h2 className={`text-xs font-bold tracking-widest ${darkMode ? 'text-gray-500' : 'text-gray-400'} uppercase`}>
                    Action Plan
                  </h2>
                  <div className={`h-px ${darkMode ? 'bg-gray-800' : 'bg-gray-300'} flex-grow ml-4`} />
                </div>

                <div className={`${darkMode ? 'bg-[#22262e] border-gray-800' : 'bg-white border-gray-200'} border rounded-lg shadow-lg h-[calc(100vh-200px)] overflow-auto custom-scrollbar`}>
                  {hasAnalyzed && analysisResults ? (
                    <table className="w-full">
                      <thead className={`${darkMode ? 'bg-[#1a1d23] border-gray-700' : 'bg-gray-50 border-gray-200'} border-b`}>
                        <tr>
                          <th className={`text-left p-3 text-[9px] font-bold ${darkMode ? 'text-gray-400' : 'text-gray-500'} uppercase tracking-wider`}>Skill Gap</th>
                          <th className={`text-left p-3 text-[9px] font-bold ${darkMode ? 'text-gray-400' : 'text-gray-500'} uppercase tracking-wider`}>Priority</th>
                          <th className={`text-left p-3 text-[9px] font-bold ${darkMode ? 'text-gray-400' : 'text-gray-500'} uppercase tracking-wider`}>Resource</th>
                        </tr>
                      </thead>
                      <tbody>
                        {analysisResults.skill_analysis.gap_analysis.map((item, idx) => (
                          <tr key={idx} className={`${darkMode ? 'border-gray-700' : 'border-gray-200'} border-b hover:${darkMode ? 'bg-[#1a1d23]' : 'bg-gray-50'} transition-colors`}>
                            <td className="p-3">
                              <div className="font-mono font-bold text-amber-400 text-[10px] leading-tight">{item.skill}</div>
                            </td>
                            <td className="p-3">
                              <span className={`inline-block text-[8px] font-bold uppercase px-1 py-0.5 rounded ${
                                item.priority.toLowerCase() === 'high' ? 'bg-red-500/20 text-red-400' :
                                item.priority.toLowerCase() === 'medium' ? 'bg-amber-500/20 text-amber-400' :
                                'bg-blue-500/20 text-blue-400'
                              }`}>
                                {item.priority}
                              </span>
                            </td>
                            <td className="p-3">
                              <a
                                href={item.learning_url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-[9px] text-blue-400 hover:text-blue-300 flex items-center gap-1 group"
                              >
                                <span className="truncate max-w-[80px] leading-tight">{item.suggestion.substring(0, 25)}...</span>
                                <ExternalLink className="w-2.5 h-2.5 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0" />
                              </a>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : (
                    <div className="h-full flex flex-col items-center justify-center text-gray-500 p-4">
                      <div className={`w-12 h-12 rounded-full ${darkMode ? 'bg-gray-800' : 'bg-gray-100'} flex items-center justify-center mb-3`}>
                        <AlertCircle className={`w-6 h-6 ${darkMode ? 'text-gray-700' : 'text-gray-300'}`} />
                      </div>
                      <p className="text-[10px] text-center leading-tight">
                        Action items will appear here after analysis
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          </section>
        </div>

        {/* Error Alerts */}
        {error && (
          <div className="fixed bottom-4 right-4 z-50 max-w-md">
            <div className="bg-red-500/10 border border-red-500/20 text-red-400 p-4 rounded-lg shadow-lg backdrop-blur-sm">
              <div className="flex items-start gap-3">
                <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
                <div className="flex-1">
                  <p className="text-sm font-medium">Error</p>
                  <p className="text-sm mt-1">{error}</p>
                </div>
                <button
                  onClick={() => setError(null)}
                  className="text-red-400 hover:text-red-300 transition-colors"
                >
                  ×
                </button>
              </div>
            </div>
          </div>
        )}

        {/* LLM Assistant Button */}
        <div className="fixed bottom-4 right-4 z-50">
          <button
            onClick={() => setShowChat(!showChat)}
            className="bg-green-600 hover:bg-green-500 text-white px-4 py-3 rounded-full shadow-lg transition-all duration-300 hover:scale-105 flex items-center gap-2"
            title="AI Assistant"
          >
            <MessageSquare className="w-5 h-5" />
            <span className="text-sm font-medium">AI Assistant</span>
          </button>
        </div>

        {/* LLM Assistant Chat Window */}
        {showChat && (
          <div className="fixed bottom-20 right-4 bg-gray-900/95 border border-gray-700 text-gray-300 p-4 rounded-lg shadow-xl backdrop-blur-sm max-w-md max-h-96 flex flex-col z-50">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <MessageSquare className="w-4 h-4 text-green-400" />
                <h3 className="text-sm font-bold">AI Assistant</h3>
              </div>
              <button
                onClick={() => setChatMessages([])}
                className="text-gray-500 hover:text-gray-300 text-xs px-2 py-1 rounded hover:bg-gray-800"
                title="Clear chat"
              >
                Clear
              </button>
            </div>

            {/* Chat Messages */}
            <div className="flex-1 overflow-y-auto mb-3 space-y-3 pr-1">
              {chatMessages.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-center py-6">
                  <MessageSquare className={`w-8 h-8 mb-3 ${darkMode ? 'text-gray-600' : 'text-gray-300'}`} />
                  <p className="text-sm text-gray-400 mb-2">Start a conversation</p>
                  <p className="text-xs text-gray-500">Ask questions about your resume, job fit, or career advice</p>
                </div>
              ) : (
                chatMessages.map((msg, idx) => (
                  <div key={idx} className={`p-3 rounded-lg ${
                    msg.role === 'user'
                      ? 'bg-blue-600/20 border-l-4 border-l-blue-500'
                      : 'bg-green-600/20 border-l-4 border-l-green-500'
                  }`}>
                    <div className="flex items-center gap-2 mb-2">
                      <div className={`w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold ${
                        msg.role === 'user' ? 'bg-blue-500 text-white' : 'bg-green-500 text-white'
                      }`}>
                        {msg.role === 'user' ? 'U' : 'AI'}
                      </div>
                      <span className="text-xs font-bold text-gray-300">
                        {msg.role === 'user' ? 'You' : 'AI Assistant'}
                      </span>
                    </div>
                    <div className="text-sm text-gray-200 whitespace-pre-wrap leading-relaxed">
                      {msg.content}
                    </div>
                  </div>
                ))
              )}
            </div>

            {/* Chat Input */}
            <div className="flex gap-2">
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSendChatMessage()}
                placeholder="Ask about your resume, job fit, or career advice..."
                className="flex-1 bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-green-500 focus:ring-1 focus:ring-green-500/20"
                disabled={!connectionStatus.includes('connected')}
              />
              <button
                onClick={handleSendChatMessage}
                disabled={!chatInput.trim() || !connectionStatus.includes('connected')}
                className="px-3 py-2 bg-green-600 hover:bg-green-500 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-medium rounded transition-colors flex items-center gap-1"
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}
      </main>
    </div>;
}
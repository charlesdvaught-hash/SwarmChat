import React, { useState, useEffect, useRef } from 'react';
import {
  MessageSquare, Zap, Settings, Shield, BrainCircuit, AlertTriangle,
  Play, Crown, Cpu, Sparkles, Send, X, Users, Check, Plus, FolderOpen, Trash2,
  ChevronDown, ChevronRight, UserMinus, UserPlus, RefreshCw, FileText, CheckSquare, Activity
} from 'lucide-react';

interface ChatMessage {
  id: string;
  sender: string;
  role: string;
  content: string;
  is_admin: boolean;
  model_id?: string;
  phase: string;
  timestamp: number;
}

interface ModelConfig {
  id: string;
  name: string;
  role: string;
  provider: string;
  model_name: string;
  gguf_path?: string;
  api_key?: string;
  enabled: boolean;
  is_moderator: boolean;
  status: string;
  live_status?: string;
}

interface PendingVote {
  id: string;
  model_id: string;
  model_name: string;
  tool_name: string;
  args: any;
  risk_level: string;
  status: string;
}

export default function App() {
  const [phase, setPhase] = useState<'discussion' | 'execution'>('discussion');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputText, setInputText] = useState('');
  const [models, setModels] = useState<Record<string, ModelConfig>>({});
  const [knownModels, setKnownModels] = useState<Record<string, ModelConfig>>({});
  const [modelStatuses, setModelStatuses] = useState<Record<string, any>>({});
  const [pendingVotes, setPendingVotes] = useState<PendingVote[]>([]);
  const [sharedMemory, setSharedMemory] = useState<any[]>([]);
  const [hardware, setHardware] = useState<any>(null);
  const [dependencies, setDependencies] = useState<any>(null);
  const [isInstallingEngine, setIsInstallingEngine] = useState(false);
  const [selectedErrorModel, setSelectedErrorModel] = useState<string | null>(null);
  
  // UI Panels & Modals
  const [showSetup, setShowSetup] = useState(false);
  const [showMemory, setShowMemory] = useState(false);
  const [showModPrompt, setShowModPrompt] = useState(false);
  const [showOverlay, setShowOverlay] = useState(false);
  const [overlayTab, setOverlayTab] = useState<'itinerary' | 'episodes' | 'workspace' | 'health'>('itinerary');

  const [episodes, setEpisodes] = useState<any[]>([]);
  const [taskItinerary, setTaskItinerary] = useState<any[]>([]);
  const [activeTask, setActiveTask] = useState<any>(null);
  const [fileAuditLog, setFileAuditLog] = useState<any[]>([]);
  const [activeFileLocks, setActiveFileLocks] = useState<Record<string, any>>({});
  const [roomHealth, setRoomHealth] = useState<any[]>([]);

  // Workspace File Explorer State
  const [workspaceFiles, setWorkspaceFiles] = useState<any[]>([]);
  const [selectedFilePath, setSelectedFilePath] = useState<string>('');
  const [selectedFileContent, setSelectedFileContent] = useState<string>('');

  // New Itinerary Task Form
  const [newTaskTitle, setNewTaskTitle] = useState('');
  const [newTaskDesc, setNewTaskDesc] = useState('');
  const [newTaskPriority, setNewTaskPriority] = useState('medium');

  const [formerModId, setFormerModId] = useState<string | null>(null);
  const [showSidebar, setShowSidebar] = useState(true);
  const [isStepping, setIsStepping] = useState(false);
  const [collapsedCodeBlocks, setCollapsedCodeBlocks] = useState<Record<string, boolean>>({});

  // New Model Form State in Settings
  const [newModelName, setNewModelName] = useState('');
  const [newModelRole, setNewModelRole] = useState('Architect');
  const [newModelProvider, setNewModelProvider] = useState('gguf_local');
  const [newModelNameOrTag, setNewModelNameOrTag] = useState('');
  const [newGgufPath, setNewGgufPath] = useState('');
  const [newMmprojPath, setNewMmprojPath] = useState('');
  const [newApiKey, setNewApiKey] = useState('');
  const [newIsModerator, setNewIsModerator] = useState(false);

  // Path Validation & FS Browser State
  const [pathValidationMsg, setPathValidationMsg] = useState<{ valid: boolean; message: string; size_gb?: number } | null>(null);
  const [showFsBrowser, setShowFsBrowser] = useState(false);
  const [fsTargetField, setFsTargetField] = useState<'gguf' | 'mmproj' | 'search_path'>('gguf');
  const [fsCurrentPath, setFsCurrentPath] = useState('');
  const [fsParentPath, setFsParentPath] = useState<string | null>(null);
  const [fsDirs, setFsDirs] = useState<any[]>([]);
  const [fsFiles, setFsFiles] = useState<any[]>([]);

  // Search Paths State
  const [searchPaths, setSearchPaths] = useState<string[]>([]);
  const [customSearchPathInput, setCustomSearchPathInput] = useState('');

  // HuggingFace Model Search / Hiring State
  const [hfQuery, setHfQuery] = useState('');
  const [hfResults, setHfResults] = useState<any[]>([]);
  const [isSearchingHf, setIsSearchingHf] = useState(false);
  const [settingsTab, setSettingsTab] = useState<'models' | 'browse_hf' | 'search_paths'>('models');

  const chatEndRef = useRef<HTMLDivElement>(null);

  const fetchState = async () => {
    try {
      const res = await fetch('/api/state');
      if (res.ok) {
        const data = await res.json();
        setPhase(data.phase);
        setModels(data.models || {});
        setKnownModels(data.known_models || data.models || {});
        setModelStatuses(data.model_statuses || {});
        setPendingVotes(data.pending_votes || []);
        setMessages(data.chat_history || []);
        setSharedMemory(data.shared_memory || []);
        setEpisodes(data.episodes || []);
        setTaskItinerary(data.task_itinerary || []);
        setActiveTask(data.active_task || null);
        setFileAuditLog(data.file_audit_log || []);
        setActiveFileLocks(data.active_file_locks || {});
      }
      const hwRes = await fetch('/api/hardware');
      if (hwRes.ok) {
        setHardware(await hwRes.json());
      }
      const depRes = await fetch('/api/dependencies');
      if (depRes.ok) {
        setDependencies(await depRes.json());
      }

      const filesRes = await fetch('/api/workspace/files');
      if (filesRes.ok) {
        const fdata = await filesRes.json();
        setWorkspaceFiles(fdata.items || []);
      }

      const healthRes = await fetch('/api/evaluate/health');
      if (healthRes.ok) {
        const hdata = await healthRes.json();
        setRoomHealth(hdata.reports || []);
      }

      const pathsRes = await fetch('/api/models/search_paths');
      if (pathsRes.ok) {
        const pdata = await pathsRes.json();
        setSearchPaths(pdata.search_paths || []);
      }
    } catch (e) {
      console.error('Fetch state error:', e);
    }
  };

  const handleBrowseFs = async (targetPath?: string) => {
    try {
      const url = targetPath ? `/api/fs/browse?path=${encodeURIComponent(targetPath)}` : '/api/fs/browse';
      const res = await fetch(url);
      if (res.ok) {
        const data = await res.json();
        setFsCurrentPath(data.current_path);
        setFsParentPath(data.parent_path);
        setFsDirs(data.directories || []);
        setFsFiles(data.files || []);
      }
    } catch (e) {
      console.error('Browse filesystem error:', e);
    }
  };

  const handleValidatePath = async (p: string, mmP?: string) => {
    if (!p) return;
    try {
      const res = await fetch('/api/fs/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: p, mmproj_path: mmP })
      });
      if (res.ok) {
        const data = await res.json();
        setPathValidationMsg({ valid: data.valid, message: data.message, size_gb: data.file_size_gb });
      }
    } catch (e) {
      console.error('Validate path error:', e);
    }
  };

  const handleSearchHf = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!hfQuery.trim()) return;
    setIsSearchingHf(true);
    try {
      const res = await fetch(`/api/tools/search_hf?query=${encodeURIComponent(hfQuery)}`);
      if (res.ok) {
        const data = await res.json();
        setHfResults(data.models || []);
      }
    } catch (e) {
      console.error('HF Search error:', e);
    } finally {
      setIsSearchingHf(false);
    }
  };

  const handleInstallEngine = async () => {
    setIsInstallingEngine(true);
    try {
      await fetch('/api/engine/install', { method: 'POST' });
      await fetchState();
    } finally {
      setIsInstallingEngine(false);
    }
  };

  useEffect(() => {
    fetchState();
    const interval = setInterval(fetchState, 3000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const togglePhase = async () => {
    const nextPhase = phase === 'discussion' ? 'execution' : 'discussion';
    await fetch('/api/phase', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ phase: nextPhase })
    });
    fetchState();
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim()) return;
    const text = inputText;
    setInputText('');

    await fetch('/api/chat/message', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sender: 'Admin', content: text, is_admin: true })
    });

    fetchState();
  };

  const handleStepTurn = async () => {
    setIsStepping(true);
    try {
      await fetch('/api/chat/step', { method: 'POST' });
      await fetchState();
    } finally {
      setIsStepping(false);
    }
  };

  const handleVoteOverride = async (voteId: string, action: 'approve' | 'reject') => {
    await fetch('/api/votes/override', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ vote_id: voteId, action })
    });
    fetchState();
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const filePath = (file as any).path || file.name;
      setNewGgufPath(filePath);
      if (!newModelName) {
        const cleanName = file.name.replace(/\.gguf$/i, '').replace(/[-_]/g, ' ');
        setNewModelName(cleanName);
      }
    }
  };

  const handleKickModel = async (modelId: string) => {
    const isMod = models[modelId]?.is_moderator;
    const res = await fetch(`/api/models/kick?model_id=${encodeURIComponent(modelId)}`, { method: 'POST' });
    if (res.ok) {
      const data = await res.json();
      if (isMod) {
        setFormerModId(modelId);
        setShowModPrompt(true);
      }
      fetchState();
    }
  };

  const handleReaddModel = async (modelId: string) => {
    await fetch(`/api/models/readd?model_id=${encodeURIComponent(modelId)}`, { method: 'POST' });
    fetchState();
  };

  const handleSelectModerator = async (modelId: string) => {
    await fetch(`/api/models/set_moderator?model_id=${encodeURIComponent(modelId)}`, { method: 'POST' });
    setShowModPrompt(false);
    setFormerModId(null);
    fetchState();
  };

  const handleCloseModPrompt = () => {
    setShowModPrompt(false);
    setFormerModId(null);
  };

  const handleAddModelSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const mId = `model_${Date.now()}`;
    const payload = {
      id: mId,
      name: newModelName || 'New Model',
      role: newModelRole,
      provider: newModelProvider,
      model_name: newModelNameOrTag || (newModelProvider === 'gguf_local' ? newGgufPath : 'llama3.2:1b'),
      gguf_path: newGgufPath,
      mmproj_path: newMmprojPath,
      api_key: newApiKey,
      enabled: true,
      is_moderator: newIsModerator,
      status: 'active'
    };

    await fetch('/api/models/configure', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    setNewModelName('');
    setNewGgufPath('');
    setNewMmprojPath('');
    setNewApiKey('');
    setPathValidationMsg(null);
    fetchState();
  };

  const toggleCodeCollapse = (blockKey: string) => {
    setCollapsedCodeBlocks(prev => ({ ...prev, [blockKey]: !prev[blockKey] }));
  };

  const renderCleanMessage = (content: string, msgId: string) => {
    let cleanText = content
      .replace(/\[READY_FOR_EXECUTION\]/g, '')
      .replace(/\[REQUEST_DISCUSSION\]/g, '')
      .replace(/\[REQUEST_NAP\]/g, '')
      .replace(/\[LOG_TO_MEMORY:[^\]]*\]/g, '')
      .replace(/\[JOURNAL:[^\]]*\]/g, '')
      .trim();

    const parts = cleanText.split(/(```[\s\S]*?```)/g);
    return parts.map((part, idx) => {
      if (part.startsWith('```') && part.endsWith('```')) {
        const blockKey = `${msgId}_code_${idx}`;
        const isCollapsed = collapsedCodeBlocks[blockKey] ?? true;
        const lines = part.slice(3, -3).trim().split('\n');
        const lang = lines[0].match(/^[a-zA-Z0-9_-]+$/) ? lines[0] : '';
        const codeContent = lang ? lines.slice(1).join('\n') : lines.join('\n');

        return (
          <div key={idx} className="my-2 border border-slate-700/80 rounded-xl overflow-hidden bg-slate-950 font-mono text-xs">
            <button
              type="button"
              onClick={() => toggleCodeCollapse(blockKey)}
              className="w-full px-3 py-2 bg-slate-900 hover:bg-slate-850 border-b border-slate-800 flex items-center justify-between text-slate-300 transition"
            >
              <div className="flex items-center gap-2 font-semibold">
                {isCollapsed ? <ChevronRight className="w-3.5 h-3.5 text-emerald-400" /> : <ChevronDown className="w-3.5 h-3.5 text-emerald-400" />}
                <span>Technical Implementation / Code ({lang || 'code'})</span>
              </div>
              <span className="text-[10px] text-slate-500 underline">{isCollapsed ? 'Expand to view' : 'Collapse'}</span>
            </button>
            {!isCollapsed && (
              <pre className="p-3 text-slate-200 overflow-x-auto whitespace-pre leading-relaxed text-[11px] bg-slate-950">
                <code>{codeContent}</code>
              </pre>
            )}
          </div>
        );
      }
      return <span key={idx}>{part}</span>;
    });
  };

  return (
    <div className="flex flex-col h-screen bg-slate-950 text-slate-100 font-sans">
      {/* --- TOP BANNER: Phase Indicator & Quick Controls --- */}
      <header className="flex items-center justify-between px-5 py-2.5 bg-slate-900/90 border-b border-slate-800 backdrop-blur shrink-0">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 font-bold text-lg text-emerald-400">
            <Sparkles className="w-5 h-5 text-emerald-400" />
            <span>SwarmChat</span>
          </div>
          <div className="h-4 w-px bg-slate-700" />
          
          {/* Phase Badge & Switch Button */}
          <button
            onClick={togglePhase}
            className={`flex items-center gap-2 px-3 py-1 rounded-full font-medium text-xs tracking-wide transition-all shadow-sm ${
              phase === 'discussion'
                ? 'bg-amber-500/10 text-amber-400 border border-amber-500/30 hover:bg-amber-500/20'
                : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 hover:bg-emerald-500/20'
            }`}
          >
            {phase === 'discussion' ? (
              <>
                <MessageSquare className="w-3.5 h-3.5" />
                <span>💬 DISCUSSION PHASE</span>
              </>
            ) : (
              <>
                <Zap className="w-3.5 h-3.5" />
                <span>⚡ EXECUTION PHASE</span>
              </>
            )}
            <span className="text-[10px] opacity-70 underline ml-1">Switch</span>
          </button>

          <div className="h-4 w-px bg-slate-700" />

          {/* SINGLE-LINE ACTIVE TASK / EPISODE HEADER BANNER (Futuristic & Non-Cluttered) */}
          <button
            onClick={() => setShowOverlay(true)}
            className="flex items-center gap-2 bg-slate-950/80 hover:bg-slate-800/80 border border-slate-800 hover:border-cyan-500/50 text-slate-200 px-3 py-1 rounded-xl text-xs transition group"
            title="Click to view Task Itinerary, Episodes & Workspace Files"
          >
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
            <span className="font-semibold text-cyan-300">Active Task:</span>
            <span className="text-slate-200 font-medium truncate max-w-xs">
              {activeTask ? activeTask.title : 'No Active Task'}
            </span>
            <span className="text-slate-500">•</span>
            <span className="text-slate-400">
              Episodes: <strong className="text-purple-400">{episodes.length}</strong>
            </span>
            <span className="text-[10px] bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 px-1.5 py-0.5 rounded ml-1 group-hover:bg-cyan-500/20 transition">
              Open Menu
            </span>
          </button>
        </div>

        {/* Hardware Status & Top Action Buttons */}
        <div className="flex items-center gap-3">
          {hardware && (
            <div className="flex items-center gap-3 text-xs bg-slate-800/60 px-3 py-1 rounded-lg border border-slate-700/50">
              <div className="flex items-center gap-1 text-slate-300">
                <Cpu className="w-3.5 h-3.5 text-slate-400" />
                <span>RAM: {hardware.ram_available_gb}GB</span>
              </div>
            </div>
          )}

          <button
            onClick={() => setShowSetup(true)}
            className="flex items-center gap-1.5 text-xs bg-emerald-600 hover:bg-emerald-500 text-white font-medium px-3 py-1 rounded-lg transition shadow-sm"
          >
            <Settings className="w-3.5 h-3.5" />
            <span>Settings</span>
          </button>

          <button
            onClick={() => setShowMemory(!showMemory)}
            className="flex items-center gap-1.5 text-xs bg-slate-800 hover:bg-slate-700 text-slate-200 px-3 py-1 rounded-lg border border-slate-700 transition"
          >
            <BrainCircuit className="w-3.5 h-3.5 text-purple-400" />
            <span>Shared Memory</span>
          </button>
        </div>
      </header>

      {/* --- MAIN CENTER-STAGE CONTENT AREA --- */}
      <div className="flex flex-1 overflow-hidden">
        {/* CENTER CHAT ARENA (Takes Center Stage) */}
        <div className="flex-1 flex flex-col bg-slate-950 relative">
          
          {/* Pending Tool Vote Banner */}
          {pendingVotes.filter(v => v.status === 'pending').length > 0 && (
            <div className="bg-amber-500/10 border-b border-amber-500/30 px-5 py-2.5 flex items-center justify-between text-xs text-amber-300">
              <div className="flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-400" />
                <span>
                  <strong>Pending Tool Call Vote:</strong> {pendingVotes[0].model_name} requested <code>{pendingVotes[0].tool_name}</code> ({pendingVotes[0].risk_level} risk).
                </span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => handleVoteOverride(pendingVotes[0].id, 'approve')}
                  className="bg-emerald-600 hover:bg-emerald-500 text-white px-2.5 py-1 rounded font-medium flex items-center gap-1"
                >
                  <Check className="w-3 h-3" /> Approve (Admin Override)
                </button>
                <button
                  onClick={() => handleVoteOverride(pendingVotes[0].id, 'reject')}
                  className="bg-rose-600 hover:bg-rose-500 text-white px-2.5 py-1 rounded font-medium flex items-center gap-1"
                >
                  <X className="w-3 h-3" /> Reject
                </button>
              </div>
            </div>
          )}

          {/* CHAT MESSAGES STREAM */}
          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            {messages.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-center text-slate-500 space-y-3">
                <MessageSquare className="w-12 h-12 stroke-1 text-slate-600" />
                <div className="max-w-md">
                  <h3 className="text-slate-300 font-semibold text-base mb-1">SwarmChat Arena</h3>
                  <p className="text-xs text-slate-400">
                    Models collaborate here. Configure local `.gguf` files or cloud models in Settings (⚙️).
                  </p>
                </div>
              </div>
            ) : (
              messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex flex-col max-w-3xl ${
                    msg.is_admin ? 'ml-auto items-end' : 'mr-auto items-start'
                  }`}
                >
                  {/* Sender Header */}
                  <div className="flex items-center gap-2 mb-1 text-xs">
                    <span className={`font-semibold ${msg.is_admin ? 'text-emerald-400' : 'text-cyan-300'}`}>
                      {msg.sender}
                    </span>
                    <span className="text-slate-500 text-[10px] px-1.5 py-0.5 rounded bg-slate-800">
                      {msg.role}
                    </span>
                    <span className="text-[10px] text-slate-600">
                      {new Date(msg.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                  </div>

                  {/* Message Bubble */}
                  <div
                    className={`px-4 py-3 rounded-2xl text-sm leading-relaxed border ${
                      msg.is_admin
                        ? 'bg-emerald-950/40 border-emerald-800/50 text-emerald-100 rounded-tr-none'
                        : 'bg-slate-900 border-slate-800 text-slate-200 rounded-tl-none shadow-sm'
                    }`}
                  >
                    <div>{renderCleanMessage(msg.content, msg.id)}</div>
                  </div>
                </div>
              ))
            )}
            <div ref={chatEndRef} />
          </div>

          {/* INPUT BAR & CONTROLS */}
          <div className="p-4 bg-slate-900/80 border-t border-slate-800 backdrop-blur">
            <form onSubmit={handleSendMessage} className="flex gap-2">
              <input
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Message the room or instruct models (Admin)..."
                className="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-emerald-500/50 text-slate-100 placeholder-slate-500 transition"
              />
              <button
                type="submit"
                className="bg-emerald-600 hover:bg-emerald-500 text-white px-4 py-2.5 rounded-xl font-medium text-sm flex items-center gap-1.5 transition shadow-sm"
              >
                <Send className="w-4 h-4" />
                <span>Send</span>
              </button>

              <button
                type="button"
                onClick={handleStepTurn}
                disabled={isStepping}
                className="bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white px-4 py-2.5 rounded-xl font-medium text-sm flex items-center gap-1.5 transition shadow-sm"
                title="Trigger Next Model Turn"
              >
                <Play className="w-4 h-4" />
                <span>{isStepping ? 'Model Thinking...' : 'Trigger Model Turn'}</span>
              </button>
            </form>
          </div>
        </div>

        {/* RIGHT SIDEBAR: Room Participants */}
        {showSidebar && (
          <aside className="w-80 bg-slate-900 border-l border-slate-800 flex flex-col shrink-0">
            <div className="p-4 border-b border-slate-800 flex items-center justify-between">
              <div className="flex items-center gap-2 font-semibold text-sm text-slate-200">
                <Users className="w-4 h-4 text-cyan-400" />
                <span>Active Participants ({Object.keys(models).length})</span>
              </div>
              <button
                onClick={() => setShowSetup(true)}
                className="text-xs bg-slate-800 hover:bg-slate-700 text-emerald-400 px-2 py-1 rounded border border-slate-700 font-medium"
              >
                + Add Model
              </button>
            </div>

            <div className="p-4 space-y-3 overflow-y-auto flex-1">
              {Object.values(models).map((m) => {
                const st = modelStatuses[m.id] || { status: 'online', tok_per_sec: 0, error: null };
                const isError = st.status === 'error' || !!st.error;
                const tokPerSec = st.tok_per_sec || 0;

                return (
                  <div
                    key={m.id}
                    className="bg-slate-950 border border-slate-800/80 rounded-xl p-3 flex flex-col gap-2 relative group"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        {/* Status Icon Indicator */}
                        {isError ? (
                          <button
                            type="button"
                            onClick={() => setSelectedErrorModel(m.id)}
                            className="text-rose-400 hover:text-rose-300 transition"
                            title={`Error: ${st.error || 'Click for details'}`}
                          >
                            <AlertTriangle className="w-4 h-4 fill-rose-500/20 stroke-rose-400" />
                          </button>
                        ) : (
                          <div
                            className="w-2.5 h-2.5 rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.6)]"
                            title="Online & Ready"
                          />
                        )}

                        <span className="font-semibold text-sm text-slate-100">{m.name}</span>

                        {m.is_moderator && (
                          <span className="flex items-center gap-1 bg-amber-500/10 text-amber-400 border border-amber-500/30 text-[10px] px-1.5 py-0.5 rounded-full">
                            <Crown className="w-3 h-3" /> Moderator
                          </span>
                        )}
                      </div>

                      <div className="flex items-center gap-1.5">
                        <span className="text-[10px] text-slate-400 bg-slate-800 px-2 py-0.5 rounded uppercase font-mono">
                          {m.provider === 'gguf_local' ? 'GGUF' : m.provider}
                        </span>
                        <button
                          onClick={() => handleKickModel(m.id)}
                          title="Remove model from chatroom"
                          className="text-slate-500 hover:text-rose-400 p-1 rounded hover:bg-slate-800 transition"
                        >
                          <UserMinus className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>

                    <div className="text-xs text-slate-400 flex items-center justify-between">
                      <div>
                        <span>Role: </span>
                        <span className="text-slate-200 font-medium">{m.role}</span>
                      </div>
                      {tokPerSec > 0 && (
                        <div className="text-[10px] text-cyan-400 font-mono bg-cyan-950/60 border border-cyan-800/40 px-1.5 py-0.5 rounded">
                          {tokPerSec} tok/s
                        </div>
                      )}
                    </div>

                    {/* Live Truth-Backed Status Indicator (Discord Style) */}
                    <div className="text-[11px] text-emerald-400 font-medium flex items-center gap-1 bg-emerald-950/40 border border-emerald-800/40 px-2 py-0.5 rounded-md">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse shrink-0" />
                      <span className="truncate">{m.live_status || 'Idle / Live in Chat'}</span>
                    </div>

                    <div className="text-[11px] text-slate-500 truncate" title={m.gguf_path || m.model_name}>
                      {m.provider === 'gguf_local' ? (
                        <code>File: {m.gguf_path || m.model_name}</code>
                      ) : (
                        <code>Model: {m.model_name}</code>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </aside>
        )}
      </div>

      {/* --- MODEL TROUBLESHOOTING / ERROR DETAIL MODAL --- */}
      {selectedErrorModel && (
        <div className="fixed inset-0 bg-slate-950/85 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-lg w-full p-6 shadow-xl space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="font-bold text-base text-rose-400 flex items-center gap-2">
                <AlertTriangle className="w-5 h-5" />
                <span>Model Diagnostics & Error Log</span>
              </h3>
              <button onClick={() => setSelectedErrorModel(null)} className="text-slate-400 hover:text-slate-200">
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-3 text-xs">
              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 space-y-1">
                <div className="font-semibold text-slate-200">{models[selectedErrorModel]?.name || selectedErrorModel}</div>
                <div className="text-rose-300 font-mono text-[11px] leading-relaxed break-words">
                  {modelStatuses[selectedErrorModel]?.error || 'Unknown status error or engine not loaded.'}
                </div>
              </div>

              {(!dependencies?.llama_cpp_installed || modelStatuses[selectedErrorModel]?.error?.includes('llama-cpp-python')) && (
                <div className="bg-amber-500/10 border border-amber-500/30 p-3 rounded-xl space-y-2">
                  <div className="font-semibold text-amber-300 flex items-center gap-1.5">
                    <Sparkles className="w-4 h-4 text-amber-400" />
                    <span>1-Click Engine Troubleshooter</span>
                  </div>
                  <p className="text-slate-300 text-[11px] leading-relaxed">
                    Local GGUF models require <code>llama-cpp-python</code> engine. Click below to automatically install it.
                  </p>
                  <button
                    onClick={handleInstallEngine}
                    disabled={isInstallingEngine}
                    className="w-full bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white py-2 rounded-lg font-medium transition shadow"
                  >
                    {isInstallingEngine ? 'Installing Engine (pip)...' : '⚡ 1-Click Install GGUF Engine'}
                  </button>
                </div>
              )}
            </div>

            <div className="pt-2 border-t border-slate-800 flex justify-end">
              <button
                onClick={() => setSelectedErrorModel(null)}
                className="bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs px-4 py-2 rounded-xl"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* --- SERVER FILESYSTEM BROWSER MODAL --- */}
      {showFsBrowser && (
        <div className="fixed inset-0 bg-slate-950/90 backdrop-blur-md flex items-center justify-center p-4 z-[60]">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-2xl w-full p-6 shadow-2xl flex flex-col h-[75vh]">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-3">
              <div className="flex items-center gap-2 font-bold text-slate-100 text-base">
                <FolderOpen className="w-5 h-5 text-amber-400" />
                <span>Server Filesystem Explorer</span>
              </div>
              <button onClick={() => setShowFsBrowser(false)} className="text-slate-400 hover:text-slate-200">
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800 font-mono text-xs text-slate-300 mb-3 flex items-center justify-between">
              <span className="truncate">Path: {fsCurrentPath}</span>
              {fsParentPath && (
                <button
                  onClick={() => handleBrowseFs(fsParentPath)}
                  className="bg-slate-800 hover:bg-slate-700 text-cyan-300 px-2.5 py-1 rounded text-[11px] font-sans font-medium"
                >
                  ⬆️ Parent Dir
                </button>
              )}
            </div>

            <div className="flex-1 overflow-y-auto space-y-1.5 pr-2 text-xs">
              {/* Directories */}
              {fsDirs.map((d) => (
                <button
                  key={d.path}
                  onClick={() => handleBrowseFs(d.path)}
                  className="w-full text-left bg-slate-950 hover:bg-slate-800/80 border border-slate-800/80 p-2.5 rounded-xl flex items-center gap-2 text-slate-200 transition group"
                >
                  <FolderOpen className="w-4 h-4 text-amber-400 shrink-0" />
                  <span className="font-semibold text-slate-200 group-hover:text-amber-300">{d.name}</span>
                </button>
              ))}

              {/* Files */}
              {fsFiles.map((f) => (
                <div
                  key={f.path}
                  className={`p-2.5 rounded-xl border flex items-center justify-between transition ${
                    f.is_gguf
                      ? 'bg-emerald-950/20 border-emerald-800/50 text-emerald-200'
                      : f.is_mmproj
                      ? 'bg-purple-950/20 border-purple-800/50 text-purple-200'
                      : 'bg-slate-950 border-slate-800/60 text-slate-400'
                  }`}
                >
                  <div className="flex items-center gap-2 truncate">
                    <FileText className={`w-4 h-4 shrink-0 ${f.is_gguf ? 'text-emerald-400' : f.is_mmproj ? 'text-purple-400' : 'text-slate-500'}`} />
                    <span className="font-mono text-[11px] truncate">{f.name}</span>
                    <span className="text-[10px] text-slate-500">({f.size_mb} MB)</span>
                  </div>

                  <button
                    onClick={() => {
                      if (fsTargetField === 'gguf') {
                        setNewGgufPath(f.path);
                        if (!newModelName) {
                          const cleanName = f.name.replace(/\.gguf$/i, '').replace(/[-_]/g, ' ');
                          setNewModelName(cleanName);
                        }
                        handleValidatePath(f.path, newMmprojPath);
                      } else if (fsTargetField === 'mmproj') {
                        setNewMmprojPath(f.path);
                        handleValidatePath(newGgufPath, f.path);
                      } else if (fsTargetField === 'search_path') {
                        setCustomSearchPathInput(f.path);
                      }
                      setShowFsBrowser(false);
                    }}
                    className="bg-emerald-600 hover:bg-emerald-500 text-white px-3 py-1 rounded-lg text-xs font-medium shrink-0 shadow transition"
                  >
                    Select Path
                  </button>
                </div>
              ))}
            </div>

            <div className="pt-3 border-t border-slate-800 flex justify-end">
              <button onClick={() => setShowFsBrowser(false)} className="bg-slate-800 hover:bg-slate-700 text-slate-200 px-4 py-1.5 rounded-xl text-xs">
                Close Explorer
              </button>
            </div>
          </div>
        </div>
      )}

      {/* --- ALL-IN-ONE SETTINGS & MODEL PICKER MODAL --- */}
      {showSetup && (
        <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-2xl w-full p-6 shadow-xl space-y-5 max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="font-bold text-lg text-slate-100 flex items-center gap-2">
                <Settings className="w-5 h-5 text-emerald-400" />
                <span>Settings & Model Manager</span>
              </h3>
              <button onClick={() => setShowSetup(false)} className="text-slate-400 hover:text-slate-200">
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* SETTINGS TABS SELECTOR */}
            <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
              <button
                onClick={() => setSettingsTab('models')}
                className={`px-3.5 py-1.5 rounded-xl font-medium text-xs transition ${
                  settingsTab === 'models' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30' : 'bg-slate-950 text-slate-400 hover:bg-slate-800'
                }`}
              >
                ⚙️ Add & Manage Models
              </button>
              <button
                onClick={() => setSettingsTab('browse_hf')}
                className={`px-3.5 py-1.5 rounded-xl font-medium text-xs transition ${
                  settingsTab === 'browse_hf' ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/30' : 'bg-slate-950 text-slate-400 hover:bg-slate-800'
                }`}
              >
                🤗 Brainstorm & Hire from HuggingFace
              </button>
              <button
                onClick={() => setSettingsTab('search_paths')}
                className={`px-3.5 py-1.5 rounded-xl font-medium text-xs transition ${
                  settingsTab === 'search_paths' ? 'bg-purple-500/10 text-purple-400 border border-purple-500/30' : 'bg-slate-950 text-slate-400 hover:bg-slate-800'
                }`}
              >
                📁 GGUF Search Paths ({searchPaths.length})
              </button>
            </div>

            {/* TAB 1: ADD & MANAGE MODELS */}
            {settingsTab === 'models' && (
              <div className="space-y-4">
                {/* Add New Model Form */}
                <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-3">
                  <h4 className="font-semibold text-sm text-emerald-400 flex items-center gap-2">
                    <Plus className="w-4 h-4" />
                    <span>Add Model / Local GGUF Participant</span>
                  </h4>

                  <form onSubmit={handleAddModelSubmit} className="space-y-3 text-xs">
                    <div>
                      <label className="block text-slate-300 font-medium mb-1">Model Provider Type</label>
                      <select
                        value={newModelProvider}
                        onChange={(e) => setNewModelProvider(e.target.value)}
                        className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-slate-100 focus:outline-none focus:border-emerald-500"
                      >
                        <option value="gguf_local">📁 Local GGUF File (.gguf on hard drive)</option>
                        <option value="ollama">🦙 Ollama Local Model</option>
                        <option value="claude">☁️ Claude API (Anthropic)</option>
                        <option value="groq">⚡ Groq API</option>
                        <option value="gemini">♊ Gemini API (Google)</option>
                      </select>
                    </div>

                    {/* Local GGUF File Picker */}
                    {newModelProvider === 'gguf_local' && (
                      <div className="space-y-3 bg-slate-900 border border-slate-800 p-3 rounded-xl">
                        <label className="block text-slate-300 font-medium">Select GGUF File from Hard Drive</label>
                        <div className="flex gap-2 items-center">
                          <button
                            type="button"
                            onClick={() => {
                              setFsTargetField('gguf');
                              handleBrowseFs();
                              setShowFsBrowser(true);
                            }}
                            className="bg-emerald-950 hover:bg-emerald-900 text-emerald-300 border border-emerald-800/80 px-3 py-2 rounded-lg font-medium flex items-center gap-1.5 shrink-0"
                          >
                            <FolderOpen className="w-4 h-4 text-emerald-400" />
                            <span>Browse Server Drives...</span>
                          </button>

                          <input
                            type="text"
                            value={newGgufPath}
                            onChange={(e) => {
                              setNewGgufPath(e.target.value);
                              handleValidatePath(e.target.value, newMmprojPath);
                            }}
                            placeholder="Type file name or path e.g. Bonsai-27B-Q1_0.gguf"
                            className="flex-1 bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-100 placeholder-slate-500 font-mono text-[11px]"
                          />

                          <button
                            type="button"
                            onClick={() => handleValidatePath(newGgufPath, newMmprojPath)}
                            className="bg-slate-800 hover:bg-slate-700 text-slate-200 px-3 py-2 rounded-lg font-medium shrink-0 border border-slate-700"
                          >
                            Validate
                          </button>
                        </div>

                        {/* Optional mmproj (Multimodal Projector) File Picker */}
                        <div className="pt-2 border-t border-slate-800/80 space-y-1">
                          <label className="block text-slate-400 text-[11px]">
                            Optional Multimodal Projector / Vision File (<code>mmproj</code>)
                          </label>
                          <div className="flex gap-2 items-center">
                            <button
                              type="button"
                              onClick={() => {
                                setFsTargetField('mmproj');
                                handleBrowseFs();
                                setShowFsBrowser(true);
                              }}
                              className="bg-purple-950 hover:bg-purple-900 text-purple-300 border border-purple-800/80 px-2.5 py-1.5 rounded-lg text-[11px] font-medium flex items-center gap-1 shrink-0"
                            >
                              <FolderOpen className="w-3.5 h-3.5 text-purple-400" />
                              <span>Browse mmproj...</span>
                            </button>
                            <input
                              type="text"
                              value={newMmprojPath}
                              onChange={(e) => {
                                setNewMmprojPath(e.target.value);
                                handleValidatePath(newGgufPath, e.target.value);
                              }}
                              placeholder="e.g. mmproj-model-f16.gguf"
                              className="flex-1 bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-slate-100 font-mono text-[11px]"
                            />
                          </div>
                        </div>

                        {/* Path Validation Status Message Badge */}
                        {pathValidationMsg && (
                          <div className={`p-2.5 rounded-lg text-[11px] font-mono leading-relaxed ${
                            pathValidationMsg.valid ? 'bg-emerald-950/60 border border-emerald-800 text-emerald-300' : 'bg-rose-950/60 border border-rose-800 text-rose-300'
                          }`}>
                            {pathValidationMsg.message}
                          </div>
                        )}
                      </div>
                    )}

                    {newModelProvider !== 'gguf_local' && (
                      <div>
                        <label className="block text-slate-300 font-medium mb-1">Model Tag / API Key</label>
                        <input
                          type="text"
                          value={newModelNameOrTag}
                          onChange={(e) => setNewModelNameOrTag(e.target.value)}
                          placeholder="e.g. llama3.2:1b, claude-3-5-sonnet-20241022, groq-llama3"
                          className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-slate-100 font-mono"
                        />
                      </div>
                    )}

                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="block text-slate-300 font-medium mb-1">Display Name</label>
                        <input
                          type="text"
                          value={newModelName}
                          onChange={(e) => setNewModelName(e.target.value)}
                          placeholder="e.g. Bonsai Solver, Llama Architect"
                          className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-slate-100"
                          required
                        />
                      </div>

                      <div>
                        <label className="block text-slate-300 font-medium mb-1">Assigned Role</label>
                        <select
                          value={newModelRole}
                          onChange={(e) => setNewModelRole(e.target.value)}
                          className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-slate-100"
                        >
                          <option value="Architect">🏗️ Architect</option>
                          <option value="Critic">🧐 Critic</option>
                          <option value="Solver">💡 Solver</option>
                          <option value="Coder">💻 Coder</option>
                          <option value="Tester/Debugger">🧪 Tester/Debugger</option>
                        </select>
                      </div>
                    </div>

                    <div className="flex items-center justify-between pt-1">
                      <div className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          id="mod-check-settings"
                          checked={newIsModerator}
                          onChange={(e) => setNewIsModerator(e.target.checked)}
                          className="rounded border-slate-800 text-emerald-500 focus:ring-emerald-500"
                        />
                        <label htmlFor="mod-check-settings" className="text-slate-300 font-medium cursor-pointer">
                          👑 Designate as Moderator
                        </label>
                      </div>

                      <button
                        type="submit"
                        className="bg-emerald-600 hover:bg-emerald-500 text-white px-4 py-2 rounded-xl font-medium"
                      >
                        Add Model to Room
                      </button>
                    </div>
                  </form>
                </div>

                {/* Known Models Library & Room Status */}
                <div className="space-y-2">
                  <h4 className="font-semibold text-xs text-slate-300 uppercase tracking-wider">
                    Known Models Library ({Object.keys(knownModels).length})
                  </h4>
                  <div className="space-y-2 text-xs">
                    {Object.values(knownModels).map((m) => {
                      const isInRoom = !!models[m.id];
                      return (
                        <div
                          key={m.id}
                          className="bg-slate-950 border border-slate-800 p-3 rounded-xl flex items-center justify-between"
                        >
                          <div className="space-y-0.5">
                            <div className="flex items-center gap-2 font-semibold text-slate-200">
                              <span>{m.name}</span>
                              <span className="text-[10px] bg-slate-800 text-slate-400 px-2 py-0.5 rounded font-mono uppercase">
                                {m.provider === 'gguf_local' ? 'GGUF' : m.provider}
                              </span>
                              {m.is_moderator && <span className="text-amber-400 text-[10px]">👑 Moderator</span>}
                              {isInRoom ? (
                                <span className="text-[10px] bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 px-1.5 py-0.5 rounded-full font-medium">In Room</span>
                              ) : (
                                <span className="text-[10px] bg-slate-800 text-slate-500 px-1.5 py-0.5 rounded-full font-medium">Kicked / Offline</span>
                              )}
                            </div>
                            <div className="text-[11px] text-slate-400">
                              Role: {m.role} | {m.gguf_path ? `Path: ${m.gguf_path}` : `Model: ${m.model_name}`}
                            </div>
                          </div>

                          <div className="flex items-center gap-2">
                            {isInRoom ? (
                              <button
                                onClick={() => handleKickModel(m.id)}
                                className="bg-rose-950/60 hover:bg-rose-900/80 text-rose-300 border border-rose-800/50 px-2.5 py-1 rounded-lg flex items-center gap-1 font-medium transition"
                              >
                                <UserMinus className="w-3.5 h-3.5" />
                                <span>Remove</span>
                              </button>
                            ) : (
                              <button
                                onClick={() => handleReaddModel(m.id)}
                                className="bg-emerald-600 hover:bg-emerald-500 text-white px-2.5 py-1 rounded-lg flex items-center gap-1 font-medium transition"
                              >
                                <UserPlus className="w-3.5 h-3.5" />
                                <span>Add to Room</span>
                              </button>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}

            {/* TAB 2: BRAINSTORM & HIRE FROM HUGGINGFACE */}
            {settingsTab === 'browse_hf' && (
              <div className="space-y-4 text-xs">
                <form onSubmit={handleSearchHf} className="flex gap-2">
                  <input
                    type="text"
                    value={hfQuery}
                    onChange={(e) => setHfQuery(e.target.value)}
                    placeholder="Search HuggingFace models e.g. Llama-3, Bonsai, Dolphin, Qwen GGUF"
                    className="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-slate-100 placeholder-slate-500 font-mono"
                  />
                  <button
                    type="submit"
                    disabled={isSearchingHf}
                    className="bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white px-4 py-2 rounded-xl font-medium"
                  >
                    {isSearchingHf ? 'Searching HF...' : 'Search HuggingFace'}
                  </button>
                </form>

                <div className="space-y-2 max-h-80 overflow-y-auto pr-1">
                  {hfResults.length === 0 ? (
                    <div className="text-slate-500 text-center py-8">
                      Search HuggingFace to discover and brainstorm new models for your team!
                    </div>
                  ) : (
                    hfResults.map((hf) => (
                      <div key={hf.model_id} className="bg-slate-950 border border-slate-800 p-3 rounded-xl flex items-center justify-between">
                        <div className="space-y-1">
                          <div className="flex items-center gap-2 font-semibold text-cyan-300">
                            <a href={hf.url} target="_blank" rel="noreferrer" className="hover:underline">
                              {hf.model_id}
                            </a>
                            {hf.is_gguf && (
                              <span className="text-[9px] bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 px-1.5 py-0.5 rounded uppercase font-mono">
                                GGUF Ready
                              </span>
                            )}
                          </div>
                          <div className="text-[10px] text-slate-400 flex items-center gap-3">
                            <span>❤️ Likes: {hf.likes}</span>
                            <span>📥 Downloads: {hf.downloads}</span>
                            <span>Tags: {hf.tags ? hf.tags.join(', ') : 'None'}</span>
                          </div>
                        </div>

                        <button
                          onClick={() => {
                            setNewModelName(hf.model_id.split('/').pop() || hf.model_id);
                            setNewGgufPath(`${hf.model_id.split('/').pop()}.gguf`);
                            setNewModelProvider('gguf_local');
                            setSettingsTab('models');
                          }}
                          className="bg-cyan-950 hover:bg-cyan-900 text-cyan-300 border border-cyan-800 px-3 py-1.5 rounded-lg font-medium transition shrink-0"
                        >
                          Hire Model
                        </button>
                      </div>
                    ))
                  )}
                </div>
              </div>
            )}

            {/* TAB 3: MODEL SEARCH DIRECTORIES */}
            {settingsTab === 'search_paths' && (
              <div className="space-y-4 text-xs">
                <div className="bg-slate-950 border border-slate-800 p-3 rounded-xl space-y-2">
                  <h4 className="font-semibold text-slate-200">Active GGUF Search Directories</h4>
                  <p className="text-slate-400 text-[11px]">
                    When you enter a filename like <code>Bonsai-27B-Q1_0.gguf</code>, SwarmChat automatically scans these folders to resolve the full absolute path!
                  </p>
                  <div className="space-y-1 font-mono text-[11px] text-emerald-400">
                    {searchPaths.map((p, idx) => (
                      <div key={idx} className="bg-slate-900 p-2 rounded border border-slate-800 truncate">
                        📁 {p}
                      </div>
                    ))}
                  </div>
                </div>

                <div className="bg-slate-950 border border-slate-800 p-3 rounded-xl space-y-2">
                  <h4 className="font-semibold text-slate-200">Add Custom Model Directory</h4>
                  <form
                    onSubmit={async (e) => {
                      e.preventDefault();
                      if (!customSearchPathInput) return;
                      await fetch('/api/models/search_paths', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ path: customSearchPathInput })
                      });
                      setCustomSearchPathInput('');
                      fetchState();
                    }}
                    className="flex gap-2"
                  >
                    <input
                      type="text"
                      value={customSearchPathInput}
                      onChange={(e) => setCustomSearchPathInput(e.target.value)}
                      placeholder="e.g. D:\MyGGUFModels or /home/user/downloads"
                      className="flex-1 bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-slate-100 font-mono"
                    />
                    <button type="submit" className="bg-purple-600 hover:bg-purple-500 text-white px-4 py-2 rounded-xl font-medium">
                      Add Path
                    </button>
                  </form>
                </div>
              </div>
            )}

            <div className="flex justify-end pt-2 border-t border-slate-800">
              <button
                onClick={() => setShowSetup(false)}
                className="bg-slate-800 hover:bg-slate-700 text-slate-200 px-4 py-2 rounded-xl text-xs font-medium"
              >
                Close Settings
              </button>
            </div>
          </div>
        </div>
      )}

      {/* --- MODERATOR REPLACEMENT PROMPT MODAL --- */}
      {showModPrompt && (
        <div className="fixed inset-0 bg-slate-950/85 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-md w-full p-6 shadow-xl space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="font-bold text-base text-amber-400 flex items-center gap-2">
                <Crown className="w-5 h-5" />
                <span>Select Replacement Moderator</span>
              </h3>
              <button onClick={handleCloseModPrompt} className="text-slate-400 hover:text-slate-200">
                <X className="w-5 h-5" />
              </button>
            </div>

            <p className="text-xs text-slate-300 leading-relaxed">
              The former moderator was removed. Please select a replacement moderator from the active chatroom models below.
              <span className="text-slate-400 block mt-1">(If closed without selecting, the app will automatically select an active participant).</span>
            </p>

            <div className="space-y-2 max-h-60 overflow-y-auto pr-1">
              {Object.values(models)
                .filter(m => m.id !== formerModId)
                .map((m) => (
                  <button
                    key={m.id}
                    onClick={() => handleSelectModerator(m.id)}
                    className="w-full bg-slate-950 hover:bg-slate-800 border border-slate-800 hover:border-amber-500/50 p-3 rounded-xl flex items-center justify-between text-left transition group"
                  >
                    <div>
                      <div className="font-semibold text-xs text-slate-200 group-hover:text-amber-300">{m.name}</div>
                      <div className="text-[10px] text-slate-400">Role: {m.role} | {m.provider}</div>
                    </div>
                    <Crown className="w-4 h-4 text-slate-600 group-hover:text-amber-400 transition" />
                  </button>
                ))}
            </div>

            <div className="pt-2 border-t border-slate-800 flex justify-end">
              <button
                onClick={handleCloseModPrompt}
                className="bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs px-3 py-1.5 rounded-lg transition"
              >
                Auto-assign & Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* --- TASK ITINERARY, EPISODES & WORKSPACE OVERLAY MENU --- */}
      {showOverlay && (
        <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-4xl w-full p-6 shadow-2xl flex flex-col h-[85vh]">
            
            {/* Modal Header & Navigation Tabs */}
            <div className="flex items-center justify-between border-b border-slate-800 pb-4 mb-4">
              <div className="flex items-center gap-2 font-bold text-lg text-slate-100">
                <Sparkles className="w-5 h-5 text-cyan-400" />
                <span>SwarmChat Operations & Workspace Control</span>
              </div>
              <button onClick={() => setShowOverlay(false)} className="text-slate-400 hover:text-slate-200 p-1">
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* TAB SELECTOR BUTTONS */}
            <div className="flex items-center gap-2 border-b border-slate-800 pb-3 mb-4">
              <button
                onClick={() => setOverlayTab('itinerary')}
                className={`flex items-center gap-2 px-4 py-2 rounded-xl font-medium text-xs transition ${
                  overlayTab === 'itinerary'
                    ? 'bg-cyan-500/10 text-cyan-300 border border-cyan-500/30'
                    : 'bg-slate-950 text-slate-400 hover:bg-slate-800 hover:text-slate-200'
                }`}
              >
                <CheckSquare className="w-4 h-4 text-cyan-400" />
                <span>Task Itinerary & Meetings</span>
              </button>

              <button
                onClick={() => setOverlayTab('episodes')}
                className={`flex items-center gap-2 px-4 py-2 rounded-xl font-medium text-xs transition ${
                  overlayTab === 'episodes'
                    ? 'bg-purple-500/10 text-purple-300 border border-purple-500/30'
                    : 'bg-slate-950 text-slate-400 hover:bg-slate-800 hover:text-slate-200'
                }`}
              >
                <BrainCircuit className="w-4 h-4 text-purple-400" />
                <span>Episodes & Thread Checkpoints ({episodes.length})</span>
              </button>

              <button
                onClick={() => setOverlayTab('workspace')}
                className={`flex items-center gap-2 px-4 py-2 rounded-xl font-medium text-xs transition ${
                  overlayTab === 'workspace'
                    ? 'bg-emerald-500/10 text-emerald-300 border border-emerald-500/30'
                    : 'bg-slate-950 text-slate-400 hover:bg-slate-800 hover:text-slate-200'
                }`}
              >
                <FileText className="w-4 h-4 text-emerald-400" />
                <span>Workspace Files & Change Logs</span>
              </button>

              <button
                onClick={() => setOverlayTab('health')}
                className={`flex items-center gap-2 px-4 py-2 rounded-xl font-medium text-xs transition ${
                  overlayTab === 'health'
                    ? 'bg-amber-500/10 text-amber-300 border border-amber-500/30'
                    : 'bg-slate-950 text-slate-400 hover:bg-slate-800 hover:text-slate-200'
                }`}
              >
                <Activity className="w-4 h-4 text-amber-400" />
                <span>Model Health & Evaluation</span>
              </button>
            </div>

            {/* TAB CONTENT PANELS */}
            <div className="flex-1 overflow-y-auto pr-2 space-y-4">
              
              {/* TAB 1: TASK ITINERARY */}
              {overlayTab === 'itinerary' && (
                <div className="space-y-4">
                  {/* Create Task Form */}
                  <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-3">
                    <h4 className="font-semibold text-xs text-cyan-400 flex items-center gap-2">
                      <Plus className="w-4 h-4" /> Add Itinerary Task / Meeting Item
                    </h4>
                    <form
                      onSubmit={async (e) => {
                        e.preventDefault();
                        if (!newTaskTitle) return;
                        await fetch('/api/itinerary/task', {
                          method: 'POST',
                          headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify({
                            title: newTaskTitle,
                            description: newTaskDesc,
                            priority: newTaskPriority
                          })
                        });
                        setNewTaskTitle('');
                        setNewTaskDesc('');
                        fetchState();
                      }}
                      className="space-y-3 text-xs"
                    >
                      <div className="grid grid-cols-3 gap-3">
                        <input
                          type="text"
                          value={newTaskTitle}
                          onChange={(e) => setNewTaskTitle(e.target.value)}
                          placeholder="Task title e.g. Refactor Memory Engine"
                          className="col-span-2 bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-slate-100 placeholder-slate-500"
                          required
                        />
                        <select
                          value={newTaskPriority}
                          onChange={(e) => setNewTaskPriority(e.target.value)}
                          className="bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-slate-100"
                        >
                          <option value="high">🔴 High Priority</option>
                          <option value="medium">🟡 Medium Priority</option>
                          <option value="low">🟢 Low Priority</option>
                        </select>
                      </div>
                      <input
                        type="text"
                        value={newTaskDesc}
                        onChange={(e) => setNewTaskDesc(e.target.value)}
                        placeholder="Detailed task instructions or meeting agenda objective..."
                        className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-slate-100 placeholder-slate-500"
                      />
                      <div className="flex justify-end">
                        <button type="submit" className="bg-cyan-600 hover:bg-cyan-500 text-white px-4 py-1.5 rounded-xl font-medium">
                          Add Task
                        </button>
                      </div>
                    </form>
                  </div>

                  {/* Task List */}
                  <div className="space-y-2">
                    <h4 className="font-semibold text-xs text-slate-400 uppercase">Itinerary Agenda Items ({taskItinerary.length})</h4>
                    {taskItinerary.length === 0 ? (
                      <div className="text-slate-500 text-xs py-4 text-center">No task items created yet.</div>
                    ) : (
                      taskItinerary.map((task) => (
                        <div key={task.id} className="bg-slate-950 border border-slate-800 p-3 rounded-xl flex items-center justify-between text-xs">
                          <div className="space-y-1 max-w-xl">
                            <div className="flex items-center gap-2 font-semibold text-slate-200">
                              <span>{task.title}</span>
                              <span className={`text-[10px] px-2 py-0.5 rounded font-mono uppercase ${
                                task.priority === 'high' ? 'bg-rose-500/10 text-rose-400 border border-rose-500/30' : 'bg-slate-800 text-slate-400'
                              }`}>
                                {task.priority}
                              </span>
                              <span className={`text-[10px] px-2 py-0.5 rounded font-mono uppercase ${
                                task.status === 'in_progress' ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/30' : 'bg-slate-800 text-slate-500'
                              }`}>
                                {task.status}
                              </span>
                            </div>
                            <div className="text-slate-400 text-[11px]">{task.description}</div>
                          </div>

                          <div className="flex items-center gap-2">
                            {task.status !== 'in_progress' && (
                              <button
                                onClick={async () => {
                                  await fetch('/api/itinerary/update', {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({ task_id: task.id, status: 'in_progress' })
                                  });
                                  fetchState();
                                }}
                                className="bg-cyan-950/80 hover:bg-cyan-900 text-cyan-300 border border-cyan-800 px-2.5 py-1 rounded font-medium"
                              >
                                Set Active
                              </button>
                            )}
                            {task.status !== 'completed' && (
                              <button
                                onClick={async () => {
                                  await fetch('/api/itinerary/update', {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({ task_id: task.id, status: 'completed' })
                                  });
                                  fetchState();
                                }}
                                className="bg-emerald-950/80 hover:bg-emerald-900 text-emerald-300 border border-emerald-800 px-2.5 py-1 rounded font-medium"
                              >
                                Complete
                              </button>
                            )}
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              )}

              {/* TAB 2: EPISODES & THREAD CHECKPOINTS */}
              {overlayTab === 'episodes' && (
                <div className="space-y-3">
                  <h4 className="font-semibold text-xs text-purple-400 uppercase">NAC-Style Episodes & Checkpoint History</h4>
                  {episodes.length === 0 ? (
                    <div className="text-slate-500 text-xs py-6 text-center">No episode checkpoints recorded yet. Models generate episodes on naps and handoffs.</div>
                  ) : (
                    episodes.map((ep) => (
                      <div key={ep.id} className="bg-slate-950 border border-slate-800 p-4 rounded-xl space-y-2 text-xs">
                        <div className="flex items-center justify-between">
                          <span className="font-semibold text-purple-300">📦 Episode #{ep.id} — [{ep.author}]</span>
                          <span className="text-[10px] text-slate-500">{new Date(ep.timestamp * 1000).toLocaleTimeString()}</span>
                        </div>
                        <div className="text-slate-300 font-medium">Action: {ep.action}</div>
                        <div className="text-slate-400 leading-relaxed">{ep.summary}</div>
                        {ep.modified_files && ep.modified_files.length > 0 && (
                          <div className="text-[11px] text-emerald-400">
                            Files Modified: <code>{ep.modified_files.join(', ')}</code>
                          </div>
                        )}
                      </div>
                    ))
                  )}
                </div>
              )}

              {/* TAB 3: WORKSPACE FILES & CHANGE LOGS */}
              {overlayTab === 'workspace' && (
                <div className="grid grid-cols-2 gap-4 h-full">
                  {/* Left Column: File Tree */}
                  <div className="bg-slate-950 border border-slate-800 rounded-xl p-3 flex flex-col space-y-2">
                    <h4 className="font-semibold text-xs text-emerald-400 uppercase">Project File Explorer</h4>
                    <div className="flex-1 overflow-y-auto space-y-1 text-xs">
                      {workspaceFiles.map((file) => (
                        <button
                          key={file.path}
                          onClick={async () => {
                            setSelectedFilePath(file.path);
                            const res = await fetch(`/api/workspace/file?filepath=${encodeURIComponent(file.path)}`);
                            if (res.ok) {
                              const data = await res.json();
                              setSelectedFileContent(data.content || '');
                            }
                          }}
                          className={`w-full text-left px-2.5 py-1.5 rounded flex items-center justify-between font-mono text-[11px] ${
                            selectedFilePath === file.path ? 'bg-emerald-950 text-emerald-300 border border-emerald-800/50' : 'text-slate-300 hover:bg-slate-900'
                          }`}
                        >
                          <span>{file.name}</span>
                          {activeFileLocks[file.path] && (
                            <span className="text-[9px] bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 px-1 rounded">
                              By {activeFileLocks[file.path].last_edited_by}
                            </span>
                          )}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Right Column: Content Preview & Audit Log */}
                  <div className="flex flex-col space-y-3">
                    {/* File Content Preview */}
                    <div className="flex-1 bg-slate-950 border border-slate-800 rounded-xl p-3 flex flex-col font-mono text-[11px] overflow-hidden">
                      <div className="font-semibold text-xs text-slate-300 border-b border-slate-800 pb-2 mb-2 flex items-center justify-between">
                        <span>{selectedFilePath || 'Select a file to view content'}</span>
                      </div>
                      <pre className="flex-1 overflow-auto text-slate-300 leading-relaxed whitespace-pre-wrap">
                        {selectedFileContent || '// Select a file from the explorer on the left.'}
                      </pre>
                    </div>

                    {/* Change Attribution Audit Log */}
                    <div className="h-36 bg-slate-950 border border-slate-800 rounded-xl p-3 flex flex-col overflow-y-auto text-xs space-y-1">
                      <h5 className="font-semibold text-[11px] text-slate-400 uppercase">File Change Audit Log</h5>
                      {fileAuditLog.length === 0 ? (
                        <div className="text-slate-500 text-[10px]">No file modifications recorded yet.</div>
                      ) : (
                        fileAuditLog.map((log) => (
                          <div key={log.id} className="text-[10px] text-slate-300 border-b border-slate-900 pb-1">
                            <strong className="text-emerald-400">[{log.author}]</strong> modified <code>{log.filepath}</code> — {log.diff_snippet}
                          </div>
                        ))
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* TAB 4: MODEL HEALTH & EVALUATION */}
              {overlayTab === 'health' && (
                <div className="space-y-3">
                  <h4 className="font-semibold text-xs text-amber-400 uppercase">Moderator Model Health & Evaluation Report</h4>
                  <div className="space-y-2">
                    {roomHealth.map((rep) => (
                      <div key={rep.model_id} className="bg-slate-950 border border-slate-800 p-3 rounded-xl flex items-center justify-between text-xs">
                        <div>
                          <div className="font-semibold text-slate-200">{rep.model_name} ({rep.role})</div>
                          <div className="text-[11px] text-slate-400">
                            Tokens Used: {rep.tokens_used} | Turns Taken: {rep.turns_count}
                          </div>
                          {rep.recommendation && (
                            <div className="text-[11px] text-amber-300 mt-1">{rep.recommendation}</div>
                          )}
                        </div>

                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => handleKickModel(rep.model_id)}
                            className="bg-rose-950/80 hover:bg-rose-900 text-rose-300 border border-rose-800 px-3 py-1 rounded font-medium"
                          >
                            Kick Model
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

            </div>

            <div className="pt-3 border-t border-slate-800 flex justify-end">
              <button onClick={() => setShowOverlay(false)} className="bg-slate-800 hover:bg-slate-700 text-slate-200 px-4 py-1.5 rounded-xl text-xs font-medium">
                Close Operations Menu
              </button>
            </div>
          </div>
        </div>
      )}

      {/* --- SHARED MEMORY MODAL --- */}
      {showMemory && (
        <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-xl w-full p-6 shadow-xl space-y-4 max-h-[80vh] flex flex-col">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="font-bold text-lg text-slate-100 flex items-center gap-2">
                <BrainCircuit className="w-5 h-5 text-purple-400" />
                <span>Continuous Shared Memory Archive</span>
              </h3>
              <button onClick={() => setShowMemory(false)} className="text-slate-400 hover:text-slate-200">
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto space-y-2 pr-2 text-xs">
              {sharedMemory.length === 0 ? (
                <div className="text-slate-500 text-center py-6">No entries recorded in shared memory yet.</div>
              ) : (
                sharedMemory.map((mem) => (
                  <div key={mem.id} className="bg-slate-950 border border-slate-800/80 p-3 rounded-xl space-y-1">
                    <div className="flex items-center justify-between text-slate-400 text-[10px]">
                      <span className="font-semibold text-purple-300">{mem.author}</span>
                      <span>{new Date(mem.timestamp * 1000).toLocaleTimeString()}</span>
                    </div>
                    <div className="text-slate-200">{mem.content}</div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

import React, { useState, useEffect, useRef } from 'react';
import {
  MessageSquare, Zap, Settings, Shield, BrainCircuit, AlertTriangle,
  Play, Crown, Cpu, Sparkles, Send, X, Users, Check, Plus, FolderOpen, Trash2,
  ChevronDown, ChevronRight, UserMinus, UserPlus, RefreshCw
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
  const [pendingVotes, setPendingVotes] = useState<PendingVote[]>([]);
  const [sharedMemory, setSharedMemory] = useState<any[]>([]);
  const [hardware, setHardware] = useState<any>(null);
  
  // UI Panels & Modals
  const [showSetup, setShowSetup] = useState(false);
  const [showMemory, setShowMemory] = useState(false);
  const [showModPrompt, setShowModPrompt] = useState(false);
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
  const [newApiKey, setNewApiKey] = useState('');
  const [newIsModerator, setNewIsModerator] = useState(false);

  const chatEndRef = useRef<HTMLDivElement>(null);

  const fetchState = async () => {
    try {
      const res = await fetch('/api/state');
      if (res.ok) {
        const data = await res.json();
        setPhase(data.phase);
        setModels(data.models || {});
        setKnownModels(data.known_models || data.models || {});
        setPendingVotes(data.pending_votes || []);
        setMessages(data.chat_history || []);
        setSharedMemory(data.shared_memory || []);
      }
      const hwRes = await fetch('/api/hardware');
      if (hwRes.ok) {
        setHardware(await hwRes.json());
      }
    } catch (e) {
      console.error('Fetch state error:', e);
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
    setNewApiKey('');
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
      <header className="flex items-center justify-between px-5 py-3 bg-slate-900/90 border-b border-slate-800 backdrop-blur shrink-0">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 font-bold text-lg text-emerald-400">
            <Sparkles className="w-5 h-5 text-emerald-400" />
            <span>SwarmChat</span>
          </div>
          <div className="h-4 w-px bg-slate-700" />
          
          {/* Phase Badge & Switch Button */}
          <button
            onClick={togglePhase}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-full font-medium text-xs tracking-wide transition-all shadow-sm ${
              phase === 'discussion'
                ? 'bg-amber-500/10 text-amber-400 border border-amber-500/30 hover:bg-amber-500/20'
                : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 hover:bg-emerald-500/20'
            }`}
          >
            {phase === 'discussion' ? (
              <>
                <MessageSquare className="w-3.5 h-3.5" />
                <span>💬 DISCUSSION PHASE (Preparatory & Mutual Understanding)</span>
              </>
            ) : (
              <>
                <Zap className="w-3.5 h-3.5" />
                <span>⚡ EXECUTION PHASE (Tool Active & Coding Work)</span>
              </>
            )}
            <span className="text-[10px] opacity-70 underline ml-1">Click to switch</span>
          </button>
        </div>

        {/* Hardware Status & Top Action Buttons */}
        <div className="flex items-center gap-3">
          {hardware && (
            <div className="flex items-center gap-3 text-xs bg-slate-800/60 px-3 py-1.5 rounded-lg border border-slate-700/50">
              <div className="flex items-center gap-1 text-slate-300">
                <Cpu className="w-3.5 h-3.5 text-slate-400" />
                <span>RAM: {hardware.ram_available_gb}GB / {hardware.ram_total_gb}GB</span>
              </div>
              {hardware.gpu_name && (
                <span className="text-emerald-400">GPU VRAM: {hardware.vram_free_gb}GB free</span>
              )}
            </div>
          )}

          <button
            onClick={() => setShowSetup(true)}
            className="flex items-center gap-1.5 text-xs bg-emerald-600 hover:bg-emerald-500 text-white font-medium px-3 py-1.5 rounded-lg transition shadow-sm"
          >
            <Settings className="w-3.5 h-3.5" />
            <span>Settings & Models</span>
          </button>

          <button
            onClick={() => setShowMemory(!showMemory)}
            className="flex items-center gap-1.5 text-xs bg-slate-800 hover:bg-slate-700 text-slate-200 px-3 py-1.5 rounded-lg border border-slate-700 transition"
          >
            <BrainCircuit className="w-3.5 h-3.5 text-purple-400" />
            <span>Shared Memory ({sharedMemory.length})</span>
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
              {Object.values(models).map((m) => (
                <div
                  key={m.id}
                  className="bg-slate-950 border border-slate-800/80 rounded-xl p-3 flex flex-col gap-2 relative group"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
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

                  <div className="text-xs text-slate-400 flex items-center gap-1">
                    <span>Role:</span>
                    <span className="text-slate-200 font-medium">{m.role}</span>
                  </div>

                  <div className="text-[11px] text-slate-500 truncate" title={m.gguf_path || m.model_name}>
                    {m.provider === 'gguf_local' ? (
                      <code>File: {m.gguf_path || m.model_name}</code>
                    ) : (
                      <code>Model: {m.model_name}</code>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </aside>
        )}
      </div>

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

            {/* SECTION 1: Add New Model / GGUF File Picker */}
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
                  <div className="space-y-2 bg-slate-900 border border-slate-800 p-3 rounded-xl">
                    <label className="block text-slate-300 font-medium">Select GGUF File from Hard Drive</label>
                    <div className="flex gap-2 items-center">
                      <input
                        type="file"
                        accept=".gguf"
                        onChange={handleFileSelect}
                        className="hidden"
                        id="gguf-file-picker-settings"
                      />
                      <label
                        htmlFor="gguf-file-picker-settings"
                        className="bg-slate-800 hover:bg-slate-700 text-slate-200 px-3 py-2 rounded-lg cursor-pointer flex items-center gap-1.5 font-medium shrink-0 border border-slate-700"
                      >
                        <FolderOpen className="w-4 h-4 text-amber-400" />
                        <span>Browse GGUF...</span>
                      </label>
                      <input
                        type="text"
                        value={newGgufPath}
                        onChange={(e) => setNewGgufPath(e.target.value)}
                        placeholder="Or type full filepath e.g. C:\models\bonsai-1.7b.gguf"
                        className="flex-1 bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-100 placeholder-slate-500 font-mono text-[11px]"
                      />
                    </div>
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

            {/* SECTION 2: Known Models Library & Room Status */}
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

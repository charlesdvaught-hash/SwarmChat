import React, { useState, useEffect, useRef } from 'react';
import {
  MessageSquare, Zap, Settings, Shield, BrainCircuit, AlertTriangle,
  Play, Crown, Cpu, Sparkles, Send, X, Users, Check, Plus, FolderOpen, Trash2
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
  const [pendingVotes, setPendingVotes] = useState<PendingVote[]>([]);
  const [sharedMemory, setSharedMemory] = useState<any[]>([]);
  const [hardware, setHardware] = useState<any>(null);

  // UI Panels & Modals
  const [showSetup, setShowSetup] = useState(false);
  const [showAddModel, setShowAddModel] = useState(false);
  const [showMemory, setShowMemory] = useState(false);
  const [showEvaluate, setShowEvaluate] = useState(false);
  const [showSidebar, setShowSidebar] = useState(true);
  const [isStepping, setIsStepping] = useState(false);

  // New Model Form State
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
      // In web browser context, file.name or file.path gives the path
      const filePath = (file as any).path || file.name;
      setNewGgufPath(filePath);
      if (!newModelName) {
        // Auto derive friendly model name
        const cleanName = file.name.replace(/\.gguf$/i, '').replace(/[-_]/g, ' ');
        setNewModelName(cleanName);
      }
    }
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

    setShowAddModel(false);
    setNewModelName('');
    setNewGgufPath('');
    setNewApiKey('');
    fetchState();
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
            onClick={() => setShowAddModel(true)}
            className="flex items-center gap-1.5 text-xs bg-emerald-600 hover:bg-emerald-500 text-white font-medium px-3 py-1.5 rounded-lg transition shadow-sm"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>Add GGUF / Model</span>
          </button>

          <button
            onClick={() => setShowMemory(!showMemory)}
            className="flex items-center gap-1.5 text-xs bg-slate-800 hover:bg-slate-700 text-slate-200 px-3 py-1.5 rounded-lg border border-slate-700 transition"
          >
            <BrainCircuit className="w-3.5 h-3.5 text-purple-400" />
            <span>Shared Memory ({sharedMemory.length})</span>
          </button>

          <button
            onClick={() => setShowSetup(!showSetup)}
            className="p-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg border border-slate-700 transition"
            title="Setup & Hardware Diagnostics"
          >
            <Settings className="w-4 h-4" />
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
                    Multiple models can be loaded at once. Type a message below or click "Trigger Model Turn" to start model interactions.
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
                    <div className="whitespace-pre-wrap">{msg.content}</div>
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

        {/* RIGHT SIDEBAR: Users, Roles, Moderator & Active Models Panel */}
        {showSidebar && (
          <aside className="w-80 bg-slate-900 border-l border-slate-800 flex flex-col shrink-0">
            <div className="p-4 border-b border-slate-800 flex items-center justify-between">
              <div className="flex items-center gap-2 font-semibold text-sm text-slate-200">
                <Users className="w-4 h-4 text-cyan-400" />
                <span>Room Models ({Object.keys(models).length})</span>
              </div>
              <button
                onClick={() => setShowAddModel(true)}
                className="text-xs bg-slate-800 hover:bg-slate-700 text-emerald-400 p-1.5 rounded-lg border border-slate-700 flex items-center gap-1"
                title="Add GGUF or Model"
              >
                <Plus className="w-3.5 h-3.5" />
              </button>
            </div>

            <div className="p-4 space-y-3 overflow-y-auto flex-1">
              {Object.values(models).map((m) => (
                <div
                  key={m.id}
                  className="bg-slate-950 border border-slate-800/80 rounded-xl p-3 flex flex-col gap-2 relative"
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
                    <span className="text-[10px] text-slate-400 bg-slate-800 px-2 py-0.5 rounded uppercase font-mono">
                      {m.provider === 'gguf_local' ? 'GGUF' : m.provider}
                    </span>
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

      {/* --- ADD GGUF / MODEL MODAL --- */}
      {showAddModel && (
        <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-lg w-full p-6 shadow-xl space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="font-bold text-lg text-slate-100 flex items-center gap-2">
                <Plus className="w-5 h-5 text-emerald-400" />
                <span>Add Participant / Model to Chat Room</span>
              </h3>
              <button onClick={() => setShowAddModel(false)} className="text-slate-400 hover:text-slate-200">
                <X className="w-5 h-5" />
              </button>
            </div>

            <form onSubmit={handleAddModelSubmit} className="space-y-4 text-xs">
              {/* Provider Selection */}
              <div>
                <label className="block text-slate-300 font-medium mb-1">Model Provider Type</label>
                <select
                  value={newModelProvider}
                  onChange={(e) => setNewModelProvider(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-slate-100 focus:outline-none focus:border-emerald-500"
                >
                  <option value="gguf_local">📁 Local GGUF File (.gguf on hard drive)</option>
                  <option value="ollama">🦙 Ollama Local Model</option>
                  <option value="claude">☁️ Claude API (Anthropic)</option>
                  <option value="groq">⚡ Groq API</option>
                  <option value="gemini">♊ Gemini API (Google)</option>
                </select>
              </div>

              {/* Local GGUF Picker */}
              {newModelProvider === 'gguf_local' && (
                <div className="space-y-2 bg-slate-950 border border-slate-800 p-3 rounded-xl">
                  <label className="block text-slate-300 font-medium">Select GGUF File from Hard Drive</label>
                  <div className="flex gap-2 items-center">
                    <input
                      type="file"
                      accept=".gguf"
                      onChange={handleFileSelect}
                      className="hidden"
                      id="gguf-file-picker"
                    />
                    <label
                      htmlFor="gguf-file-picker"
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
                      className="flex-1 bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-slate-100 placeholder-slate-500 font-mono text-[11px]"
                    />
                  </div>
                </div>
              )}

              {/* Display Model Tag for Cloud / Ollama */}
              {newModelProvider !== 'gguf_local' && (
                <div>
                  <label className="block text-slate-300 font-medium mb-1">Model Name / Tag</label>
                  <input
                    type="text"
                    value={newModelNameOrTag}
                    onChange={(e) => setNewModelNameOrTag(e.target.value)}
                    placeholder="e.g. llama3.2:1b, claude-3-5-sonnet-20241022, groq-llama3"
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-slate-100 font-mono"
                  />
                </div>
              )}

              {/* Model Name in Chat */}
              <div>
                <label className="block text-slate-300 font-medium mb-1">Display Name in Chat</label>
                <input
                  type="text"
                  value={newModelName}
                  onChange={(e) => setNewModelName(e.target.value)}
                  placeholder="e.g. Bonsai Solver, Llama Architect, Claude Critic"
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-slate-100"
                  required
                />
              </div>

              {/* Role Selection */}
              <div>
                <label className="block text-slate-300 font-medium mb-1">Assigned Role</label>
                <select
                  value={newModelRole}
                  onChange={(e) => setNewModelRole(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-slate-100 focus:outline-none focus:border-emerald-500"
                >
                  <option value="Architect">🏗️ Architect — Design & System Blueprints</option>
                  <option value="Critic">🧐 Critic — Red-Teaming & Risk Analysis</option>
                  <option value="Solver">💡 Solver — Algorithms & Core Problem Solving</option>
                  <option value="Coder">💻 Coder — Concrete Code & Patches</option>
                  <option value="Tester/Debugger">🧪 Tester/Debugger — QA & Bug Verification</option>
                </select>
              </div>

              {/* Moderator Toggle */}
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="is-mod-check"
                  checked={newIsModerator}
                  onChange={(e) => setNewIsModerator(e.target.checked)}
                  className="rounded border-slate-800 text-emerald-500 focus:ring-emerald-500"
                />
                <label htmlFor="is-mod-check" className="text-slate-300 font-medium cursor-pointer">
                  👑 Designate as Moderator (manages turns & context naps)
                </label>
              </div>

              <div className="flex justify-end gap-2 pt-2 border-t border-slate-800">
                <button
                  type="button"
                  onClick={() => setShowAddModel(false)}
                  className="bg-slate-800 hover:bg-slate-700 text-slate-300 px-4 py-2 rounded-xl text-xs font-medium"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="bg-emerald-600 hover:bg-emerald-500 text-white px-4 py-2 rounded-xl text-xs font-medium"
                >
                  Add Model to Room
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* --- SETUP / DEPENDENCY WIZARD MODAL --- */}
      {showSetup && (
        <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-lg w-full p-6 shadow-xl space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="font-bold text-lg text-slate-100 flex items-center gap-2">
                <Settings className="w-5 h-5 text-emerald-400" />
                <span>Hardware & Diagnostic Manager</span>
              </h3>
              <button onClick={() => setShowSetup(false)} className="text-slate-400 hover:text-slate-200">
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-3 text-xs text-slate-300">
              <div className="bg-slate-950 border border-slate-800 p-3 rounded-xl flex items-center justify-between">
                <div>
                  <div className="font-semibold text-slate-200">Local Hardware & VRAM Status</div>
                  <div className="text-slate-400">Automatic safety headroom monitoring.</div>
                </div>
                <button
                  onClick={fetchState}
                  className="bg-emerald-600 hover:bg-emerald-500 text-white px-3 py-1.5 rounded-lg font-medium"
                >
                  Check Hardware
                </button>
              </div>

              <div className="bg-slate-950 border border-slate-800 p-3 rounded-xl space-y-2">
                <div className="font-semibold text-slate-200">Supported Quantizations & Models</div>
                <p className="text-slate-400">
                  Load multiple models concurrently: local `.gguf` files (Bonsai 1.7B 1-bit, MoEs, Qwen 0.5B, Llama 3.2 1B), Ollama, and Cloud backends.
                </p>
              </div>
            </div>

            <div className="flex justify-end">
              <button
                onClick={() => setShowSetup(false)}
                className="bg-slate-800 hover:bg-slate-700 text-slate-200 px-4 py-2 rounded-xl text-xs font-medium"
              >
                Close
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

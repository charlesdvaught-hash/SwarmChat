import React, { useState, useEffect, useRef } from 'react';
import {
  MessageSquare, Zap, Settings, Shield, BrainCircuit, AlertTriangle,
  Play, Crown, Cpu, Sparkles, Send, X, Users, Check, Plus, FolderOpen, Trash2,
  ChevronDown, ChevronRight, UserMinus, UserPlus, RefreshCw, FileText, CheckSquare, Activity,
  HelpCircle
} from 'lucide-react';

// How often Auto mode asks for the next turn. Turns are serialised client-side, so this
// is a floor on the gap between turns, not a rate: a 40s turn simply takes 40s.
const AUTO_TURN_INTERVAL_MS = 2000;

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
  status: string;
  live_status?: string;
}

/** The supervisor seat is the Architect role - there is no separate moderator flag.
 *  (Backends before this change sent `is_moderator`; role is now the only source of truth.) */
const isSupervisor = (m: { role?: string }) => {
  const role = (m.role || '').toLowerCase();
  return role.includes('architect') || role.includes('planner');
};

interface PendingVote {
  id: string;
  model_id: string;
  model_name: string;
  tool_name: string;
  args: any;
  risk_level: string;
  status: string;
}

// A planning question the room has put on the board. One routed to the Admin is PARKED,
// never blocking: the room moves on to the next question or the next task, and answering
// resumes it. So this card is not a modal and does not interrupt anything.
interface PlanQuestionOption {
  label: string;
  means: string;
}

interface PlanQuestion {
  id: string;
  number: number;
  text_internal: string;
  question_admin: string;
  options: PlanQuestionOption[];
  recommended: string;
  rationale: string;
  resolvable_by: 'model' | 'admin';
  status: 'open' | 'parked' | 'resolved';
  answer: string;
  decided_by: string;
}

// Pre-execution planning gate, in running order.
const PLAN_STAGE_LABELS: Record<string, string> = {
  awaiting_questions: '❓ QUESTIONS',
  resolving_questions: '🧩 RESOLVING',
  awaiting_plan: '📝 PLAN',
  critic_review: '🔍 CRITIC REVIEW',
  programmer_review: '🔧 BUILDABILITY',
  approved: '✅ APPROVED',
};

export default function App() {
  const [phase, setPhase] = useState<'discussion' | 'execution'>('discussion');
  // Pre-execution planning gate, mirrored from /api/state.
  const [planStage, setPlanStage] = useState<string>('awaiting_plan');
  const [planStageOwnerName, setPlanStageOwnerName] = useState<string | null>(null);
  const [planRevision, setPlanRevision] = useState<number>(0);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputText, setInputText] = useState('');
  const [models, setModels] = useState<Record<string, ModelConfig>>({});
  const [knownModels, setKnownModels] = useState<Record<string, ModelConfig>>({});
  const [modelStatuses, setModelStatuses] = useState<Record<string, any>>({});
  const [pendingVotes, setPendingVotes] = useState<PendingVote[]>([]);
  const [planQuestions, setPlanQuestions] = useState<PlanQuestion[]>([]);
  // Only questions actually routed to the Admin appear in the UI. Questions the room
  // settles itself are room business and would be noise on a non-coder's screen.
  const parkedQuestions = planQuestions.filter(
    (q) => q.status === 'parked' && q.resolvable_by === 'admin' && (q.options || []).length > 0
  );
  const [sharedMemory, setSharedMemory] = useState<any[]>([]);
  const [hardware, setHardware] = useState<any>(null);
  const [dependencies, setDependencies] = useState<any>(null);
  const [isInstallingEngine, setIsInstallingEngine] = useState(false);
  const [selectedErrorModel, setSelectedErrorModel] = useState<string | null>(null);
  
  // UI Panels & Modals
  const [showSetup, setShowSetup] = useState(false);
  const [showMemory, setShowMemory] = useState(false);
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

  // Projects (each project owns its own tasks, shared memory and bot workspaces)
  const [projects, setProjects] = useState<any[]>([]);
  const [activeProjectId, setActiveProjectId] = useState<string>('default_project');
  const [newProjectName, setNewProjectName] = useState('');

  // Is the served UI bundle older than frontend/src?
  const [frontendStatus, setFrontendStatus] = useState<any>(null);
  const [isRebuilding, setIsRebuilding] = useState(false);

  // New Itinerary Task Form
  const [newTaskTitle, setNewTaskTitle] = useState('');
  const [newTaskDesc, setNewTaskDesc] = useState('');
  const [newTaskPriority, setNewTaskPriority] = useState('medium');

  const [showSidebar, setShowSidebar] = useState(true);
  const [isStepping, setIsStepping] = useState(false);
  const [collapsedCodeBlocks, setCollapsedCodeBlocks] = useState<Record<string, boolean>>({});

  // Roster & Auto-Turn State
  const [turnSchedule, setTurnSchedule] = useState<string[]>([]);
  const [isAutoTurn, setIsAutoTurn] = useState(false);
  // True while the backend is already running its own conversation loop (which is what
  // sending a chat message starts). Auto must yield to it rather than step in parallel.
  const [loopActive, setLoopActive] = useState(false);
  const [showRosterModal, setShowRosterModal] = useState(false);
  const [editedRoster, setEditedRoster] = useState<string[]>([]);

  // @ Mention Autocomplete state
  const [showMentionDropdown, setShowMentionDropdown] = useState(false);
  const [mentionQuery, setMentionQuery] = useState('');
  const [mentionCursorPos, setMentionCursorPos] = useState<number>(0);

  // New Model Form State in Settings
  const [newModelName, setNewModelName] = useState('');
  const [newModelRole, setNewModelRole] = useState('Architect');
  const [newModelProvider, setNewModelProvider] = useState('gguf_local');
  const [newModelNameOrTag, setNewModelNameOrTag] = useState('');
  const [newGgufPath, setNewGgufPath] = useState('');
  const [newMmprojPath, setNewMmprojPath] = useState('');
  const [newApiKey, setNewApiKey] = useState('');

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
  const loopActiveRef = useRef(false);

  // Backend failures are shown to the Admin instead of only reaching the browser console.
  const [apiError, setApiError] = useState<string | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);

  const describeFailure = async (res: Response): Promise<string> => {
    try {
      const body = await res.json();
      const detail = body?.detail ?? body?.error;
      if (typeof detail === 'string' && detail) return detail;
    } catch {
      // Body was not JSON; fall back to the status line.
    }
    return `HTTP ${res.status} ${res.statusText}`.trim();
  };

  const apiRequest = async <T,>(url: string, init?: RequestInit): Promise<T> => {
    const res = await fetch(url, init);
    if (!res.ok) throw new Error(await describeFailure(res));
    return (await res.json()) as T;
  };

  const postJson = <T,>(url: string, body: unknown): Promise<T> => apiRequest<T>(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });

  const post = <T,>(url: string): Promise<T> => apiRequest<T>(url, { method: 'POST' });

  const errorText = (e: unknown): string => (e instanceof Error ? e.message : String(e));

  /** Runs an action, surfacing any failure to the Admin as a labelled banner. */
  const runAction = async (label: string, action: () => Promise<void>): Promise<boolean> => {
    try {
      await action();
      setApiError(null);
      return true;
    } catch (e) {
      setApiError(`${label} failed: ${errorText(e)}`);
      return false;
    }
  };

  // Everything on screen that belongs to ONE project. Switching or archiving a project
  // must blank these BEFORE the request goes out, not after the refetch lands.
  //
  // The backend already swaps correctly (chat history, the question board and pending votes
  // are all per-project and are cleared in Orchestrator.set_project). The stale window was
  // purely visual: the switch does real file I/O, so for that beat the room still showed the
  // previous project's admin chat - including "[YOUR CALL]" question cards belonging to a
  // project you had just left, which are clickable. Worse, if the refetch failed, the old
  // conversation simply stayed on screen with nothing saying it was stale. An empty room
  // plus an error is honest; another project's room is not.
  const clearRoomView = () => {
    setMessages([]);
    setPlanQuestions([]);
    setPendingVotes([]);
    setSharedMemory([]);
    setEpisodes([]);
    setTaskItinerary([]);
    setActiveTask(null);
    setFileAuditLog([]);
    setActiveFileLocks({});
    setTurnSchedule([]);
    setPlanStage('awaiting_questions');
    setPlanStageOwnerName(null);
    setPlanRevision(0);
  };

  const fetchState = async () => {
    try {
      const data = await apiRequest<any>('/api/state');
      setPhase(data.phase);
      setPlanStage(data.plan_stage || 'awaiting_questions');
      setPlanStageOwnerName(data.plan_stage_owner_name || null);
      setPlanRevision(data.plan_revision || 0);
      setModels(data.models || {});
      setKnownModels(data.known_models || data.models || {});
      setModelStatuses(data.model_statuses || {});
      setPendingVotes(data.pending_votes || []);
      setPlanQuestions(data.plan_questions || []);
      setMessages(data.chat_history || []);
      setSharedMemory(data.shared_memory || []);
      setEpisodes(data.episodes || []);
      setTaskItinerary(data.task_itinerary || []);
      setActiveTask(data.active_task || null);
      setFileAuditLog(data.file_audit_log || []);
      setActiveFileLocks(data.active_file_locks || {});
      setTurnSchedule(data.turn_schedule || []);
      setLoopActive(Boolean(data.loop_active));
      setProjects(data.projects || []);
      setActiveProjectId(data.project_id || 'default_project');
      setFrontendStatus(data.frontend_status || null);
      // Failures that happen outside a request (shared memory, background conversation loop).
      const serverIssue = data.memory_error || data.last_background_error;
      setConnectionError(serverIssue ? `Server: ${serverIssue}` : null);

      setHardware(await apiRequest<any>('/api/hardware'));
      setDependencies(await apiRequest<any>('/api/dependencies'));
      const fdata = await apiRequest<any>('/api/workspace/files');
      setWorkspaceFiles(fdata.items || []);
      const hdata = await apiRequest<any>('/api/evaluate/health');
      setRoomHealth(hdata.reports || []);
      const pdata = await apiRequest<any>('/api/models/search_paths');
      setSearchPaths(pdata.search_paths || []);
    } catch (e) {
      setConnectionError(`Could not refresh state: ${errorText(e)}`);
    }
  };

  const handleBrowseFs = async (targetPath?: string) => {
    const url = targetPath ? `/api/fs/browse?path=${encodeURIComponent(targetPath)}` : '/api/fs/browse';
    await runAction('Browsing the server filesystem', async () => {
      const data = await apiRequest<any>(url);
      setFsCurrentPath(data.current_path);
      setFsParentPath(data.parent_path);
      setFsDirs(data.directories || []);
      setFsFiles(data.files || []);
    });
  };

  const handleValidatePath = async (p: string, mmP?: string) => {
    if (!p) return;
    try {
      const data = await postJson<any>('/api/fs/validate', { path: p, mmproj_path: mmP });
      setPathValidationMsg({ valid: data.valid, message: data.message, size_gb: data.file_size_gb });
    } catch (e) {
      setPathValidationMsg({ valid: false, message: `Validation request failed: ${errorText(e)}` });
    }
  };

  const handleSearchHf = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!hfQuery.trim()) return;
    setIsSearchingHf(true);
    try {
      const data = await apiRequest<any>(`/api/tools/search_hf?query=${encodeURIComponent(hfQuery)}`);
      setHfResults(data.models || []);
      setApiError(null);
    } catch (e) {
      // An empty list would read as "no such model", so the lookup failure is reported instead.
      setHfResults([]);
      setApiError(`HuggingFace search failed: ${errorText(e)}`);
    } finally {
      setIsSearchingHf(false);
    }
  };

  const handleInstallEngine = async () => {
    setIsInstallingEngine(true);
    try {
      await runAction('Installing the llama-cpp-python engine', async () => {
        await post('/api/engine/install');
      });
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

  // Auto mode.
  //
  // The toggle used to be decoration: `isAutoTurn` was read in exactly two places, both
  // of them its own label and colour. Nothing advanced a turn, so "Auto: ON" and
  // "Auto: OFF" behaved identically and the room only ever moved when the Admin pressed
  // "Next Turn" or sent a message.
  //
  // It now drives the same endpoint the manual button uses - one turn at a time, never
  // overlapping, and never while the backend is already running its own loop. A rejected
  // step (409: no eligible speaker) switches Auto off and says why, instead of retrying
  // forever against a room that cannot move.
  useEffect(() => {
    if (!isAutoTurn) return;
    let cancelled = false;
    let inFlight = false;

    const tick = async () => {
      if (cancelled || inFlight) return;
      // The backend loop started by /api/chat/message is already taking turns. Wait it out.
      if (loopActiveRef.current) return;
      inFlight = true;
      try {
        await post('/api/chat/step');
        if (!cancelled) await fetchState();
      } catch (e) {
        if (!cancelled) {
          setIsAutoTurn(false);
          setApiError(`Auto mode stopped: ${errorText(e)}`);
        }
      } finally {
        inFlight = false;
      }
    };

    void tick();
    const id = setInterval(tick, AUTO_TURN_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [isAutoTurn]);

  // A ref, not the state value: the interval closure above is created once per toggle and
  // would otherwise read whatever `loopActive` was at that moment, forever.
  useEffect(() => {
    loopActiveRef.current = loopActive;
  }, [loopActive]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const togglePhase = async () => {
    const nextPhase = phase === 'discussion' ? 'execution' : 'discussion';
    await runAction(`Switching to the ${nextPhase} phase`, async () => {
      await postJson('/api/phase', { phase: nextPhase });
    });
    fetchState();
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    const cursor = e.target.selectionStart || 0;
    setInputText(value);

    // Detect @ symbol before cursor
    const lastAt = value.lastIndexOf('@', cursor - 1);
    if (lastAt !== -1 && !value.slice(lastAt, cursor).includes(' ')) {
      const q = value.slice(lastAt + 1, cursor).toLowerCase();
      setMentionQuery(q);
      setMentionCursorPos(lastAt);
      setShowMentionDropdown(true);
    } else {
      setShowMentionDropdown(false);
    }
  };

  const handleSelectMention = (nameOrRole: string) => {
    const before = inputText.slice(0, mentionCursorPos);
    const after = inputText.slice(mentionCursorPos + mentionQuery.length + 1);
    setInputText(`${before}@${nameOrRole} ${after}`);
    setShowMentionDropdown(false);
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim()) return;
    const text = inputText;
    setInputText('');
    setShowMentionDropdown(false);

    const sent = await runAction('Sending your message', async () => {
      await postJson('/api/chat/message', { sender: 'Admin', content: text, is_admin: true });
    });
    // A rejected message would otherwise vanish from the composer without a trace.
    if (!sent) setInputText(text);

    fetchState();
  };

  const handleStepTurn = async () => {
    setIsStepping(true);
    try {
      await runAction('Stepping the next model turn', async () => {
        await post('/api/chat/step');
      });
      await fetchState();
    } finally {
      setIsStepping(false);
    }
  };

  const handleEmergencyStop = async () => {
    await runAction('Emergency stop', async () => {
      await post('/api/chat/stop');
      setIsAutoTurn(false);
    });
    await fetchState();
  };

  const handleRefreshRoster = async () => {
    await runAction('Refreshing the turn roster', async () => {
      const data = await post<any>('/api/roster/refresh');
      setTurnSchedule(data.turn_schedule || []);
    });
  };

  const handleSaveRoster = async () => {
    const saved = await runAction('Saving the turn roster', async () => {
      await postJson('/api/roster/update', { schedule: editedRoster });
    });
    if (saved) setShowRosterModal(false);
    await fetchState();
  };

  const handleVoteOverride = async (voteId: string, action: 'approve' | 'reject') => {
    await runAction(`Vote override (${action})`, async () => {
      const data = await postJson<any>('/api/votes/override', { vote_id: voteId, action });
      // The vote can be recorded while the approved tool itself fails.
      if (data.executed === false && data.error) throw new Error(data.error);
    });
    fetchState();
  };

  const handleAnswerQuestion = async (questionId: string, answer: string) => {
    await runAction('Answer planning question', async () => {
      await postJson<any>('/api/plan/questions/answer', { question_id: questionId, answer });
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
    // No "pick a new moderator" prompt any more - the supervisor seat follows the Architect
    // role, so removing the Architect means adding another Architect-role model, not
    // choosing a replacement from whoever is left.
    await runAction('Removing the model from the room', async () => {
      await post(`/api/models/kick?model_id=${encodeURIComponent(modelId)}`);
    });
    fetchState();
  };

  const handleReaddModel = async (modelId: string) => {
    await runAction('Re-adding the model to the room', async () => {
      await post(`/api/models/readd?model_id=${encodeURIComponent(modelId)}`);
    });
    fetchState();
  };

  // handleSelectModerator / handleCloseModPrompt removed with the moderator picker.

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
      status: 'active'
    };

    const configured = await runAction('Adding the model', async () => {
      await postJson('/api/models/configure', payload);
    });
    if (configured) {
      setNewModelName('');
      setNewGgufPath('');
      setNewMmprojPath('');
      setNewApiKey('');
      setPathValidationMsg(null);
    }
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
      {/* --- STALE UI BANNER ---
          frontend/dist is a build artifact: editing frontend/src changes nothing in
          the browser until it is rebuilt, and a stale bundle is indistinguishable
          from a broken feature. The backend compares timestamps and says so. */}
      {frontendStatus?.is_stale && (
        <div className="shrink-0 z-[70] px-5 pt-2">
          <div className="flex items-center gap-2 bg-amber-950/80 border border-amber-700 text-amber-200 rounded-xl px-3 py-2 text-xs">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span className="flex-1">
              The interface source has changed since this bundle was built — what you see is out of date.
              {!frontendStatus.node_available && ' Node.js is not installed, so the app cannot rebuild itself here.'}
            </span>
            {frontendStatus.node_available && (
              <button
                onClick={async () => {
                  setIsRebuilding(true);
                  const ok = await runAction('Rebuilding the interface', async () => {
                    await post('/api/frontend/rebuild');
                  });
                  setIsRebuilding(false);
                  if (ok) window.location.reload();
                  else fetchState();
                }}
                disabled={isRebuilding}
                className="bg-amber-600 hover:bg-amber-500 disabled:opacity-60 text-white px-3 py-1 rounded-lg font-medium shrink-0"
              >
                {isRebuilding ? 'Rebuilding…' : 'Rebuild & Reload'}
              </button>
            )}
          </div>
        </div>
      )}

      {/* --- BACKEND FAILURE BANNERS --- */}
      {(apiError || connectionError) && (
        <div className="shrink-0 z-[70] space-y-1 px-5 pt-2">
          {apiError && (
            <div className="flex items-start gap-2 bg-rose-950/80 border border-rose-700 text-rose-200 rounded-xl px-3 py-2 text-xs">
              <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
              <span className="flex-1 break-words">{apiError}</span>
              <button onClick={() => setApiError(null)} className="text-rose-300 hover:text-rose-100" aria-label="Dismiss error">
                <X className="w-4 h-4" />
              </button>
            </div>
          )}
          {connectionError && (
            <div className="flex items-start gap-2 bg-amber-950/70 border border-amber-700 text-amber-200 rounded-xl px-3 py-2 text-xs">
              <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
              <span className="flex-1 break-words">{connectionError}</span>
            </div>
          )}
        </div>
      )}

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

          {/* Where the room is inside the pre-execution planning gate. "DISCUSSION PHASE"
              alone could not distinguish "waiting on the Critic" from "stuck". */}
          {phase === 'discussion' && (
            <div
              title="Architect proposes → Critic reviews → Programmer signs off → Architect opens Execution"
              className="flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium tracking-wide bg-slate-800/60 text-slate-300 border border-slate-700"
            >
              <span>{PLAN_STAGE_LABELS[planStage] || planStage}</span>
              {planStageOwnerName && (
                <span className="text-[10px] opacity-70">@{planStageOwnerName}</span>
              )}
              {planRevision > 0 && (
                <span className="text-[10px] opacity-50">rev {planRevision}</span>
              )}
            </div>
          )}

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

        {/* Hardware Status, Emergency Stop & Top Action Buttons */}
        <div className="flex items-center gap-3">
          {hardware && (
            <div className="flex items-center gap-3 text-xs bg-slate-800/60 px-3 py-1 rounded-lg border border-slate-700/50">
              <div className="flex items-center gap-1 text-slate-300">
                <Cpu className="w-3.5 h-3.5 text-slate-400" />
                <span>RAM: {hardware.ram_available_gb}GB</span>
              </div>
              <div className="h-3 w-px bg-slate-700" />
              <div className="flex items-center gap-1 text-slate-300" title={hardware.gpu_name || 'GPU / VRAM'}>
                <Zap className="w-3.5 h-3.5 text-cyan-400" />
                <span>VRAM: {hardware.vram_free_gb ?? 0} / {hardware.vram_total_gb ?? 0} GB</span>
              </div>
            </div>
          )}

          {/* EMERGENCY STOP BUTTON */}
          <button
            onClick={handleEmergencyStop}
            className="flex items-center gap-1.5 text-xs bg-rose-600 hover:bg-rose-500 text-white font-bold px-3 py-1.5 rounded-lg transition shadow-[0_0_12px_rgba(225,29,72,0.4)] animate-pulse"
            title="Emergency Stop all loops and tool executions immediately"
          >
            <X className="w-4 h-4 stroke-[3]" />
            <span>EMERGENCY STOP</span>
          </button>

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
          
          {/* PENDING ADMIN CHOICE. Deliberately phrased by what the person USING the
              software would see, never by implementation noun - a question whose options
              are engineering terms is a broken question, and the backend refuses to route
              one here. Nothing is blocked while this sits unanswered. */}
          {parkedQuestions.length > 0 && (
            <div className="bg-sky-500/10 border-b border-sky-500/30 px-5 py-3 text-xs text-sky-200">
              <div className="flex items-start gap-2 mb-2">
                <HelpCircle className="w-4 h-4 text-sky-300 mt-0.5 shrink-0" />
                <div>
                  <div className="font-semibold text-sky-100">
                    Your call{parkedQuestions.length > 1 ? ` (${parkedQuestions.length} waiting)` : ''}:
                  </div>
                  <div className="text-sky-200/90 mt-0.5">
                    {parkedQuestions[0].question_admin || parkedQuestions[0].text_internal}
                  </div>
                  {parkedQuestions[0].rationale && (
                    <div className="text-sky-300/60 mt-0.5">Why it matters: {parkedQuestions[0].rationale}</div>
                  )}
                </div>
              </div>
              <div className="flex flex-wrap gap-2 pl-6">
                {parkedQuestions[0].options.map((opt) => (
                  <button
                    key={opt.label}
                    onClick={() => handleAnswerQuestion(parkedQuestions[0].id, opt.label)}
                    title={opt.means}
                    className={`text-left px-3 py-1.5 rounded-lg border transition max-w-sm ${
                      opt.label === parkedQuestions[0].recommended
                        ? 'bg-sky-600/30 border-sky-400/50 hover:bg-sky-600/50'
                        : 'bg-slate-800/70 border-slate-700 hover:bg-slate-700/70'
                    }`}
                  >
                    <span className="font-medium text-sky-100">{opt.label}</span>
                    {opt.label === parkedQuestions[0].recommended && (
                      <span className="ml-1 text-[10px] text-sky-300">· suggested</span>
                    )}
                    {opt.means && opt.means !== opt.label && (
                      <span className="block text-[11px] text-slate-300/80 mt-0.5">{opt.means}</span>
                    )}
                  </button>
                ))}
              </div>
              <div className="pl-6 mt-2 text-[11px] text-sky-300/50">
                The room keeps working while you decide — it will use the suggestion if it has to move on.
              </div>
            </div>
          )}

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
          <div className="p-4 bg-slate-900/80 border-t border-slate-800 backdrop-blur relative">
            {/* @ Mention Autocomplete Dropdown */}
            {showMentionDropdown && (
              <div className="absolute bottom-16 left-4 bg-slate-900 border border-slate-800 rounded-xl shadow-2xl p-2 z-50 w-64 space-y-1">
                <div className="text-[10px] font-semibold text-slate-400 px-2 py-1 uppercase border-b border-slate-800">
                  Mention Model or Role
                </div>
                {Object.values(models)
                  .filter(m => m.name.toLowerCase().includes(mentionQuery) || m.role.toLowerCase().includes(mentionQuery))
                  .map(m => (
                    <button
                      key={m.id}
                      type="button"
                      onClick={() => handleSelectMention(m.name)}
                      className="w-full text-left px-2 py-1.5 rounded-lg hover:bg-slate-800 flex items-center justify-between text-xs text-slate-200 transition"
                    >
                      <span className="font-semibold text-emerald-400">@{m.name}</span>
                      <span className="text-[10px] text-slate-500">{m.role}</span>
                    </button>
                  ))}
              </div>
            )}

            {/* Roster Queue Dashboard & Auto Control Banner */}
            <div className="mb-2 flex items-center justify-between bg-slate-950/80 border border-slate-800 rounded-xl px-3 py-1.5 text-xs text-slate-300">
              <div className="flex items-center gap-2">
                <Users className="w-4 h-4 text-cyan-400" />
                <span className="font-semibold text-cyan-300">Roster Step:</span>
                <span className="bg-slate-900 border border-slate-800 px-2 py-0.5 rounded font-mono text-[11px] text-slate-200">
                  {turnSchedule.length > 0
                    ? `Next: @${models[turnSchedule[0]]?.name || turnSchedule[0]} (${turnSchedule.length} in queue)`
                    : 'Queue Empty (Moderator Alerted)'}
                </span>
                <button
                  onClick={() => {
                    setEditedRoster([...turnSchedule]);
                    setShowRosterModal(true);
                  }}
                  className="text-[10px] bg-slate-800 hover:bg-slate-700 text-cyan-300 px-2 py-0.5 rounded border border-slate-700 transition"
                >
                  Edit Roster
                </button>
                <button
                  onClick={handleRefreshRoster}
                  className="text-[10px] bg-slate-800 hover:bg-slate-700 text-emerald-400 px-2 py-0.5 rounded border border-slate-700 flex items-center gap-1 transition"
                  title="Refresh Roster Queue"
                >
                  <RefreshCw className="w-3 h-3" /> Refresh
                </button>
              </div>

              {/* Auto Mode Toggle */}
              <button
                onClick={() => setIsAutoTurn(!isAutoTurn)}
                className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold transition shadow ${
                  isAutoTurn
                    ? 'bg-emerald-600 text-white shadow-emerald-900/50 animate-pulse'
                    : 'bg-slate-800 text-slate-400 border border-slate-700 hover:text-slate-200'
                }`}
              >
                <Zap className="w-3.5 h-3.5" />
                <span>Auto: {isAutoTurn ? (loopActive ? 'ON (room busy)' : 'ON') : 'OFF'}</span>
              </button>
            </div>

            <form onSubmit={handleSendMessage} className="flex gap-2">
              <input
                type="text"
                value={inputText}
                onChange={handleInputChange}
                placeholder="Message the room or instruct models (Admin)... (type @ to mention)"
                className="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-emerald-500/50 text-slate-100 placeholder-slate-500 transition"
              />
              <button
                type="submit"
                className="bg-emerald-600 hover:bg-emerald-500 text-white px-4 py-2.5 rounded-xl font-medium text-sm flex items-center gap-1.5 transition shadow-sm"
              >
                <Send className="w-4 h-4" />
                <span>Send</span>
              </button>

              {/* Compact Next Turn Button */}
              <button
                type="button"
                onClick={handleStepTurn}
                disabled={isStepping}
                className="bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white px-3 py-2.5 rounded-xl font-medium text-xs flex items-center gap-1 transition shadow-sm shrink-0"
                title="Trigger Next Model Turn"
              >
                <Play className="w-3.5 h-3.5" />
                <span>{isStepping ? 'Thinking...' : 'Next Turn'}</span>
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
                const isUnloaded = st.status === 'offline' || !m.enabled;
                const tokPerSec = st.tok_per_sec || 0;

                // Status text & icon logic
                let statusBadgeText = 'Live / In Chat';
                let statusColorClass = 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.6)]';
                let statusBgBox = 'bg-emerald-950/40 border-emerald-800/40 text-emerald-400';

                if (isError) {
                  statusBadgeText = 'Disconnected / Error';
                  statusColorClass = 'bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.6)]';
                  statusBgBox = 'bg-rose-950/40 border-rose-800/40 text-rose-400';
                } else if (isUnloaded) {
                  statusBadgeText = 'Asleep / Unloaded';
                  statusColorClass = 'bg-slate-600';
                  statusBgBox = 'bg-slate-900 border-slate-800 text-slate-500';
                } else if (m.live_status && m.live_status !== 'Idle / Live in Chat') {
                  statusBadgeText = m.live_status;
                  statusColorClass = 'bg-cyan-400 animate-pulse';
                  statusBgBox = 'bg-cyan-950/40 border-cyan-800/40 text-cyan-300';
                }

                return (
                  <div
                    key={m.id}
                    className="bg-slate-950 border border-slate-800/80 rounded-xl p-3 flex flex-col gap-2 relative group"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        {/* Status Icon Indicator */}
                        <div
                          className={`w-2.5 h-2.5 rounded-full ${statusColorClass}`}
                          title={statusBadgeText}
                        />

                        <span className="font-semibold text-sm text-slate-100">{m.name}</span>

                        {isSupervisor(m) && (
                          <span className="flex items-center gap-1 bg-amber-500/10 text-amber-400 border border-amber-500/30 text-[10px] px-1.5 py-0.5 rounded-full">
                            <Crown className="w-3 h-3" /> Supervisor
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
                    <div className={`text-[11px] font-medium flex items-center gap-1.5 px-2 py-0.5 rounded-md border ${statusBgBox}`}>
                      <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${statusColorClass}`} />
                      <span className="truncate">{statusBadgeText}</span>
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
                      {/* The "Designate as Moderator" checkbox was removed: Architect and
                          Moderator are one seat, decided by the Role field above. */}
                      <span className="text-[11px] text-slate-500">
                        The Architect role is the supervisor seat.
                      </span>

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
                              {isSupervisor(m) && <span className="text-amber-400 text-[10px]">👑 Supervisor</span>}
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
                      const added = await runAction('Adding the model directory', async () => {
                        await postJson('/api/models/search_paths', { path: customSearchPathInput });
                      });
                      if (added) setCustomSearchPathInput('');
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

      {/* The "Select Replacement Moderator" modal was removed along with the moderator flag.
          The supervisor seat is whichever model holds the Architect role, so there is nothing
          to pick: to change supervisor, change a model's role. If the room is left with no
          Architect at all, the backend posts a [NO ARCHITECT] notice into chat. */}

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
                  {/* Project Switcher — a project owns its tasks, shared memory and bot workspaces */}
                  <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-3">
                    <h4 className="font-semibold text-xs text-violet-400 flex items-center gap-2">
                      <FolderOpen className="w-4 h-4" /> Project
                    </h4>
                    <div className="flex flex-wrap items-center gap-2 text-xs">
                      <select
                        value={activeProjectId}
                        onChange={async (e) => {
                          const target = e.target.value;
                          if (target === activeProjectId) return;
                          // Blank the room first: a project owns its chat, its question board
                          // and its tasks, and none of it should survive the switch on screen.
                          clearRoomView();
                          await runAction('Switching project', async () => {
                            await postJson('/api/projects/switch', { project_id: target });
                          });
                          fetchState();
                        }}
                        className="bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-slate-100 min-w-[220px]"
                      >
                        {projects.length === 0 && <option value={activeProjectId}>{activeProjectId}</option>}
                        {projects.map((p) => (
                          <option key={p.project_id} value={p.project_id}>
                            {p.project_id} — {p.open_task_count}/{p.task_count} open
                          </option>
                        ))}
                      </select>
                      <input
                        type="text"
                        value={newProjectName}
                        onChange={(e) => setNewProjectName(e.target.value)}
                        placeholder="New project name"
                        className="bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-slate-100 placeholder-slate-500"
                      />
                      <button
                        onClick={async () => {
                          if (!newProjectName.trim()) return;
                          const ok = await runAction('Creating the project', async () => {
                            await postJson('/api/projects/create', { project_id: newProjectName.trim() });
                          });
                          if (ok) setNewProjectName('');
                          fetchState();
                        }}
                        className="bg-violet-600 hover:bg-violet-500 text-white px-3 py-2 rounded-xl font-medium"
                      >
                        Create
                      </button>
                      <button
                        onClick={async () => {
                          // Selecting a project in the switcher IS switching to it, so the only
                          // project this button can ever target is the active one. The backend
                          // used to refuse exactly that, which made the button impossible to
                          // satisfy; it now switches the room to another project first.
                          const fallback = projects.find((p) => p.project_id !== activeProjectId)?.project_id;
                          if (!fallback) {
                            window.alert(`"${activeProjectId}" is your only project. Create another one first.`);
                            return;
                          }
                          if (!window.confirm(
                            `Archive project "${activeProjectId}"?\n\nIts memory, chat history and bot workspaces move to .swarmchat/trash, and the room switches to "${fallback}".`
                          )) return;
                          // Deleting the active project auto-switches the room server-side,
                          // so the same rule applies: clear before, not after.
                          clearRoomView();
                          await runAction('Deleting the project', async () => {
                            await postJson('/api/projects/delete', { project_id: activeProjectId });
                          });
                          fetchState();
                        }}
                        disabled={projects.length < 2}
                        title={projects.length < 2
                          ? 'Create a second project before deleting this one'
                          : 'Archive this project and switch the room to another one'}
                        className="bg-rose-950/80 hover:bg-rose-900 text-rose-300 border border-rose-900 px-3 py-2 rounded-xl font-medium disabled:opacity-40 disabled:cursor-not-allowed"
                      >
                        Delete Project
                      </button>
                      <button
                        onClick={async () => {
                          await runAction('Cleaning the workspaces', async () => {
                            await postJson('/api/workspace/clean', { max_age_days: 7, prune_orphans: true });
                          });
                          fetchState();
                        }}
                        className="bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 px-3 py-2 rounded-xl font-medium ml-auto"
                        title="Trash stale files (>7 days) and workspaces of models no longer in the room"
                      >
                        Clean Workspaces
                      </button>
                    </div>
                  </div>

                  {/* Create Task Form */}
                  <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-3">
                    <h4 className="font-semibold text-xs text-cyan-400 flex items-center gap-2">
                      <Plus className="w-4 h-4" /> Add Itinerary Task / Meeting Item
                    </h4>
                    <form
                      onSubmit={async (e) => {
                        e.preventDefault();
                        if (!newTaskTitle) return;
                        const created = await runAction('Adding the itinerary task', async () => {
                          await postJson('/api/itinerary/task', {
                            title: newTaskTitle,
                            description: newTaskDesc,
                            priority: newTaskPriority
                          });
                        });
                        if (created) {
                          setNewTaskTitle('');
                          setNewTaskDesc('');
                        }
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
                                  await runAction('Activating the task', async () => {
                                    await postJson('/api/itinerary/update', { task_id: task.id, status: 'in_progress' });
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
                                  await runAction('Completing the task', async () => {
                                    await postJson('/api/itinerary/update', { task_id: task.id, status: 'completed' });
                                  });
                                  fetchState();
                                }}
                                className="bg-emerald-950/80 hover:bg-emerald-900 text-emerald-300 border border-emerald-800 px-2.5 py-1 rounded font-medium"
                              >
                                Complete
                              </button>
                            )}
                            {/* Sort the task into another project */}
                            {projects.filter((p) => p.project_id !== activeProjectId).length > 0 && (
                              <select
                                value=""
                                onChange={async (e) => {
                                  const target = e.target.value;
                                  if (!target) return;
                                  await runAction('Moving the task', async () => {
                                    await postJson('/api/itinerary/move', { task_id: task.id, target_project_id: target });
                                  });
                                  fetchState();
                                }}
                                className="bg-slate-900 border border-slate-800 rounded px-2 py-1 text-slate-300"
                                title="Move this task to another project"
                              >
                                <option value="">Move to…</option>
                                {projects
                                  .filter((p) => p.project_id !== activeProjectId)
                                  .map((p) => (
                                    <option key={p.project_id} value={p.project_id}>{p.project_id}</option>
                                  ))}
                              </select>
                            )}
                            <button
                              onClick={async () => {
                                if (!window.confirm(`Delete task "${task.title}"? Any file it produced moves to .swarmchat/trash.`)) return;
                                await runAction('Deleting the task', async () => {
                                  await postJson('/api/itinerary/delete', { task_id: task.id, trash_artifacts: true });
                                });
                                fetchState();
                              }}
                              className="bg-rose-950/80 hover:bg-rose-900 text-rose-300 border border-rose-900 px-2 py-1 rounded font-medium flex items-center gap-1"
                              title="Delete this task"
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                            </button>
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
                            try {
                              const data = await apiRequest<any>(`/api/workspace/file?filepath=${encodeURIComponent(file.path)}`);
                              setSelectedFileContent(data.content || '');
                            } catch (err) {
                              setSelectedFileContent(`⚠️ Could not read ${file.path}: ${errorText(err)}`);
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

      {/* --- ADMIN ROSTER CUSTOMIZATION POPUP MODAL --- */}
      {showRosterModal && (
        <div className="fixed inset-0 bg-slate-950/85 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-md w-full p-6 shadow-xl space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="font-bold text-base text-cyan-400 flex items-center gap-2">
                <Users className="w-5 h-5" />
                <span>Admin Roster Customization</span>
              </h3>
              <button onClick={() => setShowRosterModal(false)} className="text-slate-400 hover:text-slate-200">
                <X className="w-5 h-5" />
              </button>
            </div>

            <p className="text-xs text-slate-300">
              Customize or reorder the speaker turn sequence for active models in the room:
            </p>

            <div className="space-y-2 max-h-60 overflow-y-auto pr-1">
              {editedRoster.map((mId, idx) => (
                <div key={idx} className="flex items-center justify-between bg-slate-950 border border-slate-800 p-2.5 rounded-xl text-xs text-slate-200">
                  <span className="font-mono text-cyan-300">#{idx + 1} @{models[mId]?.name || mId} ({models[mId]?.role || 'Participant'})</span>
                  <button
                    type="button"
                    onClick={() => setEditedRoster(editedRoster.filter((_, i) => i !== idx))}
                    className="text-rose-400 hover:text-rose-300 p-1 rounded"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
            </div>

            <div className="space-y-2 pt-2 border-t border-slate-800">
              <div className="text-xs font-semibold text-slate-400 uppercase">Add Participant to Turn Queue</div>
              <div className="flex gap-2">
                {Object.values(models).map((m) => (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => setEditedRoster([...editedRoster, m.id])}
                    className="bg-slate-800 hover:bg-slate-700 text-slate-200 px-2.5 py-1 rounded-lg text-xs font-medium border border-slate-700"
                  >
                    + {m.name}
                  </button>
                ))}
              </div>
            </div>

            <div className="pt-3 border-t border-slate-800 flex justify-between">
              <button
                type="button"
                onClick={handleRefreshRoster}
                className="bg-slate-800 hover:bg-slate-700 text-emerald-400 text-xs px-3 py-1.5 rounded-xl font-medium"
              >
                Auto-Generate Roster
              </button>
              <button
                type="button"
                onClick={handleSaveRoster}
                className="bg-cyan-600 hover:bg-cyan-500 text-white text-xs px-4 py-1.5 rounded-xl font-medium"
              >
                Save Roster
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

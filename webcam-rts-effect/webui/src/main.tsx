import React, { FormEvent, PointerEvent, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

type AnimationName = "none" | "pop" | "slide" | "bounce" | "spin";
type LoopName = "none" | "pulse" | "bob" | "shake";
type AssetKind = "image" | "audio" | "file";

type StickerLayer = {
  id: string;
  name: string;
  asset_path: string;
  x: number;
  y: number;
  scale: number;
  rotation: number;
  opacity: number;
  hidden: boolean;
  chroma_key_green: boolean;
  chroma_tolerance: number;
  enter_animation: AnimationName;
  loop_animation: LoopName;
  animation_speed: number;
};

type AudioTrack = {
  id: string;
  name: string;
  path: string;
  volume: number;
  loop: boolean;
  muted: boolean;
};

type EffectDefinition = {
  name: string;
  right_sticker: string;
  left_sticker: string | null;
  audio: string;
  audio_tracks: AudioTrack[];
  selected_audio: string;
  audio_volume: number;
  audio_loop: boolean;
  scale: number;
  right_x: number;
  right_y: number;
  left_x: number;
  left_y: number;
  layers: StickerLayer[];
  trigger_labels: string[];
  activate_threshold: number;
  deactivate_threshold: number;
};

type EffectLibrary = { selected_id: string; effects: Record<string, EffectDefinition> };
type AssetItem = { path: string; name: string; type: AssetKind; tag?: string };
type FieldError = { field: string; message: string };
type VersionItem = { created_at: string; effect: EffectDefinition };
type RuntimeStatus = { connected: boolean; label: string; confidence: number; fps: number; active_effect: string | null };

const defaultAssets: AssetItem[] = [
  { path: "assets/nick.gif", name: "nick.gif", type: "image" },
  { path: "assets/cat.gif", name: "cat.gif", type: "image" },
  { path: "assets/example.png", name: "example.png", type: "image" },
  { path: "assets/Kicau Mania Cutted.mp3", name: "Kicau Mania Cutted.mp3", type: "audio" },
];

function makeId(prefix: string): string {
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
}

function sourceFor(path: string | null): string {
  if (!path) return "";
  return path.startsWith("assets/") ? `/${path}` : path;
}

function isSound(path: string): boolean {
  return /\.(mp3|wav|m4a|ogg)$/i.test(path);
}

function clamp(value: number, min = 0, max = 1): number {
  return Math.max(min, Math.min(max, value));
}

function selectedEffect(library: EffectLibrary): EffectDefinition {
  return library.effects[library.selected_id];
}

function layerFromLegacy(id: string, name: string, path: string | null, x: number, y: number, scale: number, chroma = false): StickerLayer | null {
  if (!path) return null;
  return {
    id,
    name,
    asset_path: path,
    x,
    y,
    scale,
    rotation: 0,
    opacity: 1,
    hidden: false,
    chroma_key_green: chroma,
    chroma_tolerance: 80,
    enter_animation: "none",
    loop_animation: "none",
    animation_speed: 1,
  };
}

function normalizeTrack(track: string | Partial<AudioTrack>, index: number, fallbackVolume = 1, fallbackLoop = true): AudioTrack | null {
  if (typeof track === "string") {
    return { id: `track-${index + 1}`, name: track.split("/").pop() || `Track ${index + 1}`, path: track, volume: fallbackVolume, loop: fallbackLoop, muted: false };
  }
  if (!track.path) return null;
  return {
    id: track.id || `track-${index + 1}`,
    name: track.name || track.path.split("/").pop() || `Track ${index + 1}`,
    path: track.path,
    volume: Number(track.volume ?? fallbackVolume),
    loop: Boolean(track.loop ?? fallbackLoop),
    muted: Boolean(track.muted),
  };
}

function normalizeEffect(raw: Partial<EffectDefinition>): EffectDefinition {
  const scale = Number(raw.scale ?? 0.25);
  const layers = (raw.layers?.length ? raw.layers : [
    layerFromLegacy("right", "Gif 1", raw.right_sticker ?? "assets/nick.gif", Number(raw.right_x ?? 0.72), Number(raw.right_y ?? 0.12), scale),
    layerFromLegacy("left", "Gif 2", raw.left_sticker ?? "assets/cat.gif", Number(raw.left_x ?? 0.04), Number(raw.left_y ?? 0.12), scale, true),
  ].filter(Boolean)) as StickerLayer[];
  const rawTracks = raw.audio_tracks?.length ? raw.audio_tracks : [raw.selected_audio || raw.audio || "assets/Kicau Mania Cutted.mp3"];
  const tracks = rawTracks.map((track, index) => normalizeTrack(track as string | Partial<AudioTrack>, index, Number(raw.audio_volume ?? 1), Boolean(raw.audio_loop ?? true))).filter(Boolean) as AudioTrack[];
  const selectedAudio = raw.selected_audio || raw.audio || tracks[0]?.path || "";
  return {
    name: raw.name || "New effect",
    right_sticker: raw.right_sticker || layers[0]?.asset_path || "",
    left_sticker: raw.left_sticker ?? layers[1]?.asset_path ?? null,
    audio: raw.audio || selectedAudio,
    audio_tracks: tracks,
    selected_audio: selectedAudio,
    audio_volume: Number(raw.audio_volume ?? tracks[0]?.volume ?? 1),
    audio_loop: Boolean(raw.audio_loop ?? tracks[0]?.loop ?? true),
    scale,
    right_x: Number(raw.right_x ?? layers[0]?.x ?? 0.72),
    right_y: Number(raw.right_y ?? layers[0]?.y ?? 0.12),
    left_x: Number(raw.left_x ?? layers[1]?.x ?? 0.04),
    left_y: Number(raw.left_y ?? layers[1]?.y ?? 0.12),
    layers,
    trigger_labels: raw.trigger_labels ?? [],
    activate_threshold: Number(raw.activate_threshold ?? 0.7),
    deactivate_threshold: Number(raw.deactivate_threshold ?? 0.45),
  };
}

function normalizeLibrary(raw: EffectLibrary): EffectLibrary {
  const effects = Object.fromEntries(Object.entries(raw.effects).map(([id, effect]) => [id, normalizeEffect(effect)]));
  const selected_id = raw.selected_id in effects ? raw.selected_id : Object.keys(effects)[0];
  return { selected_id, effects };
}

function payloadFor(effect: EffectDefinition): EffectDefinition {
  const visibleLayers = effect.layers.filter((layer) => layer.asset_path);
  const first = visibleLayers[0];
  const second = visibleLayers[1];
  const selected = effect.audio_tracks.find((track) => track.path === effect.selected_audio) ?? effect.audio_tracks[0];
  return {
    ...effect,
    layers: visibleLayers,
    right_sticker: first?.asset_path || "",
    left_sticker: second?.asset_path || null,
    right_x: first?.x ?? effect.right_x,
    right_y: first?.y ?? effect.right_y,
    left_x: second?.x ?? effect.left_x,
    left_y: second?.y ?? effect.left_y,
    scale: first?.scale ?? effect.scale,
    selected_audio: selected?.path || "",
    audio: selected?.path || "",
    audio_volume: selected?.volume ?? effect.audio_volume,
    audio_loop: selected?.loop ?? effect.audio_loop,
  };
}

function presetEffect(kind: string): EffectDefinition {
  const base = normalizeEffect({ name: "New effect", right_sticker: "", left_sticker: null, audio: "", audio_tracks: [], layers: [] });
  if (kind === "two-sticker") return normalizeEffect({ ...base, name: "Two sticker meme", right_sticker: "assets/nick.gif", left_sticker: "assets/cat.gif" });
  if (kind === "song-only") return normalizeEffect({ ...base, name: "Song only", layers: [], audio_tracks: ["assets/Kicau Mania Cutted.mp3" as never] });
  if (kind === "fullscreen") return normalizeEffect({ ...base, name: "Fullscreen overlay", layers: [{ ...layerFromLegacy("overlay", "Overlay", "assets/example.png", 0, 0, 1, false)!, opacity: 0.86 }] });
  return base;
}

function App() {
  const [library, setLibrary] = useState<EffectLibrary | null>(null);
  const [savedLibrary, setSavedLibrary] = useState<EffectLibrary | null>(null);
  const [assets, setAssets] = useState<AssetItem[]>(defaultAssets);
  const [selectedLayerId, setSelectedLayerId] = useState("right");
  const [activeTab, setActiveTab] = useState<"layers" | "audio" | "trigger" | "assets">("layers");
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("Loading effects");
  const [errors, setErrors] = useState<FieldError[]>([]);
  const [activePreview, setActivePreview] = useState(false);
  const [cameraState, setCameraState] = useState<"off" | "starting" | "on" | "denied">("off");
  const [history, setHistory] = useState<EffectDefinition[]>([]);
  const [future, setFuture] = useState<EffectDefinition[]>([]);
  const [versions, setVersions] = useState<VersionItem[]>([]);
  const [runtimeStatus, setRuntimeStatus] = useState<RuntimeStatus | null>(null);
  const [importUrl, setImportUrl] = useState("");
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [youtubeName, setYoutubeName] = useState("youtube-audio.mp3");
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioRefs = useRef<HTMLAudioElement[]>([]);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const importInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    loadLibrary();
    loadAssets();
    const interval = window.setInterval(loadRuntimeStatus, 1500);
    return () => {
      window.clearInterval(interval);
      stopCamera();
      stopPreviewAudio();
    };
  }, []);

  const effect = library ? selectedEffect(library) : null;
  const selectedLayer = effect?.layers.find((layer) => layer.id === selectedLayerId) ?? effect?.layers[0] ?? null;
  const dirty = useMemo(() => JSON.stringify(library) !== JSON.stringify(savedLibrary), [library, savedLibrary]);
  const draftKey = library ? `webui-effect-draft:${library.selected_id}:assets/effect.json` : "";
  const filteredAssets = useMemo(() => assets.filter((asset) => asset.path.toLowerCase().includes(query.toLowerCase())), [assets, query]);

  useEffect(() => {
    if (!library || !dirty) return;
    localStorage.setItem(draftKey, JSON.stringify(selectedEffect(library)));
  }, [dirty, draftKey, library]);

  useEffect(() => {
    if (activePreview) startPreviewAudio();
    else stopPreviewAudio();
  }, [activePreview, effect?.audio_tracks, effect?.selected_audio]);

  useEffect(() => {
    function handleKey(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      if (target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) return;
      const saveCombo = (event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s";
      const undoCombo = (event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "z" && !event.shiftKey;
      const redoCombo = (event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "z" && event.shiftKey;
      if (saveCombo) { event.preventDefault(); saveEffect(); }
      if (undoCombo) { event.preventDefault(); undo(); }
      if (redoCombo) { event.preventDefault(); redo(); }
      if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(event.key)) {
        event.preventDefault();
        const delta = event.shiftKey ? 0.04 : event.altKey ? 0.002 : 0.01;
        nudgeLayer(event.key === "ArrowLeft" || event.key === "ArrowRight" ? "x" : "y", event.key === "ArrowLeft" || event.key === "ArrowUp" ? -delta : delta);
      }
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  });

  async function loadLibrary() {
    setStatus("Loading effects");
    setErrors([]);
    try {
      const response = await fetch("/api/effects");
      if (!response.ok) throw new Error("load failed");
      const loadedLibrary = normalizeLibrary(await response.json());
      const key = `webui-effect-draft:${loadedLibrary.selected_id}:assets/effect.json`;
      const draft = localStorage.getItem(key);
      const nextLibrary = draft
        ? { ...loadedLibrary, effects: { ...loadedLibrary.effects, [loadedLibrary.selected_id]: normalizeEffect(JSON.parse(draft)) } }
        : loadedLibrary;
      setLibrary(nextLibrary);
      setSavedLibrary(loadedLibrary);
      setSelectedLayerId(selectedEffect(nextLibrary).layers[0]?.id || "");
      setHistory([]);
      setFuture([]);
      setStatus(draft ? "Draft restored" : "");
      loadVersions(loadedLibrary.selected_id);
    } catch {
      setStatus("Could not load effects");
    }
  }

  async function loadAssets() {
    try {
      const response = await fetch("/api/assets");
      if (!response.ok) return;
      const payload = await response.json();
      setAssets([...defaultAssets, ...(payload.assets ?? [])].filter((asset, index, all) => all.findIndex((item) => item.path === asset.path) === index));
    } catch {
      setAssets(defaultAssets);
    }
  }

  async function loadVersions(effectId: string) {
    try {
      const response = await fetch(`/api/effects/${encodeURIComponent(effectId)}/versions`);
      if (response.ok) setVersions((await response.json()).versions ?? []);
    } catch {
      setVersions([]);
    }
  }

  async function loadRuntimeStatus() {
    try {
      const response = await fetch("/api/runtime/status");
      if (response.ok) setRuntimeStatus(await response.json());
    } catch {
      setRuntimeStatus({ connected: false, label: "manual", confidence: 0, fps: 0, active_effect: null });
    }
  }

  function commitEffect(nextEffect: EffectDefinition) {
    setLibrary((current) => {
      if (!current) return current;
      setHistory((items) => [...items.slice(-49), selectedEffect(current)]);
      setFuture([]);
      return { ...current, effects: { ...current.effects, [current.selected_id]: nextEffect } };
    });
  }

  function updateEffect(patch: Partial<EffectDefinition>) {
    if (!effect) return;
    commitEffect({ ...effect, ...patch });
  }

  function updateLayer(layerId: string, patch: Partial<StickerLayer>) {
    if (!effect) return;
    const layers = effect.layers.map((layer) => layer.id === layerId ? { ...layer, ...patch } : layer);
    commitEffect(payloadFor({ ...effect, layers }));
  }

  function updateTrack(trackId: string, patch: Partial<AudioTrack>) {
    if (!effect) return;
    const audio_tracks = effect.audio_tracks.map((track) => track.id === trackId ? { ...track, ...patch } : track);
    commitEffect(payloadFor({ ...effect, audio_tracks }));
  }

  function undo() {
    if (!effect || !history.length || !library) return;
    const previous = history[history.length - 1];
    setHistory(history.slice(0, -1));
    setFuture([effect, ...future]);
    setLibrary({ ...library, effects: { ...library.effects, [library.selected_id]: previous } });
  }

  function redo() {
    if (!future.length || !library || !effect) return;
    const next = future[0];
    setFuture(future.slice(1));
    setHistory([...history, effect]);
    setLibrary({ ...library, effects: { ...library.effects, [library.selected_id]: next } });
  }

  async function selectEffect(effectId: string) {
    stopPreviewAudio();
    const response = await fetch(`/api/select-effect/${encodeURIComponent(effectId)}`, { method: "PUT" });
    if (!response.ok) return;
    const nextLibrary = normalizeLibrary(await response.json());
    setLibrary(nextLibrary);
    setSavedLibrary(nextLibrary);
    setSelectedLayerId(selectedEffect(nextLibrary).layers[0]?.id || "");
    setHistory([]);
    setFuture([]);
    loadVersions(effectId);
  }

  async function createEffect(kind: string) {
    const preset = presetEffect(kind);
    const response = await fetch("/api/effects", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ name: preset.name, effect: preset }) });
    if (!response.ok) return;
    const nextLibrary = normalizeLibrary(await response.json());
    setLibrary(nextLibrary);
    setSavedLibrary(nextLibrary);
    setSelectedLayerId(selectedEffect(nextLibrary).layers[0]?.id || "");
    setStatus("Created effect");
  }

  async function duplicateEffect() {
    if (!library) return;
    const response = await fetch(`/api/effects/${encodeURIComponent(library.selected_id)}/duplicate`, { method: "POST" });
    if (!response.ok) return;
    const nextLibrary = normalizeLibrary(await response.json());
    setLibrary(nextLibrary);
    setSavedLibrary(nextLibrary);
    setStatus("Duplicated effect");
  }

  async function deleteEffect() {
    if (!library) return;
    const response = await fetch(`/api/effects/${encodeURIComponent(library.selected_id)}`, { method: "DELETE" });
    if (!response.ok) return;
    const nextLibrary = normalizeLibrary(await response.json());
    setLibrary(nextLibrary);
    setSavedLibrary(nextLibrary);
    setStatus("Deleted effect");
  }

  async function saveEffect(event?: FormEvent) {
    event?.preventDefault();
    if (!library || !effect) return;
    setStatus("Saving");
    setErrors([]);
    const response = await fetch(`/api/effects/${encodeURIComponent(library.selected_id)}`, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify(payloadFor(effect)) });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({ errors: [{ field: "form", message: "Save failed." }] }));
      setErrors(payload.errors ?? []);
      setStatus("Save failed");
      return;
    }
    const nextLibrary = normalizeLibrary(await response.json());
    setLibrary(nextLibrary);
    setSavedLibrary(nextLibrary);
    localStorage.removeItem(draftKey);
    setHistory([]);
    setFuture([]);
    setStatus("Saved");
    loadVersions(library.selected_id);
  }

  function discardDraft() {
    if (draftKey) localStorage.removeItem(draftKey);
    loadLibrary();
  }

  async function startCamera() {
    setCameraState("starting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setCameraState("on");
    } catch {
      setCameraState("denied");
      setStatus("Camera denied. Manual preview still works.");
    }
  }

  function stopCamera() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraState("off");
  }

  function startPreviewAudio() {
    if (!effect) return;
    stopPreviewAudio();
    audioRefs.current = effect.audio_tracks.filter((track) => !track.muted && track.path).map((track) => {
      const audio = new Audio(sourceFor(track.path));
      audio.volume = clamp(track.volume);
      audio.loop = track.loop;
      audio.play().catch(() => setStatus("Audio preview failed"));
      return audio;
    });
  }

  function stopPreviewAudio() {
    audioRefs.current.forEach((audio) => { audio.pause(); audio.currentTime = 0; });
    audioRefs.current = [];
  }

  function addLayer(assetPath = "assets/nick.gif") {
    if (!effect) return;
    const layer = layerFromLegacy(makeId("layer"), `Layer ${effect.layers.length + 1}`, assetPath, 0.38, 0.18, effect.scale)!
    const next = { ...effect, layers: [...effect.layers, layer] };
    setSelectedLayerId(layer.id);
    commitEffect(payloadFor(next));
  }

  function duplicateLayer(layer: StickerLayer) {
    if (!effect) return;
    const copy = { ...layer, id: makeId("layer"), name: `${layer.name} Copy`, x: clamp(layer.x + 0.04), y: clamp(layer.y + 0.04) };
    setSelectedLayerId(copy.id);
    commitEffect(payloadFor({ ...effect, layers: [...effect.layers, copy] }));
  }

  function removeLayer(layerId: string) {
    if (!effect) return;
    const layers = effect.layers.filter((layer) => layer.id !== layerId);
    setSelectedLayerId(layers[0]?.id || "");
    commitEffect(payloadFor({ ...effect, layers }));
  }

  function reorderLayer(layerId: string, delta: number) {
    if (!effect) return;
    const index = effect.layers.findIndex((layer) => layer.id === layerId);
    const target = index + delta;
    if (index < 0 || target < 0 || target >= effect.layers.length) return;
    const layers = [...effect.layers];
    [layers[index], layers[target]] = [layers[target], layers[index]];
    commitEffect(payloadFor({ ...effect, layers }));
  }

  function nudgeLayer(axis: "x" | "y", delta: number) {
    if (!selectedLayer) return;
    updateLayer(selectedLayer.id, { [axis]: clamp(selectedLayer[axis] + delta) });
  }

  function addTrack(path = "assets/Kicau Mania Cutted.mp3") {
    if (!effect) return;
    const track = normalizeTrack(path, effect.audio_tracks.length)!;
    commitEffect(payloadFor({ ...effect, audio_tracks: [...effect.audio_tracks, track], selected_audio: path }));
  }

  function removeTrack(trackId: string) {
    if (!effect) return;
    const audio_tracks = effect.audio_tracks.filter((track) => track.id !== trackId);
    commitEffect(payloadFor({ ...effect, audio_tracks, selected_audio: audio_tracks[0]?.path || "" }));
  }

  function reorderTrack(trackId: string, delta: number) {
    if (!effect) return;
    const index = effect.audio_tracks.findIndex((track) => track.id === trackId);
    const target = index + delta;
    if (index < 0 || target < 0 || target >= effect.audio_tracks.length) return;
    const audio_tracks = [...effect.audio_tracks];
    [audio_tracks[index], audio_tracks[target]] = [audio_tracks[target], audio_tracks[index]];
    commitEffect(payloadFor({ ...effect, audio_tracks }));
  }

  async function uploadFiles(files: FileList | File[]) {
    setStatus("Uploading asset");
    for (const file of Array.from(files)) {
      const data = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result));
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(file);
      });
      const response = await fetch("/api/assets", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ name: file.name, data }) });
      if (response.ok) assignAsset((await response.json()).path);
    }
    setStatus("Asset ready");
    loadAssets();
  }

  async function importRemoteAsset() {
    const response = await fetch("/api/assets/import-url", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ url: importUrl }) });
    if (!response.ok) { setStatus("Import failed"); return; }
    const payload = await response.json();
    assignAsset(payload.path);
    setImportUrl("");
    setStatus(`Imported ${payload.path}`);
    loadAssets();
  }

  async function importYoutube() {
    const response = await fetch("/api/assets/youtube-audio", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ url: youtubeUrl, name: youtubeName }) });
    if (!response.ok) { setStatus("yt-dlp missing or extraction failed"); return; }
    const payload = await response.json();
    addTrack(payload.path);
    setStatus(`Added ${payload.path}`);
    loadAssets();
  }

  async function renameAsset(asset: AssetItem, name: string, tag = asset.tag || "user") {
    const response = await fetch(`/api/assets/${encodeURIComponent(asset.name)}`, { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify({ name, tag }) });
    if (response.ok) loadAssets();
  }

  async function deleteAsset(asset: AssetItem) {
    const first = await fetch(`/api/assets/${encodeURIComponent(asset.name)}`, { method: "DELETE" });
    if (first.status === 409 && window.confirm("Asset is used by an effect. Delete anyway?")) {
      await fetch(`/api/assets/${encodeURIComponent(asset.name)}?confirm=true`, { method: "DELETE" });
    }
    loadAssets();
  }

  function assignAsset(path: string) {
    if (isSound(path)) addTrack(path);
    else if (selectedLayer) updateLayer(selectedLayer.id, { asset_path: path });
    else addLayer(path);
  }

  async function restoreVersion(index: number) {
    if (!library) return;
    const response = await fetch(`/api/effects/${encodeURIComponent(library.selected_id)}/restore`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ index }) });
    if (response.ok) {
      const nextLibrary = normalizeLibrary(await response.json());
      setLibrary(nextLibrary);
      setSavedLibrary(nextLibrary);
      setStatus("Version restored");
    }
  }

  async function importPack(files: FileList | null) {
    const file = files?.[0];
    if (!file) return;
    const response = await fetch("/api/effects/import", { method: "POST", body: await file.arrayBuffer() });
    if (response.ok) { setStatus("Pack imported"); loadLibrary(); loadAssets(); }
  }

  if (!library || !effect) return <main className="loading-screen"><div>{status}</div></main>;

  return (
    <main className="shell">
      <aside className="sidebar">
        <p className="eyebrow">Effect Library</p>
        <h1 className="brand">Effects</h1>
        <div className="effect-list">
          {Object.entries(library.effects).map(([effectId, item]) => (
            <button key={effectId} className="effect-item" aria-current={effectId === library.selected_id} onClick={() => selectEffect(effectId)}>
              <span>{item.name}</span><small>{effectId === library.selected_id ? (dirty ? "Unsaved" : "Selected") : "Open"}</small>
            </button>
          ))}
        </div>
        <div className="preset-grid">
          <button onClick={() => createEffect("blank")}>Blank</button>
          <button onClick={() => createEffect("two-sticker")}>Two GIF</button>
          <button onClick={() => createEffect("song-only")}>Song</button>
          <button onClick={() => createEffect("fullscreen")}>Overlay</button>
        </div>
        <div className="version-list">
          <p className="eyebrow">Versions</p>
          {versions.slice(-4).reverse().map((version, index) => <button key={version.created_at} onClick={() => restoreVersion(versions.length - 1 - index)}>{new Date(version.created_at).toLocaleString()}</button>)}
        </div>
      </aside>

      <section className="main-panel">
        <section className="workbench">
          <header className="topbar">
            <nav className="tabs" aria-label="Editor mode">
              {(["layers", "audio", "trigger", "assets"] as const).map((tab) => <button key={tab} className="tab" aria-selected={activeTab === tab} onClick={() => setActiveTab(tab)}>{tab}</button>)}
            </nav>
            <div className="save-row top-save">
              <span className={errors.length ? "status error" : "status"}>{errors[0]?.message || status || (dirty ? "Unsaved changes" : "All changes saved")}</span>
              <button onClick={discardDraft}>Discard</button>
              <button onClick={duplicateEffect}>Duplicate</button>
              <button onClick={deleteEffect}>Delete</button>
              <button className="primary" onClick={() => saveEffect()}>Save</button>
            </div>
          </header>

          <div className="stage-grid">
            <PreviewPane
              effect={effect}
              selectedLayerId={selectedLayerId}
              activePreview={activePreview}
              setActivePreview={setActivePreview}
              cameraState={cameraState}
              startCamera={startCamera}
              stopCamera={stopCamera}
              videoRef={videoRef}
              runtimeStatus={runtimeStatus}
              selectLayer={setSelectedLayerId}
              updateLayer={updateLayer}
              uploadFiles={uploadFiles}
            />

            <form className="properties" onSubmit={saveEffect}>
              <label className={fieldClass(errors, "name")}>Effect name<input value={effect.name} onChange={(event) => updateEffect({ name: event.target.value })} /></label>
              {activeTab === "layers" && <LayerPanel effect={effect} selectedLayer={selectedLayer} selectLayer={setSelectedLayerId} addLayer={addLayer} updateLayer={updateLayer} duplicateLayer={duplicateLayer} removeLayer={removeLayer} reorderLayer={reorderLayer} />}
              {activeTab === "audio" && <AudioPanel effect={effect} addTrack={addTrack} updateTrack={updateTrack} removeTrack={removeTrack} reorderTrack={reorderTrack} setActivePreview={setActivePreview} />}
              {activeTab === "trigger" && <TriggerPanel effect={effect} updateEffect={updateEffect} />}
              {activeTab === "assets" && <AssetTools importUrl={importUrl} setImportUrl={setImportUrl} importRemoteAsset={importRemoteAsset} youtubeUrl={youtubeUrl} setYoutubeUrl={setYoutubeUrl} youtubeName={youtubeName} setYoutubeName={setYoutubeName} importYoutube={importYoutube} fileInputRef={fileInputRef} importInputRef={importInputRef} uploadFiles={uploadFiles} importPack={importPack} />}
              <button className="primary" type="submit">Save effect</button>
            </form>
          </div>
        </section>

        <section className="asset-rail">
          <div className="rail-tools"><input className="search" placeholder="Search assets" value={query} onChange={(event) => setQuery(event.target.value)} /><button onClick={() => fileInputRef.current?.click()}>Upload</button><a className="button-link" href="/api/effects/export">Export pack</a></div>
          <div className="assets" onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); uploadFiles(event.dataTransfer.files); }}>
            {filteredAssets.map((asset) => <AssetTile key={asset.path} asset={asset} selected={asset.path === selectedLayer?.asset_path || effect.audio_tracks.some((track) => track.path === asset.path)} assignAsset={assignAsset} renameAsset={renameAsset} deleteAsset={deleteAsset} />)}
          </div>
        </section>
      </section>
    </main>
  );
}

function fieldClass(errors: FieldError[], field: string): string {
  return errors.some((error) => error.field === field) ? "field-error" : "";
}

type PreviewPaneProps = {
  effect: EffectDefinition;
  selectedLayerId: string;
  activePreview: boolean;
  setActivePreview: (active: boolean) => void;
  cameraState: string;
  startCamera: () => void;
  stopCamera: () => void;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  runtimeStatus: RuntimeStatus | null;
  selectLayer: (id: string) => void;
  updateLayer: (id: string, patch: Partial<StickerLayer>) => void;
  uploadFiles: (files: FileList | File[]) => void;
};

function PreviewPane({ effect, selectedLayerId, activePreview, setActivePreview, cameraState, startCamera, stopCamera, videoRef, runtimeStatus, selectLayer, updateLayer, uploadFiles }: PreviewPaneProps) {
  return (
    <div className="preview-stage" onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); uploadFiles(event.dataTransfer.files); }}>
      <div className="preview-toolbar">
        <button aria-pressed={cameraState === "on"} onClick={cameraState === "on" ? stopCamera : startCamera}>{cameraState === "on" ? "Stop camera" : "Start camera"}</button>
        <button aria-pressed={activePreview} onClick={() => setActivePreview(!activePreview)}>{activePreview ? "Inactive" : "Activate"}</button>
        <span>{cameraState === "denied" ? "Camera denied" : "Manual simulator"}</span>
      </div>
      <div className="safe-frame" data-active-preview={activePreview}>
        <video ref={videoRef} className="camera-feed" autoPlay playsInline muted />
        {!effect.layers.length ? <div className="empty-state">Drop a GIF or choose one below</div> : null}
        {effect.layers.map((layer) => <StickerPreview key={layer.id} layer={layer} activePreview={activePreview} selected={layer.id === selectedLayerId} selectLayer={selectLayer} updateLayer={updateLayer} />)}
        <div className="debug-overlay"><span>{runtimeStatus?.connected ? "Runtime" : "Manual"}</span><span>{runtimeStatus?.label ?? "manual"}</span><span>{Math.round((runtimeStatus?.confidence ?? 0) * 100)}%</span><span>{Math.round(runtimeStatus?.fps ?? 0)} fps</span></div>
      </div>
    </div>
  );
}

function StickerPreview({ layer, selected, activePreview, selectLayer, updateLayer }: { layer: StickerLayer; selected: boolean; activePreview: boolean; selectLayer: (id: string) => void; updateLayer: (id: string, patch: Partial<StickerLayer>) => void }) {
  if (!layer.asset_path || layer.hidden) return null;
  function moveSticker(event: PointerEvent<HTMLDivElement>) {
    selectLayer(layer.id);
    const node = event.currentTarget;
    const frame = node.parentElement?.getBoundingClientRect();
    if (!frame) return;
    node.setPointerCapture(event.pointerId);
    const update = (clientX: number, clientY: number) => updateLayer(layer.id, { x: clamp((clientX - frame.left) / frame.width), y: clamp((clientY - frame.top) / frame.height) });
    update(event.clientX, event.clientY);
    const onMove = (moveEvent: globalThis.PointerEvent) => update(moveEvent.clientX, moveEvent.clientY);
    const onUp = () => { node.removeEventListener("pointermove", onMove); node.removeEventListener("pointerup", onUp); };
    node.addEventListener("pointermove", onMove);
    node.addEventListener("pointerup", onUp);
  }
  return (
    <div className="sticker-preview" data-active={selected} data-sim-active={activePreview} data-enter={layer.enter_animation} data-loop={activePreview ? layer.loop_animation : "none"} style={{ "--node-x": layer.x, "--node-y": layer.y, "--scale": layer.scale, "--rotate": `${layer.rotation}deg`, opacity: activePreview ? layer.opacity : 0.45 } as React.CSSProperties} onPointerDown={moveSticker}>
      <img src={sourceFor(layer.asset_path)} alt={layer.name} draggable="false" onDragStart={(event) => event.preventDefault()} />
      <span>{layer.name}</span>
    </div>
  );
}

function LayerPanel({ effect, selectedLayer, selectLayer, addLayer, updateLayer, duplicateLayer, removeLayer, reorderLayer }: { effect: EffectDefinition; selectedLayer: StickerLayer | null; selectLayer: (id: string) => void; addLayer: () => void; updateLayer: (id: string, patch: Partial<StickerLayer>) => void; duplicateLayer: (layer: StickerLayer) => void; removeLayer: (id: string) => void; reorderLayer: (id: string, delta: number) => void }) {
  return <div className="panel-stack"><div className="layer-list">{effect.layers.map((layer) => <button key={layer.id} type="button" aria-pressed={selectedLayer?.id === layer.id} onClick={() => selectLayer(layer.id)}>{layer.hidden ? "Hidden" : layer.name}</button>)}<button type="button" onClick={() => addLayer()}>Add layer</button></div>{selectedLayer && <fieldset className="placement"><legend>Layer</legend><label>Name<input value={selectedLayer.name} onChange={(event) => updateLayer(selectedLayer.id, { name: event.target.value })} /></label><label>Asset<input value={selectedLayer.asset_path} onChange={(event) => updateLayer(selectedLayer.id, { asset_path: event.target.value })} /></label><div className="action-grid"><button type="button" onClick={() => reorderLayer(selectedLayer.id, -1)}>Up</button><button type="button" onClick={() => reorderLayer(selectedLayer.id, 1)}>Down</button><button type="button" onClick={() => duplicateLayer(selectedLayer)}>Copy</button><button type="button" onClick={() => removeLayer(selectedLayer.id)}>Delete</button></div><label className="toggle-row"><input type="checkbox" checked={selectedLayer.hidden} onChange={(event) => updateLayer(selectedLayer.id, { hidden: event.target.checked })} /> Hide layer</label><label>X <input type="number" min="0" max="1" step="0.01" value={selectedLayer.x.toFixed(2)} onChange={(event) => updateLayer(selectedLayer.id, { x: clamp(Number(event.target.value)) })} /></label><label>Y <input type="number" min="0" max="1" step="0.01" value={selectedLayer.y.toFixed(2)} onChange={(event) => updateLayer(selectedLayer.id, { y: clamp(Number(event.target.value)) })} /></label><label>Scale <strong>{selectedLayer.scale.toFixed(2)}</strong><input type="range" min="0.03" max="1.6" step="0.01" value={selectedLayer.scale} onChange={(event) => updateLayer(selectedLayer.id, { scale: Number(event.target.value) })} /></label><label>Rotation <input type="number" min="-180" max="180" step="1" value={selectedLayer.rotation} onChange={(event) => updateLayer(selectedLayer.id, { rotation: Number(event.target.value) })} /></label><label>Opacity <strong>{Math.round(selectedLayer.opacity * 100)}%</strong><input type="range" min="0" max="1" step="0.01" value={selectedLayer.opacity} onChange={(event) => updateLayer(selectedLayer.id, { opacity: Number(event.target.value) })} /></label><label className="toggle-row"><input type="checkbox" checked={selectedLayer.chroma_key_green} onChange={(event) => updateLayer(selectedLayer.id, { chroma_key_green: event.target.checked })} /> Green key</label><label>Chroma tolerance <input type="range" min="0" max="180" step="1" value={selectedLayer.chroma_tolerance} onChange={(event) => updateLayer(selectedLayer.id, { chroma_tolerance: Number(event.target.value) })} /></label><label>Entrance<select value={selectedLayer.enter_animation} onChange={(event) => updateLayer(selectedLayer.id, { enter_animation: event.target.value as AnimationName })}><option>none</option><option>pop</option><option>slide</option><option>bounce</option><option>spin</option></select></label><label>Loop motion<select value={selectedLayer.loop_animation} onChange={(event) => updateLayer(selectedLayer.id, { loop_animation: event.target.value as LoopName })}><option>none</option><option>pulse</option><option>bob</option><option>shake</option></select></label><label>Speed <input type="range" min="0.2" max="3" step="0.1" value={selectedLayer.animation_speed} onChange={(event) => updateLayer(selectedLayer.id, { animation_speed: Number(event.target.value) })} /></label></fieldset>}</div>;
}

function AudioPanel({ effect, addTrack, updateTrack, removeTrack, reorderTrack, setActivePreview }: { effect: EffectDefinition; addTrack: () => void; updateTrack: (id: string, patch: Partial<AudioTrack>) => void; removeTrack: (id: string) => void; reorderTrack: (id: string, delta: number) => void; setActivePreview: (active: boolean) => void }) {
  return <div className="panel-stack"><div className="action-grid"><button type="button" onClick={() => addTrack()}>Add track</button><button type="button" onClick={() => setActivePreview(true)}>Play mix</button><button type="button" onClick={() => setActivePreview(false)}>Stop mix</button></div>{effect.audio_tracks.map((track) => <fieldset className="placement" key={track.id}><legend>{track.name}</legend><label>Name<input value={track.name} onChange={(event) => updateTrack(track.id, { name: event.target.value })} /></label><label>Path<input value={track.path} onChange={(event) => updateTrack(track.id, { path: event.target.value })} /></label><label>Volume <strong>{Math.round(track.volume * 100)}%</strong><input type="range" min="0" max="1" step="0.01" value={track.volume} onChange={(event) => updateTrack(track.id, { volume: Number(event.target.value) })} /></label><label className="toggle-row"><input type="checkbox" checked={track.loop} onChange={(event) => updateTrack(track.id, { loop: event.target.checked })} /> Loop</label><label className="toggle-row"><input type="checkbox" checked={track.muted} onChange={(event) => updateTrack(track.id, { muted: event.target.checked })} /> Mute</label><div className="action-grid"><button type="button" onClick={() => reorderTrack(track.id, -1)}>Up</button><button type="button" onClick={() => reorderTrack(track.id, 1)}>Down</button><button type="button" onClick={() => updateTrack(track.id, { muted: !track.muted })}>Toggle</button><button type="button" onClick={() => removeTrack(track.id)}>Remove</button></div></fieldset>)}</div>;
}

function TriggerPanel({ effect, updateEffect }: { effect: EffectDefinition; updateEffect: (patch: Partial<EffectDefinition>) => void }) {
  return <div className="panel-stack"><label>Trigger labels<input value={effect.trigger_labels.join(", ")} onChange={(event) => updateEffect({ trigger_labels: event.target.value.split(",").map((item) => item.trim()).filter(Boolean) })} /></label><label>Activate threshold <strong>{effect.activate_threshold.toFixed(2)}</strong><input type="range" min="0" max="1" step="0.01" value={effect.activate_threshold} onChange={(event) => updateEffect({ activate_threshold: Number(event.target.value) })} /></label><label>Deactivate threshold <strong>{effect.deactivate_threshold.toFixed(2)}</strong><input type="range" min="0" max="1" step="0.01" value={effect.deactivate_threshold} onChange={(event) => updateEffect({ deactivate_threshold: Number(event.target.value) })} /></label><div className="runtime-box">Manual trigger: {effect.trigger_labels[0] || "none"}</div></div>;
}

function AssetTools({ importUrl, setImportUrl, importRemoteAsset, youtubeUrl, setYoutubeUrl, youtubeName, setYoutubeName, importYoutube, fileInputRef, importInputRef, uploadFiles, importPack }: { importUrl: string; setImportUrl: (value: string) => void; importRemoteAsset: () => void; youtubeUrl: string; setYoutubeUrl: (value: string) => void; youtubeName: string; setYoutubeName: (value: string) => void; importYoutube: () => void; fileInputRef: React.RefObject<HTMLInputElement | null>; importInputRef: React.RefObject<HTMLInputElement | null>; uploadFiles: (files: FileList | File[]) => void; importPack: (files: FileList | null) => void }) {
  return <div className="panel-stack"><input ref={fileInputRef} className="hidden-file" type="file" multiple accept="image/*,.gif,audio/*" onChange={(event) => event.target.files && uploadFiles(event.target.files)} /><input ref={importInputRef} className="hidden-file" type="file" accept=".zip,application/zip" onChange={(event) => importPack(event.target.files)} /><div className="action-grid"><button type="button" onClick={() => fileInputRef.current?.click()}>Upload file</button><button type="button" onClick={() => importInputRef.current?.click()}>Import pack</button></div><label>Remote asset URL<input value={importUrl} onChange={(event) => setImportUrl(event.target.value)} /></label><button type="button" onClick={importRemoteAsset}>Import URL</button><label>YouTube URL<input value={youtubeUrl} onChange={(event) => setYoutubeUrl(event.target.value)} /></label><label>Output name<input value={youtubeName} onChange={(event) => setYoutubeName(event.target.value)} /></label><button type="button" onClick={importYoutube}>Extract audio</button></div>;
}

function AssetTile({ asset, selected, assignAsset, renameAsset, deleteAsset }: { asset: AssetItem; selected: boolean; assignAsset: (path: string) => void; renameAsset: (asset: AssetItem, name: string, tag: string) => void; deleteAsset: (asset: AssetItem) => void }) {
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(asset.name);
  const [tag, setTag] = useState(asset.tag || "user");
  return <article className="asset" data-selected={selected}><div className="asset-thumb">{asset.type === "audio" || isSound(asset.path) ? <span>Song</span> : <img src={sourceFor(asset.path)} alt="Sticker asset" draggable="false" onDragStart={(event) => event.preventDefault()} />}</div>{editing ? <div className="asset-edit"><input value={name} onChange={(event) => setName(event.target.value)} /><input value={tag} onChange={(event) => setTag(event.target.value)} /></div> : <p title={asset.path}>{asset.path.replace("assets/", "")} · {asset.tag}</p>}<div className="asset-actions"><button type="button" onClick={() => assignAsset(asset.path)}>{isSound(asset.path) ? "Add" : "Use"}</button>{asset.path.startsWith("assets/user/") && <button type="button" onClick={() => editing ? (renameAsset(asset, name, tag), setEditing(false)) : setEditing(true)}>{editing ? "Save" : "Rename"}</button>}{asset.path.startsWith("assets/user/") && <button type="button" onClick={() => deleteAsset(asset)}>Remove</button>}</div></article>;
}

createRoot(document.getElementById("root")!).render(<App />);

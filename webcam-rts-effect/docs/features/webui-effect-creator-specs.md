# WebUI Effect Creator Feature Specs

These specs target the local Vite React editor served by `main.py editor` / `main.py editor-api`. Current baseline already supports effect library CRUD, sticker/audio selection, file upload, audio preview, sticker placement, and JSON persistence through `assets/effect.json`.

## 1. Live Webcam Preview

Goal: Show real camera frames inside the editor preview canvas so effect placement matches runtime framing.

User: Local creator tuning effects before joining a call or recording.

Success criteria:
- User can start and stop a browser webcam preview from the editor.
- Stickers render over the live preview using the same normalized placement values as runtime.
- Camera permission denial keeps editor usable and shows recoverable status.
- No camera stream remains active after user stops preview or leaves the page.

Out of scope: Server-side webcam streaming, recording, or virtual camera output.

Data/API: Client-only `getUserMedia`; no effect JSON schema change.

Verification: Browser test mocks media stream; manual check confirms camera indicator turns off after stop.

## 2. Runtime Effect Simulator

Goal: Let users toggle an effect-active state in WebUI and preview sticker/audio behavior as runtime would show it.

User: Creator validating effect timing without starting the Python realtime app.

Success criteria:
- Preview has inactive and active modes.
- Active mode animates stickers and starts selected audio according to volume and loop settings.
- Inactive mode hides or dims effect layers and stops audio.
- Save payload remains unchanged unless user edits configuration.

Out of scope: Real pose detection in browser.

Data/API: Client state only.

Verification: Component tests assert audio start/stop calls and active preview state.

## 3. Multi-Sticker Layers

Goal: Replace fixed left/right sticker slots with ordered sticker layers.

User: Creator building effects with more than two visual assets.

Success criteria:
- User can add, rename, reorder, duplicate, hide, and delete sticker layers.
- Each layer has independent asset path, x/y position, scale, rotation, opacity, and chroma key setting.
- Existing `right_sticker` / `left_sticker` configs migrate into two layers on load.
- Runtime still supports old configs during transition.

Out of scope: Timeline animation and per-frame keyframes.

Data/API: Add `layers` array to `EffectDefinition`; preserve legacy fields for backward compatibility.

Verification: Unit tests cover migration, save/load round trip, and rendering order.

## 4. Layer Timeline Animation

Goal: Allow simple entrance and loop animations for each sticker layer.

User: Creator making effects feel less static during pose activation.

Success criteria:
- User can choose entrance animation: none, pop, slide, bounce, spin.
- User can choose loop motion: none, pulse, bob, shake.
- Preview plays animations without shifting editor layout.
- Runtime applies equivalent animation based on elapsed active time.

Out of scope: Full keyframe editor and bezier curve editing.

Data/API: Layer fields `enter_animation`, `loop_animation`, `animation_speed`.

Verification: Tests cover default values and deterministic runtime transform at known timestamps.

## 5. Chroma Key Controls

Goal: Expose green-screen removal controls for sticker assets.

User: Creator importing meme GIFs with green backgrounds.

Success criteria:
- User can enable chroma key per sticker layer.
- User can adjust tolerance with immediate preview update.
- Setting persists and runtime uses same tolerance.
- Default keeps existing left sticker green-key behavior compatible.

Out of scope: AI matting or arbitrary background segmentation.

Data/API: Layer fields `chroma_key_green`, `chroma_tolerance`.

Verification: Unit tests assert config parsing and `remove_green_screen` tolerance wiring.

## 6. Asset Library Manager

Goal: Provide a durable asset browser for uploaded and referenced files.

User: Creator managing many GIF, PNG, and audio files.

Success criteria:
- Asset rail lists files from `assets/` and `assets/user/`, not only assets referenced by effects.
- User can preview, rename, tag, and remove unused user assets.
- Deleting an asset referenced by an effect requires explicit confirmation.
- Asset paths remain relative to project root.

Out of scope: Cloud storage or external account sync.

Data/API: Add `GET /api/assets`, `PATCH /api/assets/{name}`, `DELETE /api/assets/{name}` with path traversal protection.

Verification: API tests cover listing, safe rename, blocked traversal, and referenced-delete guard.

## 7. URL Asset Importer

Goal: Download remote GIF/image/audio URLs into local assets from WebUI.

User: Creator pasting meme URLs instead of manually downloading files.

Success criteria:
- User can paste a URL and import it into `assets/user/`.
- UI shows download progress and final local path.
- Server validates scheme, size limit, extension/content type, and timeout.
- Imported asset becomes selectable immediately.

Out of scope: Search engine integration and YouTube extraction.

Data/API: `POST /api/assets/import-url` with `{ url }`; returns `{ path }`.

Verification: API tests mock successful download, oversize response, bad scheme, and timeout.

## 8. YouTube Audio Importer

Goal: Trigger existing YouTube-to-audio helper from editor UI.

User: Creator adding sound effects from a YouTube link.

Success criteria:
- User pastes YouTube URL and chooses output name.
- Server runs bounded `yt-dlp` extraction into `assets/user/`.
- UI reports missing `yt-dlp` clearly.
- Completed audio is added to selected effect tracks.

Out of scope: Playlist support and video download.

Data/API: `POST /api/assets/youtube-audio` with `{ url, name }`.

Verification: API tests mock subprocess success/failure and filename sanitization.

## 9. Audio Mixer

Goal: Support multiple audio tracks per effect with per-track settings.

User: Creator layering song, voice clip, and one-shot sounds.

Success criteria:
- User can add multiple tracks, reorder them, mute them, and set volume per track.
- Each track can loop or play once.
- Preview mixes tracks according to saved settings.
- Runtime starts/stops tracks consistently when effect activates/deactivates.

Out of scope: Waveform editing and beat matching.

Data/API: Replace flat `audio_tracks` with track objects while migrating legacy strings.

Verification: Tests cover migration, selected track behavior, and runtime audio start policy.

## 10. Trigger Mapping Per Effect

Goal: Assign each effect to one or more detector labels or manual triggers.

User: Creator building a library beyond only `kicau mania`.

Success criteria:
- Effect form includes trigger labels and confidence threshold.
- Runtime selects effect by detector label and threshold.
- Manual preview can simulate each trigger.
- Effects without trigger labels remain manually selectable only.

Out of scope: Training new labels from WebUI.

Data/API: Effect fields `trigger_labels`, `activate_threshold`, `deactivate_threshold`.

Verification: Runtime tests cover label-to-effect selection and fallback selected effect.

## 11. Effect Presets

Goal: Provide reusable starting templates for common effect styles.

User: Creator who wants fast setup without editing every field.

Success criteria:
- Create effect flow offers blank, two-sticker meme, song-only, and fullscreen overlay presets.
- Presets create complete valid effect definitions.
- Existing custom create flow remains available.
- Preset code does not duplicate default construction logic.

Out of scope: Downloadable preset marketplace.

Data/API: Client preset payloads sent to existing `POST /api/effects`, or server-side `preset` option.

Verification: Unit tests assert preset payload validity and unique effect IDs.

## 12. Effect Duplication And Versioning

Goal: Let users experiment safely by copying effects and tracking saved versions.

User: Creator tuning effects without losing known-good settings.

Success criteria:
- User can duplicate current effect with unique name and ID.
- Each save records a timestamped version snapshot.
- User can restore a prior version.
- Version history stays local and bounded.

Out of scope: Git integration and multi-user collaboration.

Data/API: `POST /api/effects/{id}/duplicate`, `GET /api/effects/{id}/versions`, `POST /api/effects/{id}/restore`.

Verification: API tests cover duplicate naming, version limit, restore payload.

## 13. Undo And Redo

Goal: Support local undo/redo for editor changes before save.

User: Creator making rapid layout and asset edits.

Success criteria:
- User can undo and redo effect edits with buttons and keyboard shortcuts.
- History resets after loading another effect or saving.
- Undo stack ignores transient status/loading state.
- Dirty indicator reflects current state versus last saved state.

Out of scope: Cross-session undo after page refresh.

Data/API: Client state only.

Verification: Component tests cover edit, undo, redo, save reset, effect switch reset.

## 14. Keyboard Shortcuts

Goal: Improve editor speed for common operations.

User: Creator placing stickers precisely.

Success criteria:
- Arrow keys nudge selected layer.
- Shift+arrow nudges by larger step; Alt+arrow nudges by smaller step.
- Cmd/Ctrl+S saves, Cmd/Ctrl+Z undo, Cmd/Ctrl+Shift+Z redo.
- Shortcuts do not hijack input fields.

Out of scope: User-configurable shortcut bindings.

Data/API: Client only.

Verification: Component tests cover shortcuts and ignored input focus.

## 15. Responsive Mobile Editor

Goal: Make WebUI usable on narrow screens for quick edits.

User: Creator opening local editor on small laptop windows or mobile browser.

Success criteria:
- Sidebar, workbench, properties, and asset rail stack without overlap below 768px.
- Touch dragging stickers works reliably.
- Buttons and inputs keep readable sizes without horizontal scrolling.
- Preview aspect ratio remains 16:9.

Out of scope: Native mobile app.

Data/API: CSS/layout only.

Verification: Playwright screenshots at 390px and 768px; no text overlap or horizontal page scroll.

## 16. Validation And Error Details

Goal: Prevent invalid effect configs from being saved silently.

User: Creator who needs clear correction prompts.

Success criteria:
- Server validates asset paths, numeric ranges, names, and audio settings.
- UI highlights fields with actionable errors.
- Invalid save does not modify effect JSON.
- Existing valid configs continue loading.

Out of scope: Formal JSON schema publishing.

Data/API: API returns `400` with `{ errors: [{ field, message }] }`.

Verification: API and UI tests cover invalid scale, missing name, bad asset path, and valid save.

## 17. Import And Export Effect Packs

Goal: Move effects and their assets between machines.

User: Creator sharing local effect bundles.

Success criteria:
- Export creates a zip containing effect JSON and referenced local assets.
- Import validates zip contents and adds effects without overwriting unless confirmed.
- Imported asset paths are remapped into `assets/user/`.
- UI shows summary before final import.

Out of scope: Remote sharing service.

Data/API: `GET /api/effects/export`, `POST /api/effects/import` multipart upload.

Verification: Integration tests cover export contents, import remap, blocked traversal in zip.

## 18. Pose Debug Overlay In Editor

Goal: Surface runtime detector/debug values in the editor when connected to local app status.

User: Creator tuning thresholds while testing webcam detection.

Success criteria:
- Editor can poll or subscribe to current label, confidence, FPS, and active effect.
- Overlay displays values on preview without blocking editing.
- Connection failure degrades to manual preview mode.
- Debug panel can be hidden.

Out of scope: Browser-based ML inference.

Data/API: `GET /api/runtime/status` or server-sent events endpoint.

Verification: API tests cover status payload; UI test covers disconnected state.

## 19. Auto-Save Drafts

Goal: Protect unsaved edits from browser refresh or accidental navigation.

User: Creator iterating quickly on effect settings.

Success criteria:
- Unsaved changes are stored locally as drafts.
- Refresh restores draft and clearly marks it unsaved.
- User can discard draft and reload server state.
- Drafts are scoped by effect ID and config path.

Out of scope: Server-side draft storage.

Data/API: Browser `localStorage` or IndexedDB only.

Verification: Component tests cover restore, discard, save clearing draft.

## 20. Visual Regression Preview Tests

Goal: Catch UI layout regressions in the effect creator.

User: Maintainer changing WebUI CSS or components.

Success criteria:
- Playwright opens editor with fixture effects.
- Tests capture desktop and mobile screenshots.
- Tests verify preview canvas, tabs, properties, asset rail, and save controls are visible.
- CI command is documented and does not require webcam permissions.

Out of scope: Pixel-perfect testing of GIF animation frames.

Data/API: Add test fixture JSON and mocked API or local editor server.

Verification: `npm run test:e2e` or documented equivalent passes locally.

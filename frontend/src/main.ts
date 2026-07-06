import "./style.css";

import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { VRButton } from "three/examples/jsm/webxr/VRButton.js";
import { XRButton } from "three/examples/jsm/webxr/XRButton.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";

type HandSide = "left" | "right";
type JointName = typeof JOINT_NAMES[number];
type BodyScale = number | [number, number, number];

type BodyInfoEntry = {
  file: string;
  is_fixed: boolean;
  name?: string;
  position: [number, number, number];
  quaternion: [number, number, number, number];
  scale?: BodyScale;
  opacity?: number;
};

type MocapInfoEntry = {
  mocap_id: number;
  body_id: number;
  name: string;
  position: [number, number, number];
  quaternion: [number, number, number, number];
};

type ClientConfig = {
  handSkeleton: boolean;
  handScale: number;
  scenePos: [number, number, number];
  vrPos: [number, number, number];
  vrTarget: [number, number, number];
  tableHeightOffset?: number;
};

type StatusResponse = {
  state: "delay" | "recording" | "idle" | "paused" | "playground";
  session_id?: string;
  frame_count: number;
  episode: number;
  tracking_active?: boolean;
  tracking_paused?: boolean;
  playground?: boolean;
  table_height_offset?: number;
  marker_count?: number;
  seconds_since_marker?: number | null;
  task?: {
    task_id?: string | number;
    title?: string;
    instruction?: string;
    difficulty?: string;
    task_index?: number;
    total_tasks?: number;
  };
  snapshots?: { task_id: string; snapshots: string[] } | null;
  mode?: "admin" | "collector" | "choreographer";
  awaiting_post_task_choice?: boolean;
  pedals?: Record<string, {
    held: boolean;
    held_seconds: number;
    long_press_seconds: number;
    press_count: number;
  }>;
  pedal_labels?: Record<string, string>;
};

const SHOT_CLOCK_SHAKE_SECONDS = 60;

const isTrackingState = (state: StatusResponse["state"]) => state === "recording" || state === "paused";

const finiteNumber = (value: number | null | undefined) =>
  typeof value === "number" && Number.isFinite(value) ? value : null;

const applyBodyScale = (obj: THREE.Object3D, scale?: BodyScale) => {
  if (Array.isArray(scale)) {
    obj.scale.set(scale[0] ?? 1.0, scale[1] ?? 1.0, scale[2] ?? 1.0);
    return;
  }
  obj.scale.setScalar(scale ?? 1.0);
};

const JENGA_BROWN_PALETTE = [
  new THREE.Color(0xc89a66),
  new THREE.Color(0xb98552),
  new THREE.Color(0x9f7044),
  new THREE.Color(0x865b36),
  new THREE.Color(0x6d492d),
];

const getJengaTint = (bodyName?: string) => {
  const match = /^jenga_([0-9]+)$/.exec(bodyName ?? "");
  if (!match) {
    return null;
  }
  const blockIndex = Number.parseInt(match[1], 10);
  return JENGA_BROWN_PALETTE[(blockIndex * 3 + Math.floor(blockIndex / 3)) % JENGA_BROWN_PALETTE.length];
};

type XRJointObject = THREE.Object3D & {
  visible: boolean;
  position: THREE.Vector3;
  quaternion: THREE.Quaternion;
};

type XRHandObject = THREE.Group & {
  joints: Record<string, XRJointObject>;
  userData: {
    spheres?: THREE.Mesh[];
    boneLines?: Array<{
      line: THREE.Line;
      geo: THREE.BufferGeometry;
      a: number;
      b: number;
    }>;
  };
};

declare global {
  interface Window {
    _gizmos?: Array<{ tc: TransformControls | null; target: THREE.Group }>;
    _mocapBodyIds?: Set<string>;
    _initMocapPoses?: Array<{
      id: number;
      pos: [number, number, number];
      quat: [number, number, number, number];
    }>;
    _handMocapMapping?: Record<HandSide, Partial<Record<JointName, number>>>;
  }
}

const JOINT_NAMES = [
  "wrist",
  "thumb-metacarpal",
  "thumb-phalanx-proximal",
  "thumb-phalanx-distal",
  "thumb-tip",
  "index-finger-metacarpal",
  "index-finger-phalanx-proximal",
  "index-finger-phalanx-intermediate",
  "index-finger-phalanx-distal",
  "index-finger-tip",
  "middle-finger-metacarpal",
  "middle-finger-phalanx-proximal",
  "middle-finger-phalanx-intermediate",
  "middle-finger-phalanx-distal",
  "middle-finger-tip",
  "ring-finger-metacarpal",
  "ring-finger-phalanx-proximal",
  "ring-finger-phalanx-intermediate",
  "ring-finger-phalanx-distal",
  "ring-finger-tip",
  "pinky-finger-metacarpal",
  "pinky-finger-phalanx-proximal",
  "pinky-finger-phalanx-intermediate",
  "pinky-finger-phalanx-distal",
  "pinky-finger-tip",
] as const;

const clientConfig = (await fetch("/api/client-config").then((resp) => resp.json())) as ClientConfig;

const info = document.getElementById("info") as HTMLDivElement;
const recDot = document.getElementById("rec-dot") as HTMLSpanElement;
const statusText = document.getElementById("status-text") as HTMLSpanElement;
const vrControls = document.getElementById("vr-controls") as HTMLDivElement;

const SKY_COLOR = 0x050816;

const scene = new THREE.Scene();
const skyBaseColor = new THREE.Color(SKY_COLOR);
// VR mode shows a dark sky + starfield + ground; AR mode hides all of them so
// passthrough shows through. Toggled in the AR sessionstart/sessionend
// handlers below.
scene.background = skyBaseColor;

const keyLight = new THREE.DirectionalLight(0xffffff, 1.15);
keyLight.position.set(2.5, 4.0, 3.0);
scene.add(keyLight);

// Hemisphere light = warm sky + cooler ground bounce. Lifts shadowed/back
// faces of mugs and plates off pitch-black without washing textures the way
// emissiveFactor did. Tunable: bump intensity if scenes still feel too dark.
const ambient = new THREE.HemisphereLight(0xffffff, 0xc0c0c8, 0.65);
scene.add(ambient);

// Group everything that should be hidden in AR so we can toggle it as a unit.
const vrScenery = new THREE.Group();
scene.add(vrScenery);

const createStarField = () => {
  const starCount = 650;
  const positions = new Float32Array(starCount * 3);
  const colors = new Float32Array(starCount * 3);
  let seed = 0x5eed1234;
  const rand = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 0xffffffff;
  };
  for (let i = 0; i < starCount; i += 1) {
    const azimuth = rand() * Math.PI * 2;
    const elevation = THREE.MathUtils.degToRad(8 + rand() * 74);
    const radius = 28 + rand() * 18;
    const base = i * 3;
    const cosElevation = Math.cos(elevation);
    positions[base] = Math.cos(azimuth) * cosElevation * radius;
    positions[base + 1] = Math.sin(elevation) * radius;
    positions[base + 2] = Math.sin(azimuth) * cosElevation * radius;
    const brightness = 0.24 + rand() * 0.38;
    colors[base] = brightness;
    colors[base + 1] = brightness * (0.90 + rand() * 0.10);
    colors[base + 2] = brightness * (0.95 + rand() * 0.12);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  const material = new THREE.PointsMaterial({
    size: 0.035,
    sizeAttenuation: true,
    vertexColors: true,
    transparent: true,
    opacity: 0.72,
    depthWrite: false,
  });
  return new THREE.Points(geometry, material);
};
vrScenery.add(createStarField());

const groundMesh = new THREE.Mesh(
  new THREE.PlaneGeometry(6, 6),
  new THREE.MeshLambertMaterial({ color: 0x000000 }),
);
groundMesh.rotation.x = -Math.PI / 2;
groundMesh.position.y = -0.001;
vrScenery.add(groundMesh);

// Two floor references with different scopes:
//   - `grid`: small (4m) high-contrast helper, useful on desktop to gauge
//     the camera target, but visually noisy in headset.
//   - `xrFloor`: a 15m-radius disc that reads as a calm "ground" surface in
//     either VR (against the dark sky) or AR (against passthrough).
// XR session toggles below swap which one is visible.
const grid = new THREE.GridHelper(4, 20, 0x444466, 0x333355);
scene.add(grid);

// One canvas tile = 1 m² of floor; UVs are scaled below so it tiles across
// the 30m disc.
const xrFloorCanvas = document.createElement("canvas");
xrFloorCanvas.width = 256;
xrFloorCanvas.height = 256;
{
  const ctx = xrFloorCanvas.getContext("2d")!;
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, 256, 256);
  ctx.strokeStyle = "#1f1f1f";
  ctx.lineWidth = 2;
  // Square outline = the meter line at the canvas edge, repeated by tiling.
  ctx.strokeRect(0, 0, 256, 256);
  ctx.strokeStyle = "#141414";
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i += 1) {
    const p = (256 * i) / 4;
    ctx.beginPath();
    ctx.moveTo(p, 0);
    ctx.lineTo(p, 256);
    ctx.moveTo(0, p);
    ctx.lineTo(256, p);
    ctx.stroke();
  }
}
const xrFloorTexture = new THREE.CanvasTexture(xrFloorCanvas);
xrFloorTexture.wrapS = THREE.RepeatWrapping;
xrFloorTexture.wrapT = THREE.RepeatWrapping;
xrFloorTexture.repeat.set(10, 10);  // 10 tiles across the 10m diameter disc
xrFloorTexture.anisotropy = 8;
const xrFloor = new THREE.Mesh(
  new THREE.CircleGeometry(5, 96),
  new THREE.MeshBasicMaterial({
    map: xrFloorTexture,
    side: THREE.DoubleSide,
  }),
);
xrFloor.rotation.x = -Math.PI / 2;
xrFloor.position.y = -0.0005;  // sit just below the MuJoCo z=0 plane
xrFloor.visible = false;
scene.add(xrFloor);

const camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.01, 100);
camera.position.set(0.5, 1.5, 1.5);
camera.lookAt(0, 0.8, 0);

// VR rig — positions the headset in the scene when entering VR/AR.
const vrPos = clientConfig.vrPos ?? [0, 0, 0];
const vrTarget = clientConfig.vrTarget ?? [0, 0.75, 0];
const vrRig = new THREE.Group();
vrRig.position.set(vrPos[0], vrPos[1], vrPos[2]);
// Yaw-only orientation computed from horizontal delta so vertical offsets never
// produce a degenerate lookAt (same-point → NaN rotation).
const dx = vrTarget[0] - vrPos[0];
const dz = vrTarget[2] - vrPos[2];
if (dx * dx + dz * dz > 1e-8) {
  vrRig.rotation.y = Math.atan2(dx, dz) + Math.PI;
}
scene.add(vrRig);
vrRig.add(camera);

const xrHud = new THREE.Group();
xrHud.visible = false;
scene.add(xrHud);

const xrHudOffset = new THREE.Vector3(-0.2, 0.12, -0.55);
const xrHudWorldPos = new THREE.Vector3();
const xrHudWorldQuat = new THREE.Quaternion();
const xrHudOffsetWorld = new THREE.Vector3();

const xrRecDot = new THREE.Mesh(
  new THREE.CircleGeometry(0.014, 24),
  new THREE.MeshBasicMaterial({
    color: 0xff3333,
    transparent: true,
    opacity: 0.95,
    depthTest: false,
    depthWrite: false,
  }),
);
xrRecDot.visible = false;
xrRecDot.renderOrder = 999;
xrHud.add(xrRecDot);

const xrRecRing = new THREE.Mesh(
  new THREE.RingGeometry(0.017, 0.02, 32),
  new THREE.MeshBasicMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.85,
    side: THREE.DoubleSide,
    depthTest: false,
    depthWrite: false,
  }),
);
xrRecRing.visible = false;
xrRecRing.renderOrder = 999;
xrHud.add(xrRecRing);

// Pedal HUD: three icons (A / B / C) at the bottom of the HUD, evenly spaced
// across the field of view. Each pedal is a canvas-textured plane: idle = dark
// pill with a white letter; held = solid blue; long-press = filled arc that
// grows from 0 to 1 over the long-press duration.
type PedalLetter = "A" | "B" | "C";
// fillLevel is the 0..1 amount of the long-press progress fill from the
// bottom: 0 = no fill, 1 = long-press complete. A short tap doesn't fill —
// it just flashes the body color blue for a moment.
type PedalState = { held: boolean; fillLevel: number };
const pedalLetters: PedalLetter[] = ["A", "B", "C"];
const pedalCanvases: Record<PedalLetter, HTMLCanvasElement> = {} as any;
const pedalTextures: Record<PedalLetter, THREE.CanvasTexture> = {} as any;
const pedalMeshes: Record<PedalLetter, THREE.Mesh> = {} as any;
const pedalLastDrawn: Record<PedalLetter, string> = { A: "", B: "", C: "" };
const pedalLastPressCount: Record<PedalLetter, number> = { A: 0, B: 0, C: 0 };
const pedalFlashUntilMs: Record<PedalLetter, number> = { A: 0, B: 0, C: 0 };
// Tap flash: a brand-new press latches the "held" state for this window so a
// tap that fell between polls still visibly lights the pedal blue. No fill is
// involved — only the body-color flash.
const PEDAL_FLASH_DURATION_MS = 500;
// `heldSeconds` is the value from the most recent poll; `heldSampleMs` is the
// wall clock time of that poll. The render loop extrapolates between polls so
// the fill animates smoothly at frame rate instead of stepping at 2Hz.
type PedalLogical = {
  heldNow: boolean;
  heldSeconds: number;
  heldSampleMs: number;
  longPress: number;
};
const pedalLogical: Record<PedalLetter, PedalLogical> = {
  A: { heldNow: false, heldSeconds: 0, heldSampleMs: 0, longPress: 0 },
  B: { heldNow: false, heldSeconds: 0, heldSampleMs: 0, longPress: 0 },
  C: { heldNow: false, heldSeconds: 0, heldSampleMs: 0, longPress: 0 },
};

const renderPedals = (now: number) => {
  for (const letter of pedalLetters) {
    const logical = pedalLogical[letter];
    // Pure 0..1 over the long-press window. Extrapolate from the last poll
    // sample using wall-clock delta so the fill animates smoothly between
    // polls (poll rate is only 2Hz).
    let fillLevel = 0;
    if (logical.heldNow && logical.longPress > 0) {
      const extrapolatedHeld =
        logical.heldSeconds + Math.max(0, (now - logical.heldSampleMs) / 1000);
      fillLevel = Math.max(0, Math.min(1, extrapolatedHeld / logical.longPress));
    }
    // A recent tap forces the "held" body color (solid blue) for the flash
    // window even after the key release, so quick taps still register
    // visually.
    const flashing = now < pedalFlashUntilMs[letter];
    updatePedalIcon(letter, { held: logical.heldNow || flashing, fillLevel });
  }
};

const drawPedalIcon = (
  ctx: CanvasRenderingContext2D,
  letter: string,
  state: PedalState,
  size: number,
  cornerRadius = 28,
) => {
  ctx.clearRect(0, 0, size, size);
  ctx.save();
  // Rounded-square pill body.
  const pad = 6;
  const x = pad;
  const y = pad;
  const w = size - 2 * pad;
  const h = size - 2 * pad;
  const r = cornerRadius;
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
  // Body color: solid blue when held, dark slate when idle.
  ctx.fillStyle = state.held ? "#3a8dff" : "#1c2433";
  ctx.fill();
  ctx.strokeStyle = state.held ? "#a7ccff" : "#3a4868";
  ctx.lineWidth = 4;
  ctx.stroke();

  // Bottom-rising fill: combined tap-flash + long-press progress in 0..1.
  if (state.fillLevel > 0) {
    const t = Math.max(0, Math.min(1, state.fillLevel));
    ctx.save();
    ctx.beginPath();
    const fillH = h * t;
    const fillY = y + h - fillH;
    ctx.rect(x, fillY, w, fillH);
    ctx.clip();
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
    ctx.fillStyle = "#ffc24d";
    ctx.fill();
    ctx.restore();
  }

  // Letter centered.
  ctx.fillStyle = "#ffffff";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = `700 ${Math.round(size * 0.5)}px Arial, Helvetica, sans-serif`;
  ctx.shadowColor = "rgba(0, 0, 0, 0.7)";
  ctx.shadowBlur = 4;
  ctx.fillText(letter, size / 2, size / 2 + size * 0.02);
  ctx.restore();
};

const updatePedalIcon = (letter: PedalLetter, state: PedalState) => {
  // Bucket the fill to 1% steps so the fill animation reads as smooth.
  // `held` is bucketed separately so the body color flips promptly.
  const bucket = `${state.held ? 1 : 0}:${Math.round(state.fillLevel * 100)}`;
  if (pedalLastDrawn[letter] === bucket) return;
  pedalLastDrawn[letter] = bucket;
  const ctx = pedalCanvases[letter].getContext("2d")!;
  drawPedalIcon(ctx, letter, state, PEDAL_CANVAS_SIZE);
  pedalTextures[letter].needsUpdate = true;
};

// Task prompt: world-space panel above the back of the table, tilted toward
// the user. Single mesh, no backplate so passthrough shows through in AR.
const taskPromptCanvas = document.createElement("canvas");
taskPromptCanvas.width = 1024;
taskPromptCanvas.height = 640;
const taskPromptTexture = new THREE.CanvasTexture(taskPromptCanvas);
// The panel is a single card — the canvas paints a translucent black
// background under white text so the prompt reads with high contrast while
// the workspace stays faintly visible behind it.
const taskPromptMaterial = new THREE.MeshBasicMaterial({
  map: taskPromptTexture,
  transparent: true,
  side: THREE.DoubleSide,
});
const worldTaskPrompt = new THREE.Mesh(
  new THREE.PlaneGeometry(1.0, 0.625),
  taskPromptMaterial,
);
// Far (default) pose: above the table, where the user reads at a glance.
// Panel sits in world space above the back of the table — glanceable while
// teleoperating, with a semi-translucent black background so the workspace
// shows through.
const TASK_PROMPT_FAR_POS = new THREE.Vector3(0.0, 1.15, -2.05);
const TASK_PROMPT_FAR_ROT_X = -THREE.MathUtils.degToRad(10);
const TASK_PROMPT_FAR_SCALE = 1.25;
worldTaskPrompt.position.copy(TASK_PROMPT_FAR_POS);
worldTaskPrompt.rotation.x = TASK_PROMPT_FAR_ROT_X;
worldTaskPrompt.scale.setScalar(TASK_PROMPT_FAR_SCALE);
worldTaskPrompt.renderOrder = 41;
worldTaskPrompt.visible = false;
scene.add(worldTaskPrompt);

// Loading panel: shown while the server is busy (rebuilding the scene,
// resetting, etc.). Same world-space style as the task prompt but smaller
// and centered so it doesn't fight with the prompt for screen real estate.
const loadingCanvas = document.createElement("canvas");
loadingCanvas.width = 768;
loadingCanvas.height = 256;
const loadingTexture = new THREE.CanvasTexture(loadingCanvas);
const loadingMaterial = new THREE.MeshBasicMaterial({
  map: loadingTexture,
  transparent: true,
  side: THREE.DoubleSide,
});
const worldLoadingPanel = new THREE.Mesh(
  new THREE.PlaneGeometry(0.6, 0.2),
  loadingMaterial,
);
worldLoadingPanel.position.set(0.0, 1.0, -1.8);
worldLoadingPanel.rotation.x = -THREE.MathUtils.degToRad(10);
worldLoadingPanel.renderOrder = 42;
worldLoadingPanel.visible = false;
scene.add(worldLoadingPanel);

let loadingActive = false;
let loadingLabel = "";
let loadingAnimationStart = 0;
let lastSimFrameAtMs = performance.now();
const STALE_FRAME_WARN_MS = 750;

const drawLoadingPanel = (phase: number) => {
  const ctx = loadingCanvas.getContext("2d")!;
  ctx.clearRect(0, 0, loadingCanvas.width, loadingCanvas.height);
  // Rounded translucent black card.
  ctx.fillStyle = "rgba(0, 0, 0, 0.82)";
  const r = 32;
  const w = loadingCanvas.width;
  const h = loadingCanvas.height;
  ctx.beginPath();
  ctx.moveTo(r, 0);
  ctx.lineTo(w - r, 0);
  ctx.quadraticCurveTo(w, 0, w, r);
  ctx.lineTo(w, h - r);
  ctx.quadraticCurveTo(w, h, w - r, h);
  ctx.lineTo(r, h);
  ctx.quadraticCurveTo(0, h, 0, h - r);
  ctx.lineTo(0, r);
  ctx.quadraticCurveTo(0, 0, r, 0);
  ctx.closePath();
  ctx.fill();
  // Animated dots after the label.
  const dotCount = (Math.floor(phase * 4) % 4);
  const dots = ".".repeat(dotCount);
  ctx.fillStyle = "#ffffff";
  ctx.textBaseline = "middle";
  ctx.textAlign = "center";
  ctx.font = "500 56px Arial, Helvetica, sans-serif";
  ctx.fillText(`${loadingLabel}${dots}`, w / 2, h / 2 - 18);
  // Progress bar (indeterminate sweeping segment).
  const barY = h - 56;
  const barX = 80;
  const barW = w - 160;
  const barH = 8;
  ctx.fillStyle = "rgba(255, 255, 255, 0.18)";
  ctx.fillRect(barX, barY, barW, barH);
  const segW = barW * 0.3;
  const t = (phase * 1.2) % 1.0;
  const segX = barX + (barW + segW) * t - segW;
  ctx.fillStyle = "#7ec8ff";
  ctx.fillRect(Math.max(barX, segX), barY, Math.min(segW, segX + segW - barX, barX + barW - Math.max(barX, segX)), barH);
  loadingTexture.needsUpdate = true;
};

// Collector slideshow pane: sibling of the task prompt, offset to the right.
// Always paints from the per-task choreographer snapshot URLs in a 1s/image
// loop. B toggles play/pause; in pause mode A/C scrub backward/forward.
const slideshowCanvas = document.createElement("canvas");
slideshowCanvas.width = 1024;
slideshowCanvas.height = 640;
const slideshowTexture = new THREE.CanvasTexture(slideshowCanvas);
const slideshowMaterial = new THREE.MeshBasicMaterial({
  map: slideshowTexture,
  transparent: true,
  side: THREE.DoubleSide,
});
const worldSlideshow = new THREE.Mesh(
  new THREE.PlaneGeometry(1.0, 0.625),
  slideshowMaterial,
);
const SLIDESHOW_FAR_POS = new THREE.Vector3(
  TASK_PROMPT_FAR_POS.x + 1.4 * TASK_PROMPT_FAR_SCALE,
  TASK_PROMPT_FAR_POS.y,
  TASK_PROMPT_FAR_POS.z,
);
worldSlideshow.position.copy(SLIDESHOW_FAR_POS);
worldSlideshow.rotation.x = TASK_PROMPT_FAR_ROT_X;
worldSlideshow.scale.setScalar(TASK_PROMPT_FAR_SCALE);
worldSlideshow.renderOrder = 41;
worldSlideshow.visible = false;
scene.add(worldSlideshow);

// Move the task prompt to the left of center to make room for the slideshow.
const TASK_PROMPT_FAR_POS_COLLECTOR = new THREE.Vector3(
  TASK_PROMPT_FAR_POS.x - 0.7 * TASK_PROMPT_FAR_SCALE,
  TASK_PROMPT_FAR_POS.y,
  TASK_PROMPT_FAR_POS.z,
);
const SLIDESHOW_FAR_POS_COLLECTOR = new THREE.Vector3(
  TASK_PROMPT_FAR_POS.x + 0.7 * TASK_PROMPT_FAR_SCALE,
  TASK_PROMPT_FAR_POS.y,
  TASK_PROMPT_FAR_POS.z,
);
// In two-pane mode the side panels toe inward toward the viewer for a
// surround-view feel: left panel yaws right, right panel yaws left.
const PANEL_SURROUND_YAW = THREE.MathUtils.degToRad(22);

const PEDAL_CANVAS_SIZE = 128;
// Pedal row: a fixed-position group anchored at the far task-prompt pose.
// Visible in XR regardless of whether the prompt is up, so the user always
// has a reference for what A/B/C do.
const PEDAL_PLANE_SIZE = 0.11;
const PEDAL_ROW_LOCAL_Y = -0.5 / 2 * TASK_PROMPT_FAR_SCALE - PEDAL_PLANE_SIZE / 2 - 0.02;
const PEDAL_ROW_SPACING = 0.34;
// Label pill below each pedal. Two lines (action on top, hold-action below)
// on a tall plane so each line gets a real font size — a single line on a
// short strip rendered the text far too small to read from VR distance.
const PEDAL_LABEL_CANVAS_W = 560;
const PEDAL_LABEL_CANVAS_H = 200;
const PEDAL_LABEL_PLANE_W = 0.32;
const PEDAL_LABEL_PLANE_H = 0.114;
const pedalGroup = new THREE.Group();
pedalGroup.position.copy(TASK_PROMPT_FAR_POS);
pedalGroup.rotation.x = TASK_PROMPT_FAR_ROT_X;
pedalGroup.position.y += PEDAL_ROW_LOCAL_Y;
pedalGroup.visible = false;
scene.add(pedalGroup);

const pedalLabelCanvases: Record<PedalLetter, HTMLCanvasElement> = {} as any;
const pedalLabelTextures: Record<PedalLetter, THREE.CanvasTexture> = {} as any;
const pedalLabelLastText: Record<PedalLetter, string> = { A: "", B: "", C: "" };

// Pick the largest font (<= maxPx) at which `text` fits `maxWidth`.
const fitFontPx = (
  ctx: CanvasRenderingContext2D,
  text: string,
  maxWidth: number,
  maxPx: number,
  minPx: number,
  weight = 600,
): number => {
  let px = maxPx;
  while (px > minPx) {
    ctx.font = `${weight} ${px}px Arial, Helvetica, sans-serif`;
    if (ctx.measureText(text).width <= maxWidth) break;
    px -= 2;
  }
  return px;
};

const drawPedalLabel = (letter: PedalLetter, text: string) => {
  if (pedalLabelLastText[letter] === text) return;
  pedalLabelLastText[letter] = text;
  const canvas = pedalLabelCanvases[letter];
  const ctx = canvas.getContext("2d")!;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!text) {
    pedalLabelTextures[letter].needsUpdate = true;
    return;
  }
  // Line 1 = tap action, line 2 = hold action (server splits on "\n").
  const [line1, line2 = ""] = text.split("\n");
  const pad = 22;
  const innerW = canvas.width - pad * 2;
  const r = 22;
  const w = canvas.width - 4;
  const h = canvas.height - 4;
  const x = 2;
  const y = 2;
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
  ctx.fillStyle = "rgba(15, 15, 18, 0.88)";
  ctx.fill();

  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  const cx = canvas.width / 2;
  if (line2) {
    // Two lines: tap action large+white on top, hold action dimmer below.
    const f1 = fitFontPx(ctx, line1, innerW, 72, 34, 700);
    ctx.font = `700 ${f1}px Arial, Helvetica, sans-serif`;
    ctx.fillStyle = "#ffffff";
    ctx.fillText(line1, cx, canvas.height * 0.32);
    const f2 = fitFontPx(ctx, line2, innerW, 52, 26, 500);
    ctx.font = `500 ${f2}px Arial, Helvetica, sans-serif`;
    ctx.fillStyle = "#bfc4cc";
    ctx.fillText(line2, cx, canvas.height * 0.70);
  } else {
    const f = fitFontPx(ctx, line1, innerW, 80, 34, 700);
    ctx.font = `700 ${f}px Arial, Helvetica, sans-serif`;
    ctx.fillStyle = "#ffffff";
    ctx.fillText(line1, cx, canvas.height / 2 + 2);
  }
  pedalLabelTextures[letter].needsUpdate = true;
};

for (let i = 0; i < pedalLetters.length; i++) {
  const letter = pedalLetters[i];
  const canvas = document.createElement("canvas");
  canvas.width = PEDAL_CANVAS_SIZE;
  canvas.height = PEDAL_CANVAS_SIZE;
  const texture = new THREE.CanvasTexture(canvas);
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(PEDAL_PLANE_SIZE, PEDAL_PLANE_SIZE),
    new THREE.MeshBasicMaterial({
      map: texture,
      transparent: true,
      depthTest: false,
      depthWrite: false,
    }),
  );
  const xOffset = (i - 1) * PEDAL_ROW_SPACING;
  mesh.position.set(xOffset, 0, 0);
  mesh.renderOrder = 42;
  pedalCanvases[letter] = canvas;
  pedalTextures[letter] = texture;
  pedalMeshes[letter] = mesh;
  pedalGroup.add(mesh);
  drawPedalIcon(
    canvas.getContext("2d")!,
    letter,
    { held: false, fillLevel: 0 },
    PEDAL_CANVAS_SIZE,
  );
  texture.needsUpdate = true;

  // Label pill below each pedal showing the current action.
  const labelCanvas = document.createElement("canvas");
  labelCanvas.width = PEDAL_LABEL_CANVAS_W;
  labelCanvas.height = PEDAL_LABEL_CANVAS_H;
  const labelTexture = new THREE.CanvasTexture(labelCanvas);
  const labelMesh = new THREE.Mesh(
    new THREE.PlaneGeometry(PEDAL_LABEL_PLANE_W, PEDAL_LABEL_PLANE_H),
    new THREE.MeshBasicMaterial({
      map: labelTexture,
      transparent: true,
      // Respect depth so scene objects in front of the label correctly
      // occlude it. Previously this drew on top of everything (depthTest=false
      // + renderOrder=42), which made the labels appear to "float over" the
      // sim view rather than sit in the world.
    }),
  );
  labelMesh.position.set(xOffset, -(PEDAL_PLANE_SIZE / 2 + PEDAL_LABEL_PLANE_H / 2 + 0.008), 0);
  pedalLabelCanvases[letter] = labelCanvas;
  pedalLabelTextures[letter] = labelTexture;
  pedalGroup.add(labelMesh);
}

// Banner: head-locked HUD copy for checkpoint warnings and rejection flashes.
const bannerCanvas = document.createElement("canvas");
bannerCanvas.width = 2048;
bannerCanvas.height = 280;
const bannerTexture = new THREE.CanvasTexture(bannerCanvas);
const bannerMaterial = new THREE.MeshBasicMaterial({
  map: bannerTexture,
  transparent: true,
  opacity: 0.94,
  side: THREE.DoubleSide,
  depthTest: false,
  depthWrite: false,
});
const xrBanner = new THREE.Mesh(new THREE.PlaneGeometry(0.5, 0.047), bannerMaterial);
xrBanner.position.set(0.2, -0.16, 0);
xrBanner.renderOrder = 1002;
xrBanner.visible = false;
xrHud.add(xrBanner);

let bannerLastKey = "";
let bannerTransientUntilMs = -Infinity;
const drawBanner = (text: string, color: string) => {
  const ctx = bannerCanvas.getContext("2d")!;
  ctx.clearRect(0, 0, bannerCanvas.width, bannerCanvas.height);
  ctx.fillStyle = "rgba(7, 9, 13, 0.92)";
  ctx.fillRect(0, 0, bannerCanvas.width, bannerCanvas.height);
  // Shrink font until the text fits with margin. Caps at 160px.
  const margin = 80;
  let fontSize = 160;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  while (fontSize > 60) {
    ctx.font = `${fontSize}px sans-serif`;
    if (ctx.measureText(text).width <= bannerCanvas.width - margin * 2) break;
    fontSize -= 4;
  }
  ctx.fillStyle = color;
  ctx.fillText(text, bannerCanvas.width / 2, bannerCanvas.height / 2);
  bannerTexture.needsUpdate = true;
};

// Per-frame: pick the persistent banner from current state. Skip if a transient
// (e.g. rejection) banner is still on screen.
const updatePersistentBanner = (
  now: number,
  shotClockSeconds: number,
  trackingActive: boolean,
) => {
  if (now < bannerTransientUntilMs) {
    return;
  }
  let text: string | null = null;
  let color = "#ff5555";
  // No playground banner: the per-pedal labels already show the controls, and
  // there are no checkpoints in playground so the shot-clock warning is moot.
  if (
    trackingActive &&
    !trackingPaused &&
    !playgroundMode &&
    shotClockSeconds >= SHOT_CLOCK_SHAKE_SECONDS
  ) {
    const seconds = Math.max(0, Math.floor(shotClockSeconds));
    text = `Checkpoint? ${seconds} since last`;
    color = "#ff7755";
  }
  const key = text ? `${text}|${color}` : "";
  if (key === bannerLastKey) {
    return;
  }
  bannerLastKey = key;
  if (text) {
    drawBanner(text, color);
    xrBanner.visible = true;
  } else {
    xrBanner.visible = false;
  }
};

// One-shot: flash a transient message for `durationMs`, then revert to whatever
// the persistent banner state would be on the next frame.
const flashBanner = (text: string, color: string, durationMs = 2000) => {
  drawBanner(text, color);
  xrBanner.visible = true;
  bannerTransientUntilMs = performance.now() + durationMs;
  bannerLastKey = `transient|${text}|${color}`;
};

const xrCheckCanvas = document.createElement("canvas");
xrCheckCanvas.width = 256;
xrCheckCanvas.height = 256;
const xrCheckTexture = new THREE.CanvasTexture(xrCheckCanvas);
const xrCheckMark = new THREE.Mesh(
  new THREE.PlaneGeometry(0.12, 0.12),
  new THREE.MeshBasicMaterial({
    map: xrCheckTexture,
    transparent: true,
    opacity: 0,
    depthTest: false,
    depthWrite: false,
  }),
);
xrCheckMark.position.set(0.11, -0.085, 0);
xrCheckMark.renderOrder = 1000;
xrCheckMark.visible = false;
xrHud.add(xrCheckMark);

const drawXRCheckMark = () => {
  const ctx = xrCheckCanvas.getContext("2d")!;
  ctx.clearRect(0, 0, xrCheckCanvas.width, xrCheckCanvas.height);
  ctx.save();
  ctx.shadowColor = "rgba(0, 0, 0, 0.35)";
  ctx.shadowBlur = 18;
  ctx.fillStyle = "rgba(13, 24, 28, 0.78)";
  ctx.beginPath();
  ctx.arc(128, 128, 88, 0, Math.PI * 2);
  ctx.fill();
  ctx.shadowBlur = 0;
  ctx.strokeStyle = "rgba(93, 224, 141, 0.95)";
  ctx.lineWidth = 12;
  ctx.beginPath();
  ctx.arc(128, 128, 74, -Math.PI * 0.8, Math.PI * 1.2);
  ctx.stroke();
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 18;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(80, 132);
  ctx.lineTo(114, 164);
  ctx.lineTo(178, 92);
  ctx.stroke();
  ctx.restore();
  xrCheckTexture.needsUpdate = true;
};

drawXRCheckMark();

let xrHudState: StatusResponse["state"] = "idle";
let trackingWasActive = false;
let trackingIsActive = false;
let trackingPaused = false;
let markerClockStartMs = performance.now();
let pauseStartedMs: number | null = null;
let markerSavedAtMs = -Infinity;
let lastMarkerCount = 0;
let lastTaskKey = "";
let currentTaskPrompt: TaskPromptContent = { header: "", bullets: [] };
let taskPromptStartedMs = -Infinity;
// Result of the just-finished task ("complete" | "skipped"), set by the
// task_finished broadcast and shown in the post-task panel header.
let lastTaskFinishResult: "complete" | "skipped" = "complete";
let prevStatusState: StatusResponse["state"] | null = null;
let playgroundMode = false;

type TaskPromptContent = {
  header: string;
  bullets: string[];
  // Footer text may contain {pedal:A|B|C} tokens that render as inline pedal
  // glyphs (the same rounded square the HUD uses, scaled to the line height).
  footer?: string;
};

// Draws an inline pedal-letter glyph at (x, y) with the given height. Returns
// the width advanced. Used inside footer text to embed (A)/(B)/(C) tokens.
const drawInlinePedalGlyph = (
  ctx: CanvasRenderingContext2D,
  letter: string,
  x: number,
  y: number,
  height: number,
) => {
  const w = height * 1.05;
  const r = height * 0.25;
  const top = y - height / 2;
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(x + r, top);
  ctx.lineTo(x + w - r, top);
  ctx.quadraticCurveTo(x + w, top, x + w, top + r);
  ctx.lineTo(x + w, top + height - r);
  ctx.quadraticCurveTo(x + w, top + height, x + w - r, top + height);
  ctx.lineTo(x + r, top + height);
  ctx.quadraticCurveTo(x, top + height, x, top + height - r);
  ctx.lineTo(x, top + r);
  ctx.quadraticCurveTo(x, top, x + r, top);
  ctx.closePath();
  ctx.fillStyle = "#3a8dff";
  ctx.fill();
  ctx.strokeStyle = "#a7ccff";
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.fillStyle = "#ffffff";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = `700 ${Math.round(height * 0.62)}px Arial, Helvetica, sans-serif`;
  ctx.fillText(letter, x + w / 2, y + height * 0.02);
  ctx.restore();
  return w;
};

const wrapToWidth = (ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string[] => {
  const words = text.split(/\s+/).filter(Boolean);
  const out: string[] = [];
  let line = "";
  for (const word of words) {
    const candidate = line ? `${line} ${word}` : word;
    if (ctx.measureText(candidate).width > maxWidth && line) {
      out.push(line);
      line = word;
    } else {
      line = candidate;
    }
  }
  if (line) {
    out.push(line);
  }
  return out;
};

const drawTaskPrompt = (content: TaskPromptContent) => {
  const ctx = taskPromptCanvas.getContext("2d")!;
  ctx.clearRect(0, 0, taskPromptCanvas.width, taskPromptCanvas.height);
  const { header, bullets, footer } = content;
  if (!header && bullets.length === 0) {
    taskPromptTexture.needsUpdate = true;
    return;
  }

  // Translucent black card with white text — keeps contrast high while
  // letting the workspace show through faintly.
  ctx.fillStyle = "rgba(0, 0, 0, 0.78)";
  ctx.fillRect(0, 0, taskPromptCanvas.width, taskPromptCanvas.height);

  ctx.save();
  ctx.fillStyle = "#ffffff";
  ctx.textBaseline = "middle";

  const headerFont = "600 68px Arial, Helvetica, sans-serif";
  const bulletFont = "400 52px Arial, Helvetica, sans-serif";
  const footerFont = "500 50px Arial, Helvetica, sans-serif";
  const headerLineHeight = 80;
  const bulletLineHeight = 76;
  const footerLineHeight = 68;
  const headerGap = 20;
  const footerGap = 28;
  const bulletIndent = 40;

  ctx.font = headerFont;
  const headerLines = header ? wrapToWidth(ctx, header, 940) : [];

  ctx.font = bulletFont;
  const wrappedBullets: string[][] = bullets.map((b) =>
    wrapToWidth(ctx, b, 940 - bulletIndent),
  );

  ctx.font = footerFont;
  const footerLines = footer ? wrapToWidth(ctx, footer, 940) : [];

  let totalHeight = headerLines.length * headerLineHeight;
  if (headerLines.length > 0 && wrappedBullets.length > 0) totalHeight += headerGap;
  for (const lines of wrappedBullets) {
    totalHeight += lines.length * bulletLineHeight;
  }
  if (footerLines.length > 0) {
    totalHeight += footerGap + footerLines.length * footerLineHeight;
  }

  let y = taskPromptCanvas.height / 2 - totalHeight / 2 + headerLineHeight / 2;

  ctx.font = headerFont;
  ctx.textAlign = "center";
  for (const line of headerLines) {
    ctx.fillText(line, taskPromptCanvas.width / 2, y);
    y += headerLineHeight;
  }
  if (headerLines.length > 0 && wrappedBullets.length > 0) {
    y += headerGap - headerLineHeight / 2 + bulletLineHeight / 2;
  } else {
    y += -headerLineHeight / 2 + bulletLineHeight / 2;
  }

  ctx.font = bulletFont;
  ctx.textAlign = "left";
  const leftMargin = 56;
  for (const lines of wrappedBullets) {
    for (let i = 0; i < lines.length; i++) {
      const prefix = i === 0 ? "• " : "  ";
      ctx.fillText(prefix + lines[i], leftMargin, y);
      y += bulletLineHeight;
    }
  }

  if (footerLines.length > 0) {
    y += footerGap - bulletLineHeight / 2 + footerLineHeight / 2;
    ctx.font = footerFont;
    ctx.fillStyle = "#ffc24d";
    const glyphH = 44;
    for (const line of footerLines) {
      // Tokenize on {pedal:X} so we can compute a precise width for centering.
      const parts = line.split(/(\{pedal:[ABC]\})/g).filter(Boolean);
      let totalWidth = 0;
      const widths: number[] = [];
      for (const part of parts) {
        const match = part.match(/^\{pedal:([ABC])\}$/);
        if (match) {
          const w = glyphH * 1.05 + 6;  // glyph width + small gap
          widths.push(w);
          totalWidth += w;
        } else {
          ctx.font = footerFont;
          const w = ctx.measureText(part).width;
          widths.push(w);
          totalWidth += w;
        }
      }
      let cursor = (taskPromptCanvas.width - totalWidth) / 2;
      for (let i = 0; i < parts.length; i++) {
        const part = parts[i];
        const match = part.match(/^\{pedal:([ABC])\}$/);
        if (match) {
          drawInlinePedalGlyph(ctx, match[1], cursor, y, glyphH);
        } else {
          ctx.font = footerFont;
          ctx.fillStyle = "#ffc24d";
          ctx.textAlign = "left";
          ctx.textBaseline = "middle";
          ctx.fillText(part, cursor, y);
        }
        cursor += widths[i];
      }
      y += footerLineHeight;
    }
  }
  ctx.restore();
  taskPromptTexture.needsUpdate = true;
};

const updateTaskPrompt = () => {
  const hasTask =
    (currentTaskPrompt.header.length > 0 || currentTaskPrompt.bullets.length > 0) &&
    !playgroundMode;
  worldTaskPrompt.visible = hasTask;
};

const emptyTaskPrompt = (): TaskPromptContent => ({ header: "", bullets: [] });

const drawContain = (
  ctx: CanvasRenderingContext2D,
  media: HTMLImageElement,
  x: number, y: number, w: number, h: number,
  srcW: number, srcH: number,
): void => {
  const aspect = srcW / srcH;
  let dW = w, dH = h;
  if (aspect > w / h) dH = w / aspect; else dW = h * aspect;
  ctx.drawImage(media, x + (w - dW) / 2, y + (h - dH) / 2, dW, dH);
};

// --- Slideshow state (collector + choreographer) ---
// The slideshow always auto-plays; it is not operator-controlled.
let slideshowUrls: string[] = [];
let slideshowIdx = 0;
let slideshowLastAdvanceMs = 0;
let slideshowTaskKey = "";
const SLIDESHOW_FRAME_MS = 1000;
const SLIDESHOW_LAST_FRAME_MS = 5000;
const slideshowImageCache: Map<string, HTMLImageElement> = new Map();

const loadSlideshowImage = (url: string): HTMLImageElement => {
  const cached = slideshowImageCache.get(url);
  if (cached) return cached;
  const img = new Image();
  // Async load: redraw once the bytes arrive so the first frame isn't blank
  // while we wait for the slideshow to advance.
  img.onload = () => {
    if (
      slideshowUrls.length > 0
      && slideshowUrls[slideshowIdx % slideshowUrls.length] === url
    ) {
      drawSlideshow();
    }
  };
  img.src = url;
  slideshowImageCache.set(url, img);
  return img;
};

const drawSlideshow = () => {
  const ctx = slideshowCanvas.getContext("2d")!;
  const W = slideshowCanvas.width;
  const H = slideshowCanvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "rgba(0, 0, 0, 0.78)";
  ctx.fillRect(0, 0, W, H);
  const PAD = 16;
  if (slideshowUrls.length === 0) {
    ctx.fillStyle = "#bdbdbd";
    ctx.font = "500 36px Arial, Helvetica, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("No slideshow available", W / 2, H / 2);
    slideshowTexture.needsUpdate = true;
    return;
  }
  const url = slideshowUrls[slideshowIdx % slideshowUrls.length];
  const img = loadSlideshowImage(url);
  if (img.complete && img.naturalWidth > 0) {
    drawContain(ctx, img, PAD, PAD, W - PAD * 2, H - PAD * 2 - 36,
      img.naturalWidth, img.naturalHeight);
  }
  slideshowTexture.needsUpdate = true;
};

const setSlideshowUrls = (taskKey: string, urls: string[]) => {
  // Key on task + URL list so newly captured choreographer snapshots (same
  // task, longer list) refresh the pane without a task switch.
  const key = `${taskKey}:${urls.join("|")}`;
  if (slideshowTaskKey === key) {
    return;
  }
  const taskChanged = slideshowTaskKey.split(":")[0] !== taskKey.split(":")[0];
  slideshowTaskKey = key;
  slideshowUrls = urls.slice();
  if (taskChanged) {
    slideshowIdx = 0;
  } else {
    slideshowIdx = slideshowUrls.length > 0
      ? Math.min(slideshowIdx, slideshowUrls.length - 1)
      : 0;
  }
  slideshowLastAdvanceMs = performance.now();
  // Pre-warm the cache so the first frames don't flicker.
  for (const url of urls) loadSlideshowImage(url);
  drawSlideshow();
};

const tickSlideshow = (now: number) => {
  if (slideshowUrls.length === 0) return;
  // Hold the last slide longer before looping back to the first.
  const isLast = slideshowIdx % slideshowUrls.length === slideshowUrls.length - 1;
  const dwell = isLast ? SLIDESHOW_LAST_FRAME_MS : SLIDESHOW_FRAME_MS;
  if (now - slideshowLastAdvanceMs >= dwell) {
    slideshowIdx = (slideshowIdx + 1) % slideshowUrls.length;
    slideshowLastAdvanceMs = now;
    drawSlideshow();
  }
};

const setTaskPromptFromStatus = (
  task: StatusResponse["task"],
  now: number,
  opts: { postTask?: boolean } = {},
) => {
  const postTask = Boolean(opts.postTask);
  // Post-task state in collector mode: replace the task pane with the
  // restart/new-task prompt (inline, no banner).
  if (postTask) {
    const lastKey =
      `__post_task__:${task?.task_id ?? ""}:${task?.task_index ?? ""}:${lastTaskFinishResult}`;
    if (lastTaskKey !== lastKey) {
      lastTaskKey = lastKey;
      currentTaskPrompt = {
        header: lastTaskFinishResult === "skipped"
          ? "Task Skipped"
          : "Task Complete!",
        bullets: [
          "{pedal:A}  Restart same task",
          "{pedal:C}  Move to a new task",
        ],
      };
      taskPromptStartedMs = now;
      drawTaskPrompt(currentTaskPrompt);
    }
    return;
  }
  // --- Task instruction text. ---
  const instruction = task?.instruction?.trim() ?? "";
  const title = task?.title?.trim() ?? "";
  const taskKey = instruction || title
    ? `${task?.task_index ?? ""}:${task?.task_id ?? ""}:${title}:${instruction}`
    : "";
  if (taskKey === lastTaskKey) {
    return;
  }
  const isFirstTask = lastTaskKey === "";
  lastTaskKey = taskKey;
  if (!instruction && !title) {
    currentTaskPrompt = emptyTaskPrompt();
    taskPromptStartedMs = -Infinity;
    drawTaskPrompt(currentTaskPrompt);
    return;
  }
  const indexText =
    typeof task?.task_index === "number" && typeof task?.total_tasks === "number"
      ? `Task ${task.task_index + 1}/${task.total_tasks}: `
      : "Task: ";
  const headerBase = title || instruction.split("\n")[0];
  const header = `${indexText}${headerBase}`;
  const bulletSource = instruction;
  const bullets = bulletSource
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .map((line) => line.replace(/^[-•*]\s*/, ""))
    .filter((line) => !title || line !== title);
  // On a real task transition (not the first poll after page load), surface
  // the "Tap (B) to start" hint. After exit_playground or a hard reset the
  // streamer leaves tracking paused; tapping B starts recording, which is
  // when the footer goes away.
  if (!isFirstTask) {
    currentTaskPrompt = { header, bullets, footer: "Tap {pedal:B} to start" };
  } else {
    currentTaskPrompt = { header, bullets };
  }
  taskPromptStartedMs = now;
  drawTaskPrompt(currentTaskPrompt);
};

const setTrackingPaused = (paused: boolean, now = performance.now()) => {
  if (trackingPaused === paused) {
    return;
  }
  if (paused) {
    pauseStartedMs = now;
  } else if (pauseStartedMs !== null) {
    markerClockStartMs += now - pauseStartedMs;
    pauseStartedMs = null;
  }
  trackingPaused = paused;
};

const resetMarkerClock = (now = performance.now()) => {
  markerClockStartMs = now;
  pauseStartedMs = trackingPaused ? now : null;
};

const getShotClockSeconds = (now = performance.now()) => {
  const clockNow = trackingPaused && pauseStartedMs !== null ? pauseStartedMs : now;
  return (clockNow - markerClockStartMs) / 1000;
};

const markMarkerSaved = (now = performance.now()) => {
  markerSavedAtMs = now;
};

const updateXRCheckAnimation = (now: number) => {
  const elapsed = now - markerSavedAtMs;
  const material = xrCheckMark.material as THREE.MeshBasicMaterial;
  if (elapsed < 0 || elapsed > 900) {
    xrCheckMark.visible = false;
    material.opacity = 0;
    return;
  }
  const progress = elapsed / 900;
  const pop = progress < 0.24 ? progress / 0.24 : 1;
  xrCheckMark.visible = true;
  xrCheckMark.scale.setScalar(0.7 + 0.38 * Math.sin(pop * Math.PI * 0.5));
  material.opacity = progress < 0.68 ? 1 : Math.max(0, 1 - (progress - 0.68) / 0.32);
};

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.xr.enabled = true;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
document.body.appendChild(renderer.domElement);

// XRButton (not ARButton) preserves WebXRManager's `local-floor` default for
// AR sessions. ARButton hard-codes `local` (eye-level), which puts the virtual
// floor at head height in passthrough. Render a separate VR button so users
// on devices that support both can pick — XRButton alone always prefers AR.
const xrFeatures = ["hand-tracking", "local-floor", "bounded-floor"];
const vrButton = VRButton.createButton(renderer, { optionalFeatures: xrFeatures });
vrControls.appendChild(vrButton);
const xrButton = XRButton.createButton(renderer, { optionalFeatures: xrFeatures });
vrControls.appendChild(xrButton);

// Full-screen split-screen splash: two large tiles for "ENTER XR (passthrough)"
// and "ENTER VR (immersive)". Each tile forwards its click to the underlying
// Three.js XR/VR button so all the WebXR feature/session plumbing keeps working.
const modeSplash = document.getElementById("mode-splash") as HTMLDivElement | null;
const modeTileXR = document.getElementById("mode-tile-xr") as HTMLButtonElement | null;
const modeTileVR = document.getElementById("mode-tile-vr") as HTMLButtonElement | null;

const syncModeTile = (tile: HTMLButtonElement | null, source: HTMLElement) => {
  if (!tile) return;
  // Disable our tile if the underlying button is disabled (e.g. AR/VR not
  // supported on this device).
  const disabled =
    (source as HTMLButtonElement).disabled
    || source.getAttribute("aria-disabled") === "true"
    || /not supported|not allowed|not found/i.test(source.textContent ?? "");
  if (disabled) {
    tile.setAttribute("disabled", "true");
  } else {
    tile.removeAttribute("disabled");
  }
};

const wireModeTile = (tile: HTMLButtonElement | null, source: HTMLElement) => {
  if (!tile) return;
  tile.addEventListener("click", () => {
    if (tile.hasAttribute("disabled")) return;
    source.click();
  });
  syncModeTile(tile, source);
  // Re-check on label change (Three updates the button text when XR
  // availability changes).
  const observer = new MutationObserver(() => syncModeTile(tile, source));
  observer.observe(source, { childList: true, characterData: true, subtree: true, attributes: true });
};

wireModeTile(modeTileXR, xrButton);
wireModeTile(modeTileVR, vrButton);

// Browser mode: just dismiss the splash and let OrbitControls take over.
// Renderer's animation loop already runs in non-XR; the scene streams as
// usual. No headset, no XR session, no hand input.
const modeTileBrowser = document.getElementById("mode-tile-browser") as HTMLButtonElement | null;
if (modeTileBrowser) {
  modeTileBrowser.addEventListener("click", () => {
    setModeSplashVisible(false);
    // Make the scene look reasonable in a desk browser: small helper grid
    // visible, no XR-only floor disc.
    grid.visible = true;
    xrFloor.visible = false;
    pedalGroup.visible = false;
  });
}

const setModeSplashVisible = (visible: boolean) => {
  if (!modeSplash) return;
  modeSplash.classList.toggle("hidden", !visible);
};

renderer.xr.addEventListener("sessionstart", () => {
  setModeSplashVisible(false);
  // In any XR session swap the small helper grid for the larger floor disc.
  grid.visible = false;
  xrFloor.visible = true;
  pedalGroup.visible = true;
  if (renderer.xr.getSession()?.mode === "immersive-ar") {
    // AR: hide the dark sky + starfield + ground so passthrough shows through.
    scene.background = null;
    renderer.setClearColor(0x000000, 0);
    vrScenery.visible = false;
  } else {
    scene.background = skyBaseColor;
    renderer.setClearColor(skyBaseColor, 1);
    vrScenery.visible = true;
  }
});

renderer.xr.addEventListener("sessionend", () => {
  setModeSplashVisible(true);
  scene.background = skyBaseColor;
  renderer.setClearColor(skyBaseColor, 1);
  vrScenery.visible = true;
  grid.visible = true;
  xrFloor.visible = false;
  pedalGroup.visible = false;
  // The XR local-floor reference space can nudge the camera's parent Y while
  // active; reassert the configured rig height after exit.
  vrRig.position.y = vrPos[1];
});

const hand0 = renderer.xr.getHand(0) as XRHandObject;
const hand1 = renderer.xr.getHand(1) as XRHandObject;

const handSide: { hand0: HandSide | null; hand1: HandSide | null } = { hand0: null, hand1: null };
type XRConnectEvent = Event & { data: { handedness: HandSide } };

hand0.addEventListener("connected", (event: Event) => {
  handSide.hand0 = (event as XRConnectEvent).data.handedness;
});
hand1.addEventListener("connected", (event: Event) => {
  handSide.hand1 = (event as XRConnectEvent).data.handedness;
});

// Always create the skeleton meshes; visibility is toggled per-frame in
// updateHandVis based on clientConfig.handSkeleton or trackingPaused.
{
  const jointMat = new THREE.MeshBasicMaterial({ color: 0x44ffaa });
  const tipMat = new THREE.MeshBasicMaterial({ color: 0xff6644 });
  const jointGeo = new THREE.SphereGeometry(0.006, 6, 6);
  const tipGeo = new THREE.SphereGeometry(0.008, 6, 6);
  const boneMat = new THREE.LineBasicMaterial({ color: 0x88ccff });
  const tips = new Set([4, 9, 14, 19, 24]);
  const bones = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8], [8, 9],
    [0, 10], [10, 11], [11, 12], [12, 13], [13, 14],
    [0, 15], [15, 16], [16, 17], [17, 18], [18, 19],
    [0, 20], [20, 21], [21, 22], [22, 23], [23, 24],
  ];

  const addJointVis = (hand: XRHandObject) => {
    const spheres: THREE.Mesh[] = [];
    for (let i = 0; i < 25; i += 1) {
      const mesh = new THREE.Mesh(tips.has(i) ? tipGeo : jointGeo, tips.has(i) ? tipMat : jointMat);
      mesh.visible = false;
      hand.add(mesh);
      spheres.push(mesh);
    }
    const boneLines = bones.map(([a, b]) => {
      const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
      const line = new THREE.Line(geo, boneMat);
      line.visible = false;
      hand.add(line);
      return { line, geo, a, b };
    });
    hand.userData.spheres = spheres;
    hand.userData.boneLines = boneLines;
  };

  addJointVis(hand0);
  addJointVis(hand1);
}

scene.add(hand0);
scene.add(hand1);

const controller0 = renderer.xr.getController(0);
const controller1 = renderer.xr.getController(1);
const controllerGrip0 = renderer.xr.getControllerGrip(0);
const controllerGrip1 = renderer.xr.getControllerGrip(1);

const makeGripSphere = () =>
  new THREE.Mesh(
    new THREE.SphereGeometry(0.02, 8, 8),
    new THREE.MeshBasicMaterial({ color: 0x44aaff }),
  );

controllerGrip0.add(makeGripSphere());
controllerGrip1.add(makeGripSphere());
scene.add(controller0);
scene.add(controller1);
scene.add(controllerGrip0);
scene.add(controllerGrip1);

const getTriggerValue = (handedness: HandSide) => {
  const session = renderer.xr.getSession();
  if (!session) {
    return 0;
  }
  for (const source of session.inputSources) {
    if (source.handedness === handedness && source.gamepad) {
      return source.gamepad.buttons[0]?.value ?? 0;
    }
  }
  return 0;
};

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(vrTarget[0], vrTarget[1], vrTarget[2]);
controls.update();

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

const sceneRoot = new THREE.Group();
sceneRoot.rotation.y = Math.PI / 2;
const scenePos = clientConfig.scenePos ?? [0, 0, 0];
// User-tunable vertical lift (meters), persisted server-side. Lifts the
// entire MuJoCo scene visually so the table appears at a comfortable height
// for the user. The XR floor disc is on `scene` (not `sceneRoot`) and stays
// at the physical floor regardless.
let tableHeightOffsetTarget = clientConfig.tableHeightOffset ?? 0;
let tableHeightOffsetCurrent = tableHeightOffsetTarget;
sceneRoot.position.set(scenePos[0], scenePos[1] + tableHeightOffsetCurrent, scenePos[2]);
scene.add(sceneRoot);

const toViewerMaterial = (
  material: THREE.Material,
  hasVertexColors: boolean,
  jengaTint: THREE.Color | null = null,
): THREE.Material => {
  const source = material as THREE.Material & {
    alphaTest?: number;
    color?: THREE.Color;
    map?: THREE.Texture | null;
    opacity?: number;
    transparent?: boolean;
    vertexColors?: boolean;
  };
  const color = source.color?.clone() ?? new THREE.Color(0xffffff);
  if (jengaTint) {
    color.lerp(jengaTint, 0.26);
  }
  const shaded = new THREE.MeshLambertMaterial({
    alphaTest: source.alphaTest ?? 0,
    color,
    emissive: jengaTint ?? new THREE.Color(0x000000),
    emissiveIntensity: jengaTint ? 0.08 : 0,
    map: source.map ?? null,
    opacity: source.opacity ?? 1,
    side: THREE.DoubleSide,
    transparent: source.transparent ?? false,
    vertexColors: hasVertexColors || Boolean(source.vertexColors),
  });
  return shaded;
};

const loader = new GLTFLoader();
const bodyMeshes: Record<string, THREE.Group> = {};
let ws: WebSocket | null = null;
let frameCount = 0;
let currentSessionId: string | null = null;
let reloadInProgress = false;

const sceneRootInv = new THREE.Matrix4();
const tmpPos = new THREE.Vector3();
const tmpQuat = new THREE.Quaternion();
const sceneRootQuat = new THREE.Quaternion();

const updateHandVis = (hand: XRHandObject) => {
  if (!hand.userData.spheres || !hand.userData.boneLines) {
    return;
  }
  // Show the skeleton when the user has it explicitly on, OR while tracking is
  // paused (so they can verify finger registration before resuming).
  const showSkeleton = clientConfig.handSkeleton || trackingPaused;
  if (!showSkeleton) {
    for (const sphere of hand.userData.spheres) sphere.visible = false;
    for (const { line } of hand.userData.boneLines) line.visible = false;
    return;
  }
  const wristJoint = hand.joints[JOINT_NAMES[0]];
  const wristPos = clientConfig.handScale !== 1 && wristJoint?.visible ? wristJoint.position : null;
  for (let i = 0; i < JOINT_NAMES.length; i += 1) {
    const joint = hand.joints[JOINT_NAMES[i]];
    const sphere = hand.userData.spheres[i];
    if (joint?.visible) {
      sphere.position.copy(joint.position);
      if (clientConfig.handScale !== 1 && wristPos && i !== 0) {
        sphere.position.sub(wristPos).multiplyScalar(clientConfig.handScale).add(wristPos);
      }
      sphere.visible = true;
    } else {
      sphere.visible = false;
    }
  }
  for (const { line, geo, a, b } of hand.userData.boneLines) {
    if (hand.userData.spheres[a].visible && hand.userData.spheres[b].visible) {
      const positions = geo.attributes.position.array as Float32Array;
      positions[0] = hand.userData.spheres[a].position.x;
      positions[1] = hand.userData.spheres[a].position.y;
      positions[2] = hand.userData.spheres[a].position.z;
      positions[3] = hand.userData.spheres[b].position.x;
      positions[4] = hand.userData.spheres[b].position.y;
      positions[5] = hand.userData.spheres[b].position.z;
      geo.attributes.position.needsUpdate = true;
      line.visible = true;
    } else {
      line.visible = false;
    }
  }
};

let wsLastBinaryAt = performance.now();
let wsLastTextAt = performance.now();
let wsBinaryCount = 0;
let wsTextCount = 0;
let wsConnectStart = 0;
let wsConnectAttempts = 0;

const connectWS = () => {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  wsConnectStart = performance.now();
  wsConnectAttempts += 1;
  console.log(`[ws] connect attempt #${wsConnectAttempts}`);
  ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    const dt = performance.now() - wsConnectStart;
    console.log(`[ws] open  (took ${dt.toFixed(0)}ms; attempt #${wsConnectAttempts})`);
    info.textContent = "Connected! Waiting for GELLO...";
  };

  ws.onerror = (event) => {
    console.log(`[ws] error  bytesSinceOpen=${wsBinaryCount} textsSinceOpen=${wsTextCount}`, event);
  };

  ws.onmessage = (event) => {
    const now = performance.now();
    if (typeof event.data === "string") {
      const gap = now - wsLastTextAt;
      wsLastTextAt = now;
      wsTextCount += 1;
      try {
        const msg = JSON.parse(event.data);
        if (msg?.type === "checkpoint_result" && msg.accepted === false) {
          flashBanner("Release object to record checkpoint", "#ffaa55");
        } else if (msg?.type === "snapshot_saved") {
          flashBanner(`Snapshot ${msg.count} saved`, "#44ff88", 1200);
        } else if (msg?.type === "task_finished") {
          lastTaskFinishResult = msg.result === "skipped" ? "skipped" : "complete";
        } else if (msg?.type === "scene_will_reload") {
          // Pre-emptive notice from the server that a rebuild is starting.
          // Show the loading panel immediately so the user sees a hold
          // signal rather than a momentarily-empty scene while we wait for
          // pollStatus to detect the session_id bump.
          loadingActive = true;
          loadingLabel = "Reloading scene";
          loadingAnimationStart = performance.now();
          worldLoadingPanel.visible = true;
          console.log(`[ws] scene_will_reload received`);
        } else if (msg?.type === "loading") {
          loadingActive = !!msg.active;
          loadingLabel = String(msg.label ?? "Loading");
          if (loadingActive) {
            loadingAnimationStart = performance.now();
            worldLoadingPanel.visible = true;
            console.log(`[ws] loading=true label=${msg.label} (binary_gap=${(now - wsLastBinaryAt).toFixed(0)}ms since last sim frame)`);
          } else {
            worldLoadingPanel.visible = false;
            console.log(`[ws] loading=false (binary_gap=${(now - wsLastBinaryAt).toFixed(0)}ms since last sim frame)`);
          }
        } else {
          console.log(`[ws] unknown text msg type=${msg?.type}`);
        }
      } catch (e) {
        console.log(`[ws] text JSON parse failed:`, e);
      }
      if (gap > 1000) {
        console.log(`[ws] text gap=${gap.toFixed(0)}ms`);
      }
      return;
    }
    const binGap = now - wsLastBinaryAt;
    wsLastBinaryAt = now;
    wsBinaryCount += 1;
    if (binGap > 500) {
      console.log(`[ws] binary gap=${binGap.toFixed(0)}ms (binary #${wsBinaryCount}, total text=${wsTextCount})`);
    }
    lastSimFrameAtMs = now;
    const buf = new Float32Array(event.data as ArrayBuffer);
    const frameWidth = 11;
    const count = buf.length / frameWidth;
    for (let i = 0; i < count; i += 1) {
      const offset = i * frameWidth;
      const bodyId = Math.round(buf[offset]).toString();
      if (window._mocapBodyIds?.has(bodyId)) {
        continue;
      }
      const obj = bodyMeshes[bodyId];
      if (obj) {
        obj.position.set(buf[offset + 1], buf[offset + 2], buf[offset + 3]);
        obj.quaternion.set(buf[offset + 4], buf[offset + 5], buf[offset + 6], buf[offset + 7]);
        obj.scale.set(buf[offset + 8] || 1.0, buf[offset + 9] || 1.0, buf[offset + 10] || 1.0);
      }
    }
    frameCount += 1;
    if (frameCount === 1) {
      info.textContent = "";
    }
  };

  ws.onclose = (event) => {
    const lifetime = performance.now() - wsConnectStart;
    console.log(`[ws] close  code=${event.code} reason=${JSON.stringify(event.reason)} wasClean=${event.wasClean} `
      + `lifetime=${lifetime.toFixed(0)}ms binary=${wsBinaryCount} text=${wsTextCount}`);
    info.textContent = "Disconnected. Reconnecting...";
    setTimeout(connectWS, 1000);
  };
};


// Tear down all body meshes and mocap gizmos so we can refetch them. Called
// when the server's session_id changes (i.e. a new server is running and the
// scene topology may have changed). Does NOT touch the VR rig or HUD meshes.
const tearDownLoadedAssets = () => {
  // Body meshes: detach from sceneRoot, dispose geometry/material/textures.
  for (const obj of Object.values(bodyMeshes)) {
    obj.traverse((child) => {
      const mesh = child as THREE.Mesh;
      if (!mesh.isMesh) return;
      mesh.geometry?.dispose();
      const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      for (const mat of mats) {
        const m = mat as THREE.MeshStandardMaterial;
        m.map?.dispose();
        m.normalMap?.dispose();
        m.dispose?.();
      }
    });
    obj.removeFromParent();
  }
  for (const key of Object.keys(bodyMeshes)) {
    delete bodyMeshes[key];
  }

  // Mocap targets: dispose meshes. (TransformControls were removed; the
  // ``tc`` slot is kept on the entry for back-compat but may be null.)
  for (const gizmo of window._gizmos ?? []) {
    if (gizmo.tc) {
      gizmo.tc.detach();
      gizmo.tc.removeFromParent();
      gizmo.tc.dispose?.();
    }
    gizmo.target.traverse((child) => {
      const mesh = child as THREE.Mesh;
      if (!mesh.isMesh) return;
      mesh.geometry?.dispose();
      const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      for (const mat of mats) (mat as THREE.Material).dispose?.();
    });
    gizmo.target.removeFromParent();
  }
  window._gizmos = [];
  window._mocapBodyIds = new Set<string>();
  window._initMocapPoses = [];
  window._handMocapMapping = { left: {}, right: {} };
};

const loadAssets = async () => {
  info.textContent = "Loading scene...";
  const bodyInfo = (await fetch("/api/bodies").then((resp) => resp.json())) as Record<string, BodyInfoEntry>;
  const cacheBust = `?t=${Date.now()}`;
  const entries = Object.entries(bodyInfo);
  let loaded = 0;

  const loadOne = async (bodyId: string, bodyData: BodyInfoEntry) => {
    try {
      const jengaTint = getJengaTint(bodyData.name);
      const gltf = await new Promise<any>((resolve, reject) => {
        loader.load(`/meshes/${bodyData.file}${cacheBust}`, resolve, undefined, reject);
      });
      const obj = gltf.scene as THREE.Group;
      const bodyOpacity = bodyData.opacity ?? 1;
      obj.traverse((child) => {
        const mesh = child as THREE.Mesh;
        if (!mesh.isMesh) {
          return;
        }
        mesh.geometry.computeVertexNormals();
        const colorAttr = mesh.geometry.getAttribute("color");
        const hasVertexColors = Boolean(colorAttr);
        let vertexAlpha = 1;
        if (colorAttr && colorAttr.itemSize === 4) {
          let minAlpha = 1;
          for (let i = 0; i < colorAttr.count; i += 1) {
            const a = colorAttr.getW(i);
            if (a < minAlpha) minAlpha = a;
          }
          vertexAlpha = minAlpha;
        }
        const meshOpacity = Math.min(bodyOpacity, vertexAlpha);
        const applyOpacity = (mat: THREE.Material) => {
          if (meshOpacity < 1) {
            mat.transparent = true;
            (mat as THREE.MeshStandardMaterial).opacity = meshOpacity;
            mat.depthWrite = false;
          }
          return mat;
        };
        if (Array.isArray(mesh.material)) {
          mesh.material = mesh.material.map((material) =>
            applyOpacity(toViewerMaterial(material, hasVertexColors, jengaTint)),
          );
        } else {
          mesh.material = applyOpacity(toViewerMaterial(mesh.material, hasVertexColors, jengaTint));
        }
      });
      obj.position.fromArray(bodyData.position);
      obj.quaternion.fromArray(bodyData.quaternion);
      applyBodyScale(obj, bodyData.scale);
      sceneRoot.add(obj);
      bodyMeshes[bodyId] = obj;
      loaded += 1;
      info.textContent = `Loading meshes: ${loaded}/${entries.length}`;
    } catch (error) {
      console.error(`Failed to load body ${bodyId}:`, error);
    }
  };
  // Parallel GLB fetches: serial loop was the dominant freeze source on
  // rebuild (n_bodies × per-fetch latency). loader.load is non-blocking
  // (network + GPU upload happen off-thread); the only sync cost is mesh
  // post-processing, which is fast.
  await Promise.all(entries.map(([id, data]) => loadOne(id, data)));

  info.textContent = `Loaded ${loaded}/${entries.length} bodies. Loading mocap...`;
  try {
    const mocapInfo = (await fetch("/api/mocap").then((resp) => resp.json())) as MocapInfoEntry[];
    for (const mocap of mocapInfo) {
      // Empty group keeps the mocap target's pose for the rest of the system
      // (initial-pose tracking, hand-mocap mapping below) but renders nothing
      // visible. Previously we drew a colored sphere + TransformControls
      // here; both were visual clutter in VR where the actual hand meshes
      // already indicate target location.
      const gizmoTarget = new THREE.Group();
      gizmoTarget.position.set(...mocap.position);
      gizmoTarget.quaternion.set(...mocap.quaternion);
      sceneRoot.add(gizmoTarget);
      window._gizmos ??= [];
      window._gizmos.push({ tc: null, target: gizmoTarget });

      window._mocapBodyIds ??= new Set<string>();
      window._mocapBodyIds.add(String(mocap.body_id));
      window._initMocapPoses ??= [];
      window._initMocapPoses.push({
        id: mocap.mocap_id,
        pos: mocap.position,
        quat: mocap.quaternion,
      });
    }

    window._handMocapMapping = { left: {}, right: {} };
    const allJoints = new Set<string>(JOINT_NAMES);
    for (const mocap of mocapInfo) {
      for (const side of ["left", "right"] as const) {
        const prefix = `${side}-`;
        if (mocap.name.startsWith(prefix)) {
          const jointName = mocap.name.slice(prefix.length) as JointName;
          if (allJoints.has(jointName)) {
            window._handMocapMapping[side][jointName] = mocap.mocap_id;
          }
        }
      }
    }

    if (mocapInfo.length > 0) {
      info.textContent =
        `${mocapInfo.length} mocap targets loaded. Press G=translate, R=rotate. In VR: hold grip to move.`;
    }
  } catch (error) {
    console.log("No mocap targets:", error);
  }
};

const loadScene = async () => {
  await loadAssets();
  connectWS();
};

// Reset all frontend-local UI/timer state so the page behaves like a fresh
// boot after a server restart. Does NOT touch ws (it'll reconnect on its own)
// nor the VR rig.
const resetLocalState = () => {
  frameCount = 0;
  lastMarkerCount = 0;
  markerClockStartMs = performance.now();
  pauseStartedMs = null;
  markerSavedAtMs = -Infinity;
  lastTaskKey = "";
  currentTaskPrompt = emptyTaskPrompt();
  taskPromptStartedMs = -Infinity;
  prevStatusState = null;
  trackingWasActive = false;
  trackingIsActive = false;
  trackingPaused = false;
  xrHudState = "idle";
  playgroundMode = false;
  bannerLastKey = "";
  bannerTransientUntilMs = -Infinity;
  xrBanner.visible = false;
};

const reloadOnSessionChange = async () => {
  if (reloadInProgress) return;
  reloadInProgress = true;
  // Show the loading overlay while we reload assets, so the user knows the
  // frontend is busy rather than that teleop has frozen. We extend whatever
  // the server already had visible.
  const wasShowingPanel = worldLoadingPanel.visible;
  loadingActive = true;
  loadingLabel = "Reloading scene assets";
  loadingAnimationStart = performance.now();
  worldLoadingPanel.visible = true;
  console.log(`[reload] starting (panel_was_visible=${wasShowingPanel})`);
  const t0 = performance.now();
  try {
    tearDownLoadedAssets();
    resetLocalState();
    await loadAssets();
  } catch (error) {
    console.error("Asset reload failed:", error);
  } finally {
    const dt = performance.now() - t0;
    console.log(`[reload] complete (took ${dt.toFixed(0)}ms)`);
    reloadInProgress = false;
    loadingActive = false;
    worldLoadingPanel.visible = false;
  }
};

renderer.xr.addEventListener("sessionstart", () => {
  for (const gizmo of window._gizmos ?? []) {
    gizmo.tc.visible = false;
    gizmo.tc.enabled = false;
    gizmo.target.visible = false;
  }
});

renderer.xr.addEventListener("sessionend", () => {
  for (const gizmo of window._gizmos ?? []) {
    gizmo.tc.visible = true;
    gizmo.tc.enabled = true;
    gizmo.target.visible = true;
  }
});

renderer.setAnimationLoop(() => {
  const now = performance.now();
  // Animate the loading panel while it's active so the user knows the server
  // is doing something rather than that the headset has frozen. If the server
  // is silent (no sim frame in a while) but never sent loading=true, fall back
  // to a "Waiting for sim..." indicator so the user knows the freeze isn't on
  // their side.
  const frameAgeMs = now - lastSimFrameAtMs;
  const stale = frameAgeMs > STALE_FRAME_WARN_MS;
  if (loadingActive || stale) {
    if (!loadingActive && stale) {
      loadingLabel = `Waiting for sim (${Math.round(frameAgeMs)}ms)`;
      loadingAnimationStart = loadingAnimationStart || now;
      worldLoadingPanel.visible = true;
    }
    const phase = ((now - loadingAnimationStart) / 1000) % 4;
    drawLoadingPanel(phase);
  } else if (worldLoadingPanel.visible && !loadingActive) {
    worldLoadingPanel.visible = false;
  }
  // Smoothly ease the scene-root vertical lift toward the server-reported
  // target so A/C-driven height adjustments don't jump in 50ms steps.
  if (tableHeightOffsetCurrent !== tableHeightOffsetTarget) {
    const diff = tableHeightOffsetTarget - tableHeightOffsetCurrent;
    if (Math.abs(diff) < 1e-4) {
      tableHeightOffsetCurrent = tableHeightOffsetTarget;
    } else {
      tableHeightOffsetCurrent += diff * 0.25;  // ~60ms low-pass at 60Hz
    }
    sceneRoot.position.y = scenePos[1] + tableHeightOffsetCurrent;
  }
  updateHandVis(hand0);
  updateHandVis(hand1);

  const activeCamera = renderer.xr.isPresenting ? renderer.xr.getCamera(camera) : camera;
  activeCamera.getWorldPosition(xrHudWorldPos);
  activeCamera.getWorldQuaternion(xrHudWorldQuat);
  xrHud.position.copy(xrHudWorldPos);
  xrHud.quaternion.copy(xrHudWorldQuat);
  xrHudOffsetWorld.copy(xrHudOffset).applyQuaternion(xrHudWorldQuat);
  xrHud.position.add(xrHudOffsetWorld);
  const trackingActive = trackingIsActive || isTrackingState(xrHudState);
  updateTaskPrompt();
  xrHud.visible = true;
  const pulse = trackingPaused ? 1 : 0.85 + 0.15 * Math.sin(now * 0.01);
  (xrRecDot.material as THREE.MeshBasicMaterial).color.set(trackingPaused ? 0xffaa00 : 0xff3333);
  xrRecDot.scale.setScalar(pulse);
  xrRecRing.scale.setScalar(1.0 + (trackingPaused ? 0 : 0.08 * Math.sin(now * 0.01 + 0.8)));
  xrRecDot.visible = trackingActive && !playgroundMode;
  xrRecRing.visible = trackingActive && !playgroundMode;
  const shotClockSeconds = trackingActive ? getShotClockSeconds(now) : 0;
  updatePersistentBanner(now, shotClockSeconds, trackingActive);
  updateXRCheckAnimation(now);
  renderPedals(now);
  tickSlideshow(now);

  if (
    !reloadInProgress
    && renderer.xr.isPresenting
    && ws
    && ws.readyState === WebSocket.OPEN
  ) {
    ws.send(
      JSON.stringify({
        type: "trigger",
        left: getTriggerValue("left"),
        right: getTriggerValue("right"),
      }),
    );

    if (window._handMocapMapping) {
      const hands: Array<{ xrHand: XRHandObject; side: HandSide }> = [];
      if (handSide.hand0) hands.push({ xrHand: hand0, side: handSide.hand0 });
      if (handSide.hand1) hands.push({ xrHand: hand1, side: handSide.hand1 });

      for (const { xrHand, side } of hands) {
        const mapping = window._handMocapMapping[side];
        if (!mapping || Object.keys(mapping).length === 0) {
          continue;
        }
        let wristLocalPos: THREE.Vector3 | null = null;
        if (clientConfig.handScale !== 1) {
          const wristJoint = xrHand.joints.wrist;
          if (wristJoint?.visible) {
            sceneRootInv.copy(sceneRoot.matrixWorld).invert();
            wristJoint.getWorldPosition(tmpPos);
            wristLocalPos = tmpPos.clone().applyMatrix4(sceneRootInv);
          }
        }

        for (const [jointName, mocapId] of Object.entries(mapping)) {
          const joint = xrHand.joints[jointName];
          if (!joint?.visible) {
            continue;
          }
          sceneRootInv.copy(sceneRoot.matrixWorld).invert();
          joint.getWorldPosition(tmpPos);
          joint.getWorldQuaternion(tmpQuat);
          if (!Number.isFinite(tmpPos.x) || !Number.isFinite(tmpPos.y) || !Number.isFinite(tmpPos.z)) {
            continue;
          }
          const localPos = tmpPos.clone().applyMatrix4(sceneRootInv);
          sceneRootQuat.setFromRotationMatrix(sceneRoot.matrixWorld);
          const localQuat = sceneRootQuat.clone().invert().multiply(tmpQuat);
          if (clientConfig.handScale !== 1 && wristLocalPos && jointName !== "wrist") {
            localPos.sub(wristLocalPos).multiplyScalar(clientConfig.handScale).add(wristLocalPos);
          }
          ws.send(
            JSON.stringify({
              type: "mocap",
              mocap_id: mocapId,
              position: [localPos.x, localPos.y, localPos.z],
              quaternion: [localQuat.x, localQuat.y, localQuat.z, localQuat.w],
            }),
          );
        }
      }
    }
  }

  renderer.render(scene, camera);
});

const pollStatus = async () => {
  try {
    const status = (await fetch("/api/status").then((resp) => resp.json())) as StatusResponse;
    if (status.session_id) {
      if (currentSessionId === null) {
        currentSessionId = status.session_id;
      } else if (status.session_id !== currentSessionId) {
        currentSessionId = status.session_id;
        reloadOnSessionChange();
        return;
      }
    }
    if (reloadInProgress) {
      return;
    }
    const now = performance.now();
    const prevPlaygroundMode = playgroundMode;
    const inPlayground = Boolean(status.playground) || status.state === "playground";
    // Collector and choreographer share the two-pane layout: task description
    // on the left, snapshot slideshow on the right. Admin mode is text-only.
    const twoPane = status.mode === "collector" || status.mode === "choreographer";
    const postTask = Boolean(status.awaiting_post_task_choice);
    if (twoPane) {
      worldTaskPrompt.position.copy(TASK_PROMPT_FAR_POS_COLLECTOR);
      worldTaskPrompt.rotation.y = PANEL_SURROUND_YAW;
      worldSlideshow.position.copy(SLIDESHOW_FAR_POS_COLLECTOR);
      worldSlideshow.rotation.y = -PANEL_SURROUND_YAW;
      const slideshowUrlsList = status.snapshots?.snapshots ?? [];
      const taskKey = `${status.snapshots?.task_id ?? ""}:${status.episode}`;
      setSlideshowUrls(taskKey, slideshowUrlsList);
      // The slideshow only plays once the user has left playground; while in
      // playground it stays hidden and its clock is held so it doesn't jump
      // ahead the moment the real task appears.
      worldSlideshow.visible = worldTaskPrompt.visible && !inPlayground;
      if (inPlayground) {
        slideshowLastAdvanceMs = performance.now();
      }
      setTaskPromptFromStatus(status.task, now, { postTask });
    } else {
      worldTaskPrompt.position.copy(TASK_PROMPT_FAR_POS);
      worldTaskPrompt.rotation.y = 0;
      worldSlideshow.visible = false;
      setTaskPromptFromStatus(status.task, now, { postTask: false });
    }
    playgroundMode = inPlayground;
    // Leaving playground is the first time the user sees a real task — pop
    // the close panel even if the task index didn't change (it didn't, the
    // task was selected at boot).
    if (prevPlaygroundMode && !playgroundMode) {
      if (
        currentTaskPrompt.header.length > 0
        || currentTaskPrompt.bullets.length > 0
      ) {
        currentTaskPrompt = {
          ...currentTaskPrompt,
          footer: "Tap {pedal:B} to start",
        };
        drawTaskPrompt(currentTaskPrompt);
      }
    }
    if (typeof status.table_height_offset === "number" && Number.isFinite(status.table_height_offset)) {
      tableHeightOffsetTarget = status.table_height_offset;
    }
    const trackingActive = Boolean(status.tracking_active) || isTrackingState(status.state);
    const statusMarkerCount = Math.max(0, Math.floor(finiteNumber(status.marker_count) ?? lastMarkerCount));
    if (trackingActive && !trackingWasActive) {
      resetMarkerClock(now);
      lastMarkerCount = statusMarkerCount;
    } else if (!trackingActive && trackingWasActive) {
      setTrackingPaused(false, now);
      resetMarkerClock(now);
      lastMarkerCount = statusMarkerCount;
    }
    trackingWasActive = trackingActive;
    const serverPaused =
      typeof status.tracking_paused === "boolean"
        ? status.tracking_paused
        : status.state === "paused"
          ? true
          : undefined;
    if (serverPaused !== undefined) {
      setTrackingPaused(serverPaused, now);
    }
    if (trackingActive) {
      const markerCount = statusMarkerCount;
      if (markerCount > lastMarkerCount) {
        markMarkerSaved(now);
      }
      lastMarkerCount = markerCount;
      if (typeof status.seconds_since_marker === "number" && Number.isFinite(status.seconds_since_marker)) {
        const clockNow = trackingPaused && pauseStartedMs !== null ? pauseStartedMs : now;
        markerClockStartMs = clockNow - status.seconds_since_marker * 1000;
      }
    }
    // The close panel is dismissed when B has been held for the full
    // Once recording actually starts, drop the "Tap (B) to start" footer so
    // the panel reads as a glanceable reference instead of a prompt.
    if (
      currentTaskPrompt.footer
      && status.state === "recording"
      && prevStatusState !== "recording"
    ) {
      currentTaskPrompt = { ...currentTaskPrompt, footer: undefined };
      drawTaskPrompt(currentTaskPrompt);
    }
    prevStatusState = status.state;

    // Refresh pedal logical state from the server. The per-frame loop turns
    // (held, heldSeconds, lastPressCount) into a fillLevel animation, so a
    // tap that fell between two polls still flashes the pedal.
    const pedals = status.pedals ?? {};
    const pedalLabels = status.pedal_labels ?? {};
    for (const letter of pedalLetters) {
      const info = pedals[letter.toLowerCase()];
      const heldNow = Boolean(info?.held);
      const heldSeconds =
        typeof info?.held_seconds === "number" ? info.held_seconds : 0;
      const longPress =
        typeof info?.long_press_seconds === "number" ? info.long_press_seconds : 0;
      const pressCount =
        typeof info?.press_count === "number" ? info.press_count : 0;
      if (pressCount > pedalLastPressCount[letter]) {
        pedalFlashUntilMs[letter] = now + PEDAL_FLASH_DURATION_MS;
        pedalLastPressCount[letter] = pressCount;
      }
      pedalLogical[letter] = { heldNow, heldSeconds, heldSampleMs: now, longPress };
      drawPedalLabel(letter, pedalLabels[letter.toLowerCase()] ?? "");
    }
    xrHudState = status.state;
    trackingIsActive = trackingActive;
    let text = "";
    if (status.state === "delay") {
      recDot.className = "";
      text = `EP ${status.episode}  Resetting...`;
    } else if (trackingActive) {
      recDot.className = trackingPaused ? "paused" : "recording";
      text = `${trackingPaused ? "PAUSED" : "REC"}  EP ${status.episode}  ${status.frame_count} frames`;
    } else {
      recDot.className = "";
      text = `EP ${status.episode}  Ready`;
    }
    if (status.task?.instruction || status.task?.title) {
      const taskLabel = status.task.title || status.task.instruction?.split("\n")[0] || "";
      text += `  |  Task ${(status.task.task_index ?? 0) + 1}/${status.task.total_tasks}: ${taskLabel}`;
    }
    statusText.textContent = text;
  } catch {
    // Keep last UI state if polling fails.
  }
};

setInterval(pollStatus, 500);

// Mode-switch keys for the desktop gizmos (rotate/translate). Not commands.
window.addEventListener("keydown", (event) => {
  for (const gizmo of window._gizmos ?? []) {
    if (event.key === "r") gizmo.tc.setMode("rotate");
    if (event.key === "g") gizmo.tc.setMode("translate");
  }
});

await loadScene();

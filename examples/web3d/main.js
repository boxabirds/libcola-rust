// 3D WebGPU demo: mesh structures reconstructed by stress majorization.

import { Renderer, buildNodeInstances, buildEdgeInstances } from './renderer.js';
import { OrbitCamera } from './camera.js';
import { icosphere, torus, cube, gridSheet, octahedron, cylinder, scramble, meanEdgeLength } from './meshes.js';

const DEFAULT_IDEAL_EDGE_LENGTH = 40;
const DEFAULT_DETAIL = 50;
const NODE_SIZE = 10; // width/height/depth for solver

// Palette: soft saturated colors for nodes
const PALETTE = [
  [0.36, 0.72, 0.96, 1],  // sky blue
  [0.96, 0.42, 0.52, 1],  // coral
  [0.40, 0.88, 0.64, 1],  // mint
  [0.96, 0.76, 0.28, 1],  // gold
  [0.68, 0.52, 0.96, 1],  // lavender
  [0.96, 0.56, 0.36, 1],  // tangerine
  [0.44, 0.84, 0.88, 1],  // teal
  [0.92, 0.48, 0.76, 1],  // pink
];

const EDGE_COLOR = [0.55, 0.58, 0.68, 1]; // muted blue-grey

let renderer, camera, layout, wasmModule;
let positions = [];      // flat [x,y,z, ...]
let edges = [];           // [[src, dst], ...]
let nodeCount = 0;
let running = true;
let converged = false;
let iterationCount = 0;

// Rendering sizes scale with mesh density
let nodeRadius = 4;
let edgeRadius = 1;

// --- Mesh generators ---

const GENERATORS = {
  sphere:   (n) => icosphere(n),
  torus:    (n) => torus(n),
  cube:     (n) => cube(n),
  cylinder: (n) => cylinder(n),
  grid:     (n) => gridSheet(n),
  octahedron: () => octahedron(),
};

// --- Layout setup ---

function setupLayout(meshType, detail, idealLength) {
  const gen = GENERATORS[meshType] || GENERATORS.sphere;
  const mesh = gen(detail);

  nodeCount = mesh.nodeCount;
  edges = mesh.edges;

  // Scale rendering sizes inversely with density
  const BASE_NODE_RADIUS = 3;
  const BASE_EDGE_RADIUS = 0.8;
  const DENSITY_REFERENCE = 40;
  const scaleFactor = Math.sqrt(DENSITY_REFERENCE / Math.max(1, nodeCount));
  nodeRadius = BASE_NODE_RADIUS * Math.max(0.3, scaleFactor);
  edgeRadius = BASE_EDGE_RADIUS * Math.max(0.2, scaleFactor);

  // Scramble positions into a compact ball
  const SCRAMBLE_RADIUS_FACTOR = 0.4;
  const scrambleRadius = idealLength * SCRAMBLE_RADIUS_FACTOR * Math.sqrt(nodeCount / 10);
  positions = scramble(nodeCount, scrambleRadius);

  converged = false;
  iterationCount = 0;

  const { ColaLayout3D } = wasmModule;
  layout = new ColaLayout3D();

  for (let i = 0; i < nodeCount; i++) {
    layout.addNode(
      positions[i * 3],
      positions[i * 3 + 1],
      positions[i * 3 + 2],
      NODE_SIZE, NODE_SIZE, NODE_SIZE,
    );
  }
  for (const [src, dst] of edges) {
    layout.addEdge(src, dst);
  }
  layout.setIdealEdgeLength(idealLength);

  updateGPU();
}

function stepLayout() {
  if (!layout || converged) return;

  const result = layout.runOnce();
  iterationCount++;

  for (let i = 0; i < nodeCount; i++) {
    positions[i * 3] = result.getX(i);
    positions[i * 3 + 1] = result.getY(i);
    positions[i * 3 + 2] = result.getZ(i);
  }
  result.free();

  const MAX_ITERATIONS = 500;
  if (iterationCount >= MAX_ITERATIONS) {
    converged = true;
  }

  updateGPU();
}

function updateGPU() {
  // Node colors from palette
  const colors = [];
  for (let i = 0; i < nodeCount; i++) {
    colors.push(PALETTE[i % PALETTE.length]);
  }
  const nodeData = buildNodeInstances(positions, nodeCount, nodeRadius, colors);
  renderer.updateNodeInstances(nodeData, nodeCount);

  const edgeData = buildEdgeInstances(positions, edges, edgeRadius, EDGE_COLOR);
  renderer.updateEdgeInstances(edgeData, edges.length);
}

// --- Main ---

async function main() {
  const canvas = document.getElementById('canvas');
  const overlay = document.getElementById('overlay');
  const statusEl = document.getElementById('status');

  // Resize canvas to fill window
  function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(window.innerWidth * dpr);
    canvas.height = Math.floor(window.innerHeight * dpr);
    canvas.style.width = window.innerWidth + 'px';
    canvas.style.height = window.innerHeight + 'px';
    if (renderer?.device) renderer.resize(canvas.width, canvas.height);
    if (camera) camera.resize(canvas.width, canvas.height);
  }
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  // Init renderer
  overlay.textContent = 'Initializing WebGPU...';
  renderer = new Renderer(canvas);
  try {
    await renderer.init();
  } catch (e) {
    overlay.textContent = 'WebGPU not available. Use Chrome/Edge 113+.';
    overlay.style.color = '#f87171';
    console.error(e);
    return;
  }

  // Init WASM
  overlay.textContent = 'Loading WASM solver...';
  try {
    wasmModule = await import('../../pkg/libcola.js');
    await wasmModule.default();
  } catch (e) {
    overlay.textContent = 'Failed to load WASM module.';
    overlay.style.color = '#f87171';
    console.error(e);
    return;
  }

  // Camera
  camera = new OrbitCamera(canvas);
  resizeCanvas();

  // Hide overlay
  overlay.style.display = 'none';

  // Controls
  const meshSelect = document.getElementById('meshType');
  const detailSlider = document.getElementById('detail');
  const detailLabel = document.getElementById('detailLabel');
  const lengthSlider = document.getElementById('idealLength');
  const lengthLabel = document.getElementById('idealLengthLabel');
  const playBtn = document.getElementById('playBtn');
  const resetBtn = document.getElementById('resetBtn');

  detailSlider.value = DEFAULT_DETAIL;
  detailLabel.textContent = DEFAULT_DETAIL;
  lengthSlider.value = DEFAULT_IDEAL_EDGE_LENGTH;
  lengthLabel.textContent = DEFAULT_IDEAL_EDGE_LENGTH;

  function resetMesh() {
    setupLayout(
      meshSelect.value,
      parseInt(detailSlider.value),
      parseInt(lengthSlider.value),
    );
    running = true;
    playBtn.textContent = 'Pause';
  }

  detailSlider.addEventListener('input', () => {
    detailLabel.textContent = detailSlider.value;
  });
  lengthSlider.addEventListener('input', () => {
    lengthLabel.textContent = lengthSlider.value;
  });

  meshSelect.addEventListener('change', resetMesh);
  resetBtn.addEventListener('click', resetMesh);

  playBtn.addEventListener('click', () => {
    if (converged) {
      resetMesh();
    } else {
      running = !running;
      playBtn.textContent = running ? 'Pause' : 'Play';
    }
  });

  // Initial mesh
  resetMesh();

  // Animation loop
  let lastTime = 0;
  const TARGET_FRAME_MS = 16;

  function frame(time) {
    const dt = time - lastTime;

    if (running && !converged && dt >= TARGET_FRAME_MS) {
      lastTime = time;
      stepLayout();
    }

    // Status
    if (converged) {
      statusEl.textContent = `Converged (${iterationCount} iters) — ${nodeCount} nodes, ${edges.length} edges`;
      playBtn.textContent = 'Restart';
    } else if (running) {
      statusEl.textContent = `Iter ${iterationCount} — ${nodeCount} nodes, ${edges.length} edges`;
    } else {
      statusEl.textContent = `Paused (${iterationCount}) — ${nodeCount} nodes, ${edges.length} edges`;
    }

    renderer.updateUniforms(camera);
    renderer.render();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main();

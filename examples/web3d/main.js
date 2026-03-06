// 3D WebGPU demo: mesh structures reconstructed by stress majorization.

import { Renderer, buildFaceVertices, buildEdgeInstances } from './renderer.js';
import { OrbitCamera } from './camera.js';
import { icosphere, torus, cube, gridSheet, cylinder, scramble } from './meshes.js';

const DEFAULT_IDEAL_EDGE_LENGTH = 40;
const DEFAULT_DETAIL = 50;
const NODE_SIZE = 10; // width/height/depth for solver

// Face colors: blend between these based on face normal direction
const FACE_COLOR_COOL = [0.22, 0.45, 0.72];  // steel blue (downward faces)
const FACE_COLOR_WARM = [0.42, 0.75, 0.88];  // sky blue (upward faces)

// Wireframe edge color: dark, subtle
const EDGE_COLOR = [0.08, 0.12, 0.18, 1]; // near-black

let renderer, camera, layout, wasmModule;
let positions = [];      // flat [x,y,z, ...]
let edges = [];           // [[src, dst], ...]
let faces = [];           // [[a,b,c], ...]
let nodeCount = 0;
let running = true;
let converged = false;
let iterationCount = 0;

// Edge wireframe thickness scales with mesh density
let edgeRadius = 0.3;

// --- Mesh generators ---

const GENERATORS = {
  sphere:     (n) => icosphere(n),
  torus:      (n) => torus(n),
  cube:       (n) => cube(n),
  cylinder:   (n) => cylinder(n),
  grid:       (n) => gridSheet(n),
};

// --- Layout setup ---

function setupLayout(meshType, detail, idealLength) {
  const gen = GENERATORS[meshType] || GENERATORS.sphere;
  const mesh = gen(detail);

  nodeCount = mesh.nodeCount;
  edges = mesh.edges;
  faces = mesh.faces;

  // Scale wireframe thickness inversely with density
  const DENSITY_REFERENCE = 40;
  const scaleFactor = Math.sqrt(DENSITY_REFERENCE / Math.max(1, nodeCount));
  const BASE_EDGE_RADIUS = 0.4;
  edgeRadius = BASE_EDGE_RADIUS * Math.max(0.15, scaleFactor);

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

  updateGPU(false);
}

function stepLayout() {
  if (!layout || converged) return;

  layout.stepOnce();
  iterationCount++;

  // Bulk position read — single WASM call, no per-node overhead
  const posArray = layout.getPositions();
  for (let i = 0; i < posArray.length; i++) {
    positions[i] = posArray[i];
  }

  const MAX_ITERATIONS = 500;
  if (iterationCount >= MAX_ITERATIONS) {
    converged = true;
  }

  updateGPU(converged);
}

function updateGPU(includeEdges) {
  // Filled triangular facets
  const { data: faceData, vertexCount } = buildFaceVertices(
    positions, faces, FACE_COLOR_COOL, FACE_COLOR_WARM,
  );
  renderer.updateFaces(faceData, vertexCount);

  // Wireframe edges: skip during animation (expensive to rebuild), show when converged
  if (includeEdges) {
    const edgeData = buildEdgeInstances(positions, edges, edgeRadius, EDGE_COLOR);
    renderer.updateEdgeInstances(edgeData, edges.length);
  } else {
    renderer.updateEdgeInstances(new Float32Array(0), 0);
  }

  renderer.updateNodeInstances(new Float32Array(0), 0);
}

// --- Main ---

async function main() {
  const canvas = document.getElementById('canvas');
  const overlay = document.getElementById('overlay');
  const statusEl = document.getElementById('status');

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
  overlay.style.display = 'none';

  // Controls
  const meshSelect = document.getElementById('meshType');
  const detailSlider = document.getElementById('detail');
  const detailLabel = document.getElementById('detailLabel');
  const lengthSlider = document.getElementById('idealLength');
  const lengthLabel = document.getElementById('idealLengthLabel');
  const delaySlider = document.getElementById('delay');
  const delayLabel = document.getElementById('delayLabel');
  const playBtn = document.getElementById('playBtn');
  const resetBtn = document.getElementById('resetBtn');

  let iterationDelayMs = 0;

  detailSlider.value = DEFAULT_DETAIL;
  detailLabel.textContent = DEFAULT_DETAIL;
  lengthSlider.value = DEFAULT_IDEAL_EDGE_LENGTH;
  lengthLabel.textContent = DEFAULT_IDEAL_EDGE_LENGTH;
  delaySlider.value = 0;
  delayLabel.textContent = '0ms';

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
  delaySlider.addEventListener('input', () => {
    iterationDelayMs = parseInt(delaySlider.value);
    delayLabel.textContent = iterationDelayMs + 'ms';
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
  let lastStepTime = 0;
  const MIN_FRAME_MS = 16;

  function frame(time) {
    const sinceLast = time - lastStepTime;
    const stepInterval = Math.max(MIN_FRAME_MS, iterationDelayMs);

    if (running && !converged && sinceLast >= stepInterval) {
      lastStepTime = time;
      stepLayout();
    }

    if (converged) {
      statusEl.textContent = `Converged (${iterationCount} iters) — ${nodeCount} verts, ${faces.length} faces`;
      playBtn.textContent = 'Restart';
    } else if (running) {
      statusEl.textContent = `Iter ${iterationCount} — ${nodeCount} verts, ${faces.length} faces`;
    } else {
      statusEl.textContent = `Paused (${iterationCount}) — ${nodeCount} verts, ${faces.length} faces`;
    }

    renderer.updateUniforms(camera);
    renderer.render();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main();

// 3D WebGPU demo: graph layout with libcola WASM solver.

import { Renderer, buildNodeInstances, buildEdgeInstances } from './renderer.js';
import { OrbitCamera } from './camera.js';

const NODE_RADIUS = 6;
const EDGE_RADIUS = 1.5;
const DEFAULT_IDEAL_EDGE_LENGTH = 60;
const DEFAULT_NODE_COUNT = 30;
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

// --- Graph generators ---

function randomGraph(n, edgeProbability = 0.08) {
  const pos = [];
  const edgeList = [];
  const SPREAD = 200;
  for (let i = 0; i < n; i++) {
    pos.push(
      (Math.random() - 0.5) * SPREAD,
      (Math.random() - 0.5) * SPREAD,
      (Math.random() - 0.5) * SPREAD,
    );
  }
  // Ensure connected: chain
  for (let i = 1; i < n; i++) {
    edgeList.push([i - 1, i]);
  }
  // Random extra edges
  for (let i = 0; i < n; i++) {
    for (let j = i + 2; j < n; j++) {
      if (Math.random() < edgeProbability) edgeList.push([i, j]);
    }
  }
  return { positions: pos, edges: edgeList, count: n };
}

function gridGraph(side) {
  const n = side * side;
  const pos = [];
  const edgeList = [];
  const SPREAD = 200;
  for (let i = 0; i < n; i++) {
    pos.push(
      (Math.random() - 0.5) * SPREAD,
      (Math.random() - 0.5) * SPREAD,
      (Math.random() - 0.5) * SPREAD,
    );
  }
  for (let r = 0; r < side; r++) {
    for (let c = 0; c < side; c++) {
      const idx = r * side + c;
      if (c + 1 < side) edgeList.push([idx, idx + 1]);
      if (r + 1 < side) edgeList.push([idx, idx + side]);
    }
  }
  return { positions: pos, edges: edgeList, count: n };
}

function treeGraph(n) {
  const pos = [];
  const edgeList = [];
  const SPREAD = 300;
  for (let i = 0; i < n; i++) {
    pos.push(
      (Math.random() - 0.5) * SPREAD,
      (Math.random() - 0.5) * SPREAD,
      (Math.random() - 0.5) * SPREAD,
    );
  }
  for (let i = 1; i < n; i++) {
    const parent = Math.floor(Math.random() * i);
    edgeList.push([parent, i]);
  }
  return { positions: pos, edges: edgeList, count: n };
}

function cycleGraph(n) {
  const pos = [];
  const edgeList = [];
  // Spherical initialization
  for (let i = 0; i < n; i++) {
    const phi = Math.acos(1 - 2 * (i + 0.5) / n);
    const theta = Math.PI * (1 + Math.sqrt(5)) * i;
    const SPHERE_RADIUS = 100;
    pos.push(
      SPHERE_RADIUS * Math.sin(phi) * Math.cos(theta),
      SPHERE_RADIUS * Math.cos(phi),
      SPHERE_RADIUS * Math.sin(phi) * Math.sin(theta),
    );
  }
  for (let i = 0; i < n; i++) {
    edgeList.push([i, (i + 1) % n]);
  }
  return { positions: pos, edges: edgeList, count: n };
}

function denseGraph(n) {
  const pos = [];
  const edgeList = [];
  const SPREAD = 150;
  for (let i = 0; i < n; i++) {
    pos.push(
      (Math.random() - 0.5) * SPREAD,
      (Math.random() - 0.5) * SPREAD,
      (Math.random() - 0.5) * SPREAD,
    );
  }
  const TARGET_DEGREE = 4;
  // Chain first
  for (let i = 1; i < n; i++) edgeList.push([i - 1, i]);
  // Random edges to target degree
  const degree = new Array(n).fill(0);
  for (const [a, b] of edgeList) { degree[a]++; degree[b]++; }
  for (let i = 0; i < n; i++) {
    while (degree[i] < TARGET_DEGREE) {
      const j = Math.floor(Math.random() * n);
      if (j !== i) {
        edgeList.push([i, j]);
        degree[i]++;
        degree[j]++;
      }
    }
  }
  return { positions: pos, edges: edgeList, count: n };
}

const GENERATORS = {
  random: (n) => randomGraph(n),
  grid: (n) => gridGraph(Math.max(2, Math.round(Math.sqrt(n)))),
  tree: (n) => treeGraph(n),
  cycle: (n) => cycleGraph(n),
  dense: (n) => denseGraph(n),
};

// --- Layout setup ---

function setupLayout(graphType, n, idealLength) {
  const gen = GENERATORS[graphType] || GENERATORS.random;
  const graph = gen(n);

  positions = graph.positions;
  edges = graph.edges;
  nodeCount = graph.count;
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

  // Extract positions from result
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
  const nodeData = buildNodeInstances(positions, nodeCount, NODE_RADIUS, colors);
  renderer.updateNodeInstances(nodeData, nodeCount);

  const edgeData = buildEdgeInstances(positions, edges, EDGE_RADIUS, EDGE_COLOR);
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
  const graphSelect = document.getElementById('graphType');
  const nodeSlider = document.getElementById('nodeCount');
  const nodeLabel = document.getElementById('nodeCountLabel');
  const lengthSlider = document.getElementById('idealLength');
  const lengthLabel = document.getElementById('idealLengthLabel');
  const playBtn = document.getElementById('playBtn');
  const resetBtn = document.getElementById('resetBtn');

  nodeSlider.value = DEFAULT_NODE_COUNT;
  nodeLabel.textContent = DEFAULT_NODE_COUNT;
  lengthSlider.value = DEFAULT_IDEAL_EDGE_LENGTH;
  lengthLabel.textContent = DEFAULT_IDEAL_EDGE_LENGTH;

  function resetGraph() {
    setupLayout(
      graphSelect.value,
      parseInt(nodeSlider.value),
      parseInt(lengthSlider.value),
    );
    running = true;
    playBtn.textContent = 'Pause';
  }

  nodeSlider.addEventListener('input', () => {
    nodeLabel.textContent = nodeSlider.value;
  });
  lengthSlider.addEventListener('input', () => {
    lengthLabel.textContent = lengthSlider.value;
  });

  graphSelect.addEventListener('change', resetGraph);
  resetBtn.addEventListener('click', resetGraph);

  playBtn.addEventListener('click', () => {
    if (converged) {
      // Restart
      resetGraph();
    } else {
      running = !running;
      playBtn.textContent = running ? 'Pause' : 'Play';
    }
  });

  // Initial graph
  resetGraph();

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
      statusEl.textContent = `Converged (${iterationCount} iterations)`;
      playBtn.textContent = 'Restart';
    } else if (running) {
      statusEl.textContent = `Iteration ${iterationCount}`;
    } else {
      statusEl.textContent = `Paused (${iterationCount})`;
    }

    renderer.updateUniforms(camera);
    renderer.render();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main();

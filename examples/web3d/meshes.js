// Mesh generators for 3D shape reconstruction demo.
// Each generator returns:
//   positions: Float64Array (flat xyz), edges: [number,number][],
//   faces: [number,number,number][], nodeCount: number
// Positions are at unit scale (edge length ≈ 1). Caller scales by idealEdgeLength.

const TAU = Math.PI * 2;
const PHI = (1 + Math.sqrt(5)) / 2; // golden ratio

// --- Icosphere (subdivided icosahedron) ---

export function icosphere(targetNodes) {
  // Pick subdivision level closest to target
  // Level 0: 12, 1: 42, 2: 162, 3: 642
  let subdiv = 0;
  if (targetNodes >= 30) subdiv = 1;
  if (targetNodes >= 100) subdiv = 2;
  if (targetNodes >= 400) subdiv = 3;

  // Base icosahedron vertices (unit sphere)
  let vertices = [
    [-1, PHI, 0], [1, PHI, 0], [-1, -PHI, 0], [1, -PHI, 0],
    [0, -1, PHI], [0, 1, PHI], [0, -1, -PHI], [0, 1, -PHI],
    [PHI, 0, -1], [PHI, 0, 1], [-PHI, 0, -1], [-PHI, 0, 1],
  ].map(v => {
    const l = Math.hypot(...v);
    return [v[0] / l, v[1] / l, v[2] / l];
  });

  let faces = [
    [0,11,5], [0,5,1], [0,1,7], [0,7,10], [0,10,11],
    [1,5,9], [5,11,4], [11,10,2], [10,7,6], [7,1,8],
    [3,9,4], [3,4,2], [3,2,6], [3,6,8], [3,8,9],
    [4,9,5], [2,4,11], [6,2,10], [8,6,7], [9,8,1],
  ];

  for (let s = 0; s < subdiv; s++) {
    const cache = {};
    function midpoint(a, b) {
      const key = Math.min(a, b) + '_' + Math.max(a, b);
      if (cache[key] !== undefined) return cache[key];
      const va = vertices[a], vb = vertices[b];
      const m = [(va[0]+vb[0])/2, (va[1]+vb[1])/2, (va[2]+vb[2])/2];
      const l = Math.hypot(...m);
      vertices.push([m[0]/l, m[1]/l, m[2]/l]);
      return (cache[key] = vertices.length - 1);
    }
    const next = [];
    for (const [a, b, c] of faces) {
      const ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
      next.push([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]);
    }
    faces = next;
  }

  // Extract unique edges
  const edgeSet = new Set();
  const edges = [];
  for (const [a, b, c] of faces) {
    for (const [i, j] of [[a, b], [b, c], [c, a]]) {
      const key = Math.min(i, j) * vertices.length + Math.max(i, j);
      if (!edgeSet.has(key)) {
        edgeSet.add(key);
        edges.push([i, j]);
      }
    }
  }

  // Flatten positions
  const positions = new Float64Array(vertices.length * 3);
  for (let i = 0; i < vertices.length; i++) {
    positions[i * 3] = vertices[i][0];
    positions[i * 3 + 1] = vertices[i][1];
    positions[i * 3 + 2] = vertices[i][2];
  }

  return { positions, edges, faces, nodeCount: vertices.length };
}

// --- Torus mesh ---

export function torus(targetNodes) {
  const MAJOR_RADIUS = 2.0;
  const MINOR_RADIUS = 0.7;

  // Pick rings/segments to approximate target node count
  const side = Math.max(4, Math.round(Math.sqrt(targetNodes)));
  const rings = side;
  const segments = side;

  const n = rings * segments;
  const positions = new Float64Array(n * 3);

  for (let i = 0; i < rings; i++) {
    const u = (i / rings) * TAU;
    for (let j = 0; j < segments; j++) {
      const v = (j / segments) * TAU;
      const idx = i * segments + j;
      positions[idx * 3]     = (MAJOR_RADIUS + MINOR_RADIUS * Math.cos(v)) * Math.cos(u);
      positions[idx * 3 + 1] = (MAJOR_RADIUS + MINOR_RADIUS * Math.cos(v)) * Math.sin(u);
      positions[idx * 3 + 2] = MINOR_RADIUS * Math.sin(v);
    }
  }

  const edges = [];
  const faces = [];
  for (let i = 0; i < rings; i++) {
    for (let j = 0; j < segments; j++) {
      const curr = i * segments + j;
      const nextJ = i * segments + (j + 1) % segments;
      const nextI = ((i + 1) % rings) * segments + j;
      const diag = ((i + 1) % rings) * segments + (j + 1) % segments;
      edges.push([curr, nextJ]);
      edges.push([curr, nextI]);
      // Two triangles per quad
      faces.push([curr, nextI, diag]);
      faces.push([curr, diag, nextJ]);
    }
  }

  return { positions, edges, faces, nodeCount: n };
}

// --- Cube mesh (subdivided faces) ---

export function cube(targetNodes) {
  // n ≈ 6d² + 2 → d ≈ sqrt((n-2)/6)
  const d = Math.max(1, Math.round(Math.sqrt(Math.max(1, targetNodes - 2) / 6)));

  const vertexMap = new Map();
  const vertices = [];
  const edgeSet = new Set();
  const edges = [];
  const faces = [];

  function addVertex(x, y, z) {
    const PRECISION = 1e6;
    const key = `${Math.round(x*PRECISION)},${Math.round(y*PRECISION)},${Math.round(z*PRECISION)}`;
    if (vertexMap.has(key)) return vertexMap.get(key);
    const idx = vertices.length;
    vertices.push([x, y, z]);
    vertexMap.set(key, idx);
    return idx;
  }

  function addEdge(a, b) {
    const key = Math.min(a, b) * 100000 + Math.max(a, b);
    if (!edgeSet.has(key)) {
      edgeSet.add(key);
      edges.push([a, b]);
    }
  }

  // 6 faces: each defined by fixed axis, fixed value, and two varying axes
  const faceConfigs = [
    { fixed: 0, val:  1, u: 1, v: 2 },  // +X
    { fixed: 0, val: -1, u: 2, v: 1 },  // -X
    { fixed: 1, val:  1, u: 0, v: 2 },  // +Y
    { fixed: 1, val: -1, u: 2, v: 0 },  // -Y
    { fixed: 2, val:  1, u: 0, v: 1 },  // +Z
    { fixed: 2, val: -1, u: 1, v: 0 },  // -Z
  ];

  for (const fc of faceConfigs) {
    const grid = [];
    for (let i = 0; i <= d; i++) {
      grid[i] = [];
      for (let j = 0; j <= d; j++) {
        const pos = [0, 0, 0];
        pos[fc.fixed] = fc.val;
        pos[fc.u] = -1 + (2 * i / d);
        pos[fc.v] = -1 + (2 * j / d);
        grid[i][j] = addVertex(pos[0], pos[1], pos[2]);
      }
    }
    for (let i = 0; i <= d; i++) {
      for (let j = 0; j <= d; j++) {
        if (i < d) addEdge(grid[i][j], grid[i + 1][j]);
        if (j < d) addEdge(grid[i][j], grid[i][j + 1]);
        if (i < d && j < d) {
          faces.push([grid[i][j], grid[i + 1][j], grid[i + 1][j + 1]]);
          faces.push([grid[i][j], grid[i + 1][j + 1], grid[i][j + 1]]);
        }
      }
    }
  }

  const positions = new Float64Array(vertices.length * 3);
  for (let i = 0; i < vertices.length; i++) {
    positions[i * 3]     = vertices[i][0];
    positions[i * 3 + 1] = vertices[i][1];
    positions[i * 3 + 2] = vertices[i][2];
  }

  return { positions, edges, faces, nodeCount: vertices.length };
}

// --- Flat grid sheet ---

export function gridSheet(targetNodes) {
  const side = Math.max(2, Math.round(Math.sqrt(targetNodes)));
  const n = side * side;
  const positions = new Float64Array(n * 3);

  for (let r = 0; r < side; r++) {
    for (let c = 0; c < side; c++) {
      const idx = r * side + c;
      positions[idx * 3]     = (c / (side - 1)) * 2 - 1;
      positions[idx * 3 + 1] = (r / (side - 1)) * 2 - 1;
      positions[idx * 3 + 2] = 0;
    }
  }

  const edges = [];
  const faces = [];
  for (let r = 0; r < side; r++) {
    for (let c = 0; c < side; c++) {
      const idx = r * side + c;
      if (c + 1 < side) edges.push([idx, idx + 1]);
      if (r + 1 < side) edges.push([idx, idx + side]);
      if (c + 1 < side && r + 1 < side) {
        faces.push([idx, idx + side, idx + side + 1]);
        faces.push([idx, idx + side + 1, idx + 1]);
      }
    }
  }

  return { positions, edges, faces, nodeCount: n };
}

// --- Octahedron ---

export function octahedron() {
  // Vertices: +X, -X, +Y, -Y, +Z, -Z
  const positions = new Float64Array([
     1, 0, 0,   -1, 0, 0,
     0, 1, 0,    0,-1, 0,
     0, 0, 1,    0, 0,-1,
  ]);
  const edges = [
    [0,2],[0,3],[0,4],[0,5],
    [1,2],[1,3],[1,4],[1,5],
    [2,4],[2,5],[3,4],[3,5],
  ];
  const faces = [
    [0,2,4], [0,4,3], [0,3,5], [0,5,2],
    [1,4,2], [1,3,4], [1,5,3], [1,2,5],
  ];
  return { positions, edges, faces, nodeCount: 6 };
}

// --- Cylinder / tube ---

export function cylinder(targetNodes) {
  const segments = Math.max(4, Math.round(Math.sqrt(targetNodes)));
  const rings = segments;
  const n = rings * segments;
  const positions = new Float64Array(n * 3);
  // Match axial spacing to ring spacing for uniform edge lengths
  const RADIUS = 1.0;
  const ringEdgeLen = TAU * RADIUS / segments;
  const height = ringEdgeLen * (rings - 1);

  for (let i = 0; i < rings; i++) {
    const y = (i / (rings - 1)) * height - height / 2;
    for (let j = 0; j < segments; j++) {
      const angle = (j / segments) * TAU;
      const idx = i * segments + j;
      positions[idx * 3]     = RADIUS * Math.cos(angle);
      positions[idx * 3 + 1] = y;
      positions[idx * 3 + 2] = RADIUS * Math.sin(angle);
    }
  }

  const edges = [];
  const faces = [];
  for (let i = 0; i < rings; i++) {
    for (let j = 0; j < segments; j++) {
      const curr = i * segments + j;
      const nextJ = i * segments + (j + 1) % segments;
      edges.push([curr, nextJ]);
      if (i + 1 < rings) {
        const below = (i + 1) * segments + j;
        const belowNext = (i + 1) * segments + (j + 1) % segments;
        edges.push([curr, below]);
        faces.push([curr, below, belowNext]);
        faces.push([curr, belowNext, nextJ]);
      }
    }
  }

  return { positions, edges, faces, nodeCount: n };
}

// --- Utility: scramble positions into a random ball ---

export function scramble(nodeCount, radius) {
  const positions = [];
  for (let i = 0; i < nodeCount; i++) {
    // Uniform random in sphere
    const u = Math.random();
    const v = Math.random();
    const theta = TAU * u;
    const phi = Math.acos(2 * v - 1);
    const r = radius * Math.cbrt(Math.random());
    positions.push(
      r * Math.sin(phi) * Math.cos(theta),
      r * Math.sin(phi) * Math.sin(theta),
      r * Math.cos(phi),
    );
  }
  return positions;
}

// --- Compute mean edge length of a mesh ---

export function meanEdgeLength(positions, edges) {
  let sum = 0;
  for (const [a, b] of edges) {
    const dx = positions[a*3] - positions[b*3];
    const dy = positions[a*3+1] - positions[b*3+1];
    const dz = positions[a*3+2] - positions[b*3+2];
    sum += Math.hypot(dx, dy, dz);
  }
  return sum / edges.length;
}

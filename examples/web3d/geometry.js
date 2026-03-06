// Unit sphere and cylinder mesh generators for instanced rendering.

const TAU = Math.PI * 2;

/**
 * Create a UV sphere (positions, normals, indices).
 * @param {number} rings - latitude divisions
 * @param {number} segments - longitude divisions
 * @returns {{ positions: Float32Array, normals: Float32Array, indices: Uint16Array }}
 */
export function createSphere(rings = 16, segments = 24) {
  const verts = [];
  const norms = [];
  const idx = [];

  for (let r = 0; r <= rings; r++) {
    const theta = (r / rings) * Math.PI;
    const sinT = Math.sin(theta);
    const cosT = Math.cos(theta);
    for (let s = 0; s <= segments; s++) {
      const phi = (s / segments) * TAU;
      const x = sinT * Math.cos(phi);
      const y = cosT;
      const z = sinT * Math.sin(phi);
      verts.push(x, y, z);
      norms.push(x, y, z);
    }
  }

  for (let r = 0; r < rings; r++) {
    for (let s = 0; s < segments; s++) {
      const a = r * (segments + 1) + s;
      const b = a + segments + 1;
      idx.push(a, b, a + 1, b, b + 1, a + 1);
    }
  }

  return {
    positions: new Float32Array(verts),
    normals: new Float32Array(norms),
    indices: new Uint16Array(idx),
  };
}

/**
 * Create a unit cylinder along Y axis (from y=0 to y=1, radius=1).
 * @param {number} segments - radial divisions
 * @returns {{ positions: Float32Array, normals: Float32Array, indices: Uint16Array }}
 */
export function createCylinder(segments = 12) {
  const verts = [];
  const norms = [];
  const idx = [];

  // Side vertices: two rings at y=0 and y=1
  for (let ring = 0; ring <= 1; ring++) {
    const y = ring;
    for (let s = 0; s <= segments; s++) {
      const phi = (s / segments) * TAU;
      const x = Math.cos(phi);
      const z = Math.sin(phi);
      verts.push(x, y, z);
      norms.push(x, 0, z); // radial normal
    }
  }

  const stride = segments + 1;
  for (let s = 0; s < segments; s++) {
    const a = s;
    const b = a + stride;
    idx.push(a, b, a + 1, b, b + 1, a + 1);
  }

  return {
    positions: new Float32Array(verts),
    normals: new Float32Array(norms),
    indices: new Uint16Array(idx),
  };
}

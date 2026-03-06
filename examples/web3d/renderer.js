// WebGPU renderer with face (triangle), edge (instanced cylinder), and optional node (instanced sphere) pipelines.

import { createSphere, createCylinder } from './geometry.js';

const UNIFORM_BUFFER_SIZE = 256; // aligned to 256 for WebGPU
const FLOAT_SIZE = 4;
const INSTANCE_FLOATS = 20; // mat4 (16) + color (4)
const INSTANCE_STRIDE = INSTANCE_FLOATS * FLOAT_SIZE;

// Face vertex: position(3) + normal(3) + color(3) = 9 floats
const FACE_VERTEX_FLOATS = 9;
const FACE_VERTEX_STRIDE = FACE_VERTEX_FLOATS * FLOAT_SIZE;

const INITIAL_MAX_NODES = 512;
const INITIAL_MAX_EDGES = 1024;
const INITIAL_MAX_FACE_VERTS = 4096;

const CLEAR_COLOR = { r: 0.059, g: 0.090, b: 0.165, a: 1.0 }; // #0f172a

export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.device = null;
    this.context = null;

    // Pipelines
    this.nodePipeline = null;
    this.edgePipeline = null;
    this.facePipeline = null;

    // Geometry (for instanced rendering)
    this.sphereGeo = null;
    this.cylinderGeo = null;

    // Instance buffers (nodes + edges)
    this.nodeInstanceBuffer = null;
    this.edgeInstanceBuffer = null;
    this.maxNodes = INITIAL_MAX_NODES;
    this.maxEdges = INITIAL_MAX_EDGES;
    this.nodeCount = 0;
    this.edgeCount = 0;

    // Face vertex buffer (direct triangles)
    this.faceVertexBuffer = null;
    this.maxFaceVerts = INITIAL_MAX_FACE_VERTS;
    this.faceVertexCount = 0;

    // Uniform buffer + bind group
    this.uniformBuffer = null;
    this.uniformBindGroup = null;

    // Depth texture
    this.depthTexture = null;
  }

  async init() {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('WebGPU not supported');
    this.device = await adapter.requestDevice();

    this.context = this.canvas.getContext('webgpu');
    this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format: this.presentationFormat,
      alphaMode: 'premultiplied',
    });

    await this._createResources();
  }

  async _createResources() {
    const device = this.device;

    // Shader module
    const shaderCode = await fetch('shaders.wgsl').then(r => r.text());
    const shaderModule = device.createShaderModule({ code: shaderCode });

    // Geometry for instanced rendering
    this.sphereGeo = this._uploadGeometry(createSphere(16, 24));
    this.cylinderGeo = this._uploadGeometry(createCylinder(12));

    // Uniform buffer
    this.uniformBuffer = device.createBuffer({
      size: UNIFORM_BUFFER_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Bind group layout
    const bindGroupLayout = device.createBindGroupLayout({
      entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' },
      }],
    });

    this.uniformBindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }],
    });

    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    });

    // Instance buffers
    this.nodeInstanceBuffer = this._createInstanceBuffer(this.maxNodes);
    this.edgeInstanceBuffer = this._createInstanceBuffer(this.maxEdges);

    // Face vertex buffer
    this.faceVertexBuffer = this._createFaceBuffer(this.maxFaceVerts);

    // --- Instanced vertex buffer layouts (nodes + edges) ---
    const geometryBufferLayout = [
      { // positions
        arrayStride: 3 * FLOAT_SIZE,
        stepMode: 'vertex',
        attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }],
      },
      { // normals
        arrayStride: 3 * FLOAT_SIZE,
        stepMode: 'vertex',
        attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x3' }],
      },
    ];

    const instanceBufferLayout = {
      arrayStride: INSTANCE_STRIDE,
      stepMode: 'instance',
      attributes: [
        { shaderLocation: 3, offset: 0, format: 'float32x4' },
        { shaderLocation: 4, offset: 4 * FLOAT_SIZE, format: 'float32x4' },
        { shaderLocation: 5, offset: 8 * FLOAT_SIZE, format: 'float32x4' },
        { shaderLocation: 6, offset: 12 * FLOAT_SIZE, format: 'float32x4' },
        { shaderLocation: 7, offset: 16 * FLOAT_SIZE, format: 'float32x4' },
      ],
    };

    // --- Face vertex buffer layout (interleaved pos+normal+color) ---
    const faceBufferLayout = [{
      arrayStride: FACE_VERTEX_STRIDE,
      stepMode: 'vertex',
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x3' },                  // position
        { shaderLocation: 1, offset: 3 * FLOAT_SIZE, format: 'float32x3' },     // normal
        { shaderLocation: 2, offset: 6 * FLOAT_SIZE, format: 'float32x3' },     // color
      ],
    }];

    const depthStencil = {
      depthWriteEnabled: true,
      depthCompare: 'less',
      format: 'depth24plus',
    };

    // Face pipeline (solid triangles)
    this.facePipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_face',
        buffers: faceBufferLayout,
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_face',
        targets: [{ format: this.presentationFormat }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil,
    });

    // Node pipeline (instanced spheres)
    this.nodePipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_node',
        buffers: [...geometryBufferLayout, instanceBufferLayout],
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_node',
        targets: [{ format: this.presentationFormat }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil,
    });

    // Edge pipeline (instanced cylinders)
    this.edgePipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_edge',
        buffers: [...geometryBufferLayout, instanceBufferLayout],
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_edge',
        targets: [{ format: this.presentationFormat }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil,
    });

    this._createDepthTexture();
  }

  _uploadGeometry(geo) {
    const device = this.device;
    const posBuf = device.createBuffer({
      size: geo.positions.byteLength,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });
    new Float32Array(posBuf.getMappedRange()).set(geo.positions);
    posBuf.unmap();

    const normBuf = device.createBuffer({
      size: geo.normals.byteLength,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });
    new Float32Array(normBuf.getMappedRange()).set(geo.normals);
    normBuf.unmap();

    const idxBuf = device.createBuffer({
      size: geo.indices.byteLength,
      usage: GPUBufferUsage.INDEX,
      mappedAtCreation: true,
    });
    new Uint16Array(idxBuf.getMappedRange()).set(geo.indices);
    idxBuf.unmap();

    return {
      positionBuffer: posBuf,
      normalBuffer: normBuf,
      indexBuffer: idxBuf,
      indexCount: geo.indices.length,
    };
  }

  _createInstanceBuffer(maxInstances) {
    return this.device.createBuffer({
      size: maxInstances * INSTANCE_STRIDE,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
  }

  _createFaceBuffer(maxVerts) {
    return this.device.createBuffer({
      size: maxVerts * FACE_VERTEX_STRIDE,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
  }

  _createDepthTexture() {
    if (this.depthTexture) this.depthTexture.destroy();
    this.depthTexture = this.device.createTexture({
      size: [this.canvas.width, this.canvas.height],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  resize(w, h) {
    this.canvas.width = w;
    this.canvas.height = h;
    this._createDepthTexture();
  }

  /** Upload face vertex data: interleaved [pos(3), normal(3), color(3)] per vertex. */
  updateFaces(data, vertexCount) {
    this.faceVertexCount = vertexCount;
    if (vertexCount > this.maxFaceVerts) {
      this.maxFaceVerts = Math.max(vertexCount, this.maxFaceVerts * 2);
      this.faceVertexBuffer.destroy();
      this.faceVertexBuffer = this._createFaceBuffer(this.maxFaceVerts);
    }
    if (vertexCount > 0) {
      this.device.queue.writeBuffer(this.faceVertexBuffer, 0, data, 0, vertexCount * FACE_VERTEX_FLOATS);
    }
  }

  /** Upload per-node instance data. */
  updateNodeInstances(data, count) {
    this.nodeCount = count;
    if (count > this.maxNodes) {
      this.maxNodes = Math.max(count, this.maxNodes * 2);
      this.nodeInstanceBuffer.destroy();
      this.nodeInstanceBuffer = this._createInstanceBuffer(this.maxNodes);
    }
    if (count > 0) {
      this.device.queue.writeBuffer(this.nodeInstanceBuffer, 0, data, 0, count * INSTANCE_FLOATS);
    }
  }

  /** Upload per-edge instance data. */
  updateEdgeInstances(data, count) {
    this.edgeCount = count;
    if (count > this.maxEdges) {
      this.maxEdges = Math.max(count, this.maxEdges * 2);
      this.edgeInstanceBuffer.destroy();
      this.edgeInstanceBuffer = this._createInstanceBuffer(this.maxEdges);
    }
    if (count > 0) {
      this.device.queue.writeBuffer(this.edgeInstanceBuffer, 0, data, 0, count * INSTANCE_FLOATS);
    }
  }

  /** Upload uniform data (viewProj, eye, lights). */
  updateUniforms(camera) {
    const vp = camera.viewProjMatrix();
    const eye = camera.eye;

    const LIGHT_DIR = [0.48, 0.64, 0.6];
    const LIGHT_COLOR = [1.0, 0.97, 0.92];
    const AMBIENT_COLOR = [0.28, 0.32, 0.45];

    const buf = new Float32Array(UNIFORM_BUFFER_SIZE / FLOAT_SIZE);
    buf.set(vp, 0);
    buf.set(eye, 16);
    buf.set(LIGHT_DIR, 20);
    buf.set(LIGHT_COLOR, 24);
    buf.set(AMBIENT_COLOR, 28);

    this.device.queue.writeBuffer(this.uniformBuffer, 0, buf);
  }

  render() {
    const commandEncoder = this.device.createCommandEncoder();
    const textureView = this.context.getCurrentTexture().createView();

    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: textureView,
        clearValue: CLEAR_COLOR,
        loadOp: 'clear',
        storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: this.depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });

    renderPass.setBindGroup(0, this.uniformBindGroup);

    // 1. Draw filled faces first
    if (this.faceVertexCount > 0) {
      renderPass.setPipeline(this.facePipeline);
      renderPass.setVertexBuffer(0, this.faceVertexBuffer);
      renderPass.draw(this.faceVertexCount);
    }

    // 2. Draw edges (thin wireframe overlay)
    if (this.edgeCount > 0) {
      renderPass.setPipeline(this.edgePipeline);
      renderPass.setVertexBuffer(0, this.cylinderGeo.positionBuffer);
      renderPass.setVertexBuffer(1, this.cylinderGeo.normalBuffer);
      renderPass.setVertexBuffer(2, this.edgeInstanceBuffer);
      renderPass.setIndexBuffer(this.cylinderGeo.indexBuffer, 'uint16');
      renderPass.drawIndexed(this.cylinderGeo.indexCount, this.edgeCount);
    }

    // 3. Draw nodes (optional, for vertex dots)
    if (this.nodeCount > 0) {
      renderPass.setPipeline(this.nodePipeline);
      renderPass.setVertexBuffer(0, this.sphereGeo.positionBuffer);
      renderPass.setVertexBuffer(1, this.sphereGeo.normalBuffer);
      renderPass.setVertexBuffer(2, this.nodeInstanceBuffer);
      renderPass.setIndexBuffer(this.sphereGeo.indexBuffer, 'uint16');
      renderPass.drawIndexed(this.sphereGeo.indexCount, this.nodeCount);
    }

    renderPass.end();
    this.device.queue.submit([commandEncoder.finish()]);
  }
}

// --- Face vertex data builder ---

const VERTS_PER_TRIANGLE = 3;

/**
 * Build face vertex data with flat normals.
 * @param {number[]} positions - flat [x,y,z, ...] from solver
 * @param {number[][]} faces - [[a,b,c], ...] triangle index triplets
 * @param {number[]} baseColor - [r,g,b] base face color
 * @param {number[]} warmColor - [r,g,b] color for upward-facing faces
 * @returns {{ data: Float32Array, vertexCount: number }}
 */
export function buildFaceVertices(positions, faces, baseColor, warmColor) {
  const vertexCount = faces.length * VERTS_PER_TRIANGLE;
  const data = new Float32Array(vertexCount * FACE_VERTEX_FLOATS);

  for (let f = 0; f < faces.length; f++) {
    const [ia, ib, ic] = faces[f];
    const ax = positions[ia * 3], ay = positions[ia * 3 + 1], az = positions[ia * 3 + 2];
    const bx = positions[ib * 3], by = positions[ib * 3 + 1], bz = positions[ib * 3 + 2];
    const cx = positions[ic * 3], cy = positions[ic * 3 + 1], cz = positions[ic * 3 + 2];

    // Flat normal = cross(B-A, C-A)
    const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
    const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
    let nx = e1y * e2z - e1z * e2y;
    let ny = e1z * e2x - e1x * e2z;
    let nz = e1x * e2y - e1y * e2x;
    const nl = Math.hypot(nx, ny, nz) || 1;
    nx /= nl; ny /= nl; nz /= nl;

    // Color: blend based on normal direction for subtle variation
    const t = ny * 0.5 + 0.5; // 0 = downward, 1 = upward
    const cr = baseColor[0] * (1 - t) + warmColor[0] * t;
    const cg = baseColor[1] * (1 - t) + warmColor[1] * t;
    const cb = baseColor[2] * (1 - t) + warmColor[2] * t;

    // Write 3 vertices with same normal and color (flat shading)
    const verts = [[ax, ay, az], [bx, by, bz], [cx, cy, cz]];
    for (let v = 0; v < VERTS_PER_TRIANGLE; v++) {
      const off = (f * VERTS_PER_TRIANGLE + v) * FACE_VERTEX_FLOATS;
      data[off]     = verts[v][0];
      data[off + 1] = verts[v][1];
      data[off + 2] = verts[v][2];
      data[off + 3] = nx;
      data[off + 4] = ny;
      data[off + 5] = nz;
      data[off + 6] = cr;
      data[off + 7] = cg;
      data[off + 8] = cb;
    }
  }

  return { data, vertexCount };
}

// --- Instance data builders (for edges and optional vertex dots) ---

export function buildNodeInstances(positions, nodeCount, radius, colors) {
  const data = new Float32Array(nodeCount * INSTANCE_FLOATS);
  const singleColor = colors.length === 4 && typeof colors[0] === 'number';

  for (let i = 0; i < nodeCount; i++) {
    const off = i * INSTANCE_FLOATS;
    const px = positions[i * 3];
    const py = positions[i * 3 + 1];
    const pz = positions[i * 3 + 2];

    // Scale + translate
    data[off] = radius; data[off + 1] = 0; data[off + 2] = 0; data[off + 3] = 0;
    data[off + 4] = 0; data[off + 5] = radius; data[off + 6] = 0; data[off + 7] = 0;
    data[off + 8] = 0; data[off + 9] = 0; data[off + 10] = radius; data[off + 11] = 0;
    data[off + 12] = px; data[off + 13] = py; data[off + 14] = pz; data[off + 15] = 1;

    const c = singleColor ? colors : (colors[i] || [0.5, 0.5, 0.5, 1]);
    data[off + 16] = c[0]; data[off + 17] = c[1]; data[off + 18] = c[2]; data[off + 19] = c[3];
  }
  return data;
}

export function buildEdgeInstances(positions, edges, edgeRadius, color) {
  const data = new Float32Array(edges.length * INSTANCE_FLOATS);

  for (let i = 0; i < edges.length; i++) {
    const [src, dst] = edges[i];
    const off = i * INSTANCE_FLOATS;

    const ax = positions[src * 3], ay = positions[src * 3 + 1], az = positions[src * 3 + 2];
    const bx = positions[dst * 3], by = positions[dst * 3 + 1], bz = positions[dst * 3 + 2];

    const dx = bx - ax, dy = by - ay, dz = bz - az;
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-6;

    const yx = dx / len, yy = dy / len, yz = dz / len;

    const PARALLEL_THRESHOLD = 0.99;
    let xx, xy, xz;
    if (Math.abs(yy) < PARALLEL_THRESHOLD) {
      xx = yz; xy = 0; xz = -yx;
    } else {
      xx = 0; xy = -yz; xz = yy;
    }
    const xl = Math.sqrt(xx * xx + xy * xy + xz * xz) || 1e-6;
    xx /= xl; xy /= xl; xz /= xl;

    const zx = xy * yz - xz * yy;
    const zy = xz * yx - xx * yz;
    const zz = xx * yy - xy * yx;

    data[off] = xx * edgeRadius; data[off + 1] = xy * edgeRadius; data[off + 2] = xz * edgeRadius; data[off + 3] = 0;
    data[off + 4] = yx * len; data[off + 5] = yy * len; data[off + 6] = yz * len; data[off + 7] = 0;
    data[off + 8] = zx * edgeRadius; data[off + 9] = zy * edgeRadius; data[off + 10] = zz * edgeRadius; data[off + 11] = 0;
    data[off + 12] = ax; data[off + 13] = ay; data[off + 14] = az; data[off + 15] = 1;
    data[off + 16] = color[0]; data[off + 17] = color[1]; data[off + 18] = color[2]; data[off + 19] = color[3];
  }
  return data;
}

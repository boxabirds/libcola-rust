// WebGPU renderer with instanced sphere (node) and cylinder (edge) pipelines.

import { createSphere, createCylinder } from './geometry.js';

const UNIFORM_BUFFER_SIZE = 256; // aligned to 256 for WebGPU
const FLOAT_SIZE = 4;
const INSTANCE_FLOATS = 20; // mat4 (16) + color (4)
const INSTANCE_STRIDE = INSTANCE_FLOATS * FLOAT_SIZE;

const INITIAL_MAX_NODES = 512;
const INITIAL_MAX_EDGES = 1024;

const CLEAR_COLOR = { r: 0.059, g: 0.090, b: 0.165, a: 1.0 }; // #0f172a

export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.device = null;
    this.context = null;

    // Pipelines
    this.nodePipeline = null;
    this.edgePipeline = null;

    // Geometry
    this.sphereGeo = null;
    this.cylinderGeo = null;

    // Instance buffers
    this.nodeInstanceBuffer = null;
    this.edgeInstanceBuffer = null;
    this.maxNodes = INITIAL_MAX_NODES;
    this.maxEdges = INITIAL_MAX_EDGES;
    this.nodeCount = 0;
    this.edgeCount = 0;

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

    // Geometry
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

    // Vertex buffer layouts
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
        { shaderLocation: 3, offset: 0, format: 'float32x4' },              // col0
        { shaderLocation: 4, offset: 4 * FLOAT_SIZE, format: 'float32x4' }, // col1
        { shaderLocation: 5, offset: 8 * FLOAT_SIZE, format: 'float32x4' }, // col2
        { shaderLocation: 6, offset: 12 * FLOAT_SIZE, format: 'float32x4' },// col3
        { shaderLocation: 7, offset: 16 * FLOAT_SIZE, format: 'float32x4' },// color
      ],
    };

    const depthStencil = {
      depthWriteEnabled: true,
      depthCompare: 'less',
      format: 'depth24plus',
    };

    // Node pipeline
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

    // Edge pipeline
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

  /** Upload per-node instance data: transform mat4 (column-major) + rgba color. */
  updateNodeInstances(data, count) {
    this.nodeCount = count;
    if (count > this.maxNodes) {
      this.maxNodes = Math.max(count, this.maxNodes * 2);
      this.nodeInstanceBuffer.destroy();
      this.nodeInstanceBuffer = this._createInstanceBuffer(this.maxNodes);
    }
    this.device.queue.writeBuffer(this.nodeInstanceBuffer, 0, data, 0, count * INSTANCE_FLOATS);
  }

  /** Upload per-edge instance data: transform mat4 (column-major) + rgba color. */
  updateEdgeInstances(data, count) {
    this.edgeCount = count;
    if (count > this.maxEdges) {
      this.maxEdges = Math.max(count, this.maxEdges * 2);
      this.edgeInstanceBuffer.destroy();
      this.edgeInstanceBuffer = this._createInstanceBuffer(this.maxEdges);
    }
    this.device.queue.writeBuffer(this.edgeInstanceBuffer, 0, data, 0, count * INSTANCE_FLOATS);
  }

  /** Upload uniform data (viewProj, eye, lights). */
  updateUniforms(camera) {
    const vp = camera.viewProjMatrix();
    const eye = camera.eye;

    // Light direction: upper-right-front key light
    const LIGHT_DIR = [0.48, 0.64, 0.6];
    const LIGHT_COLOR = [1.0, 0.97, 0.92];
    const AMBIENT_COLOR = [0.28, 0.32, 0.45];

    const buf = new Float32Array(UNIFORM_BUFFER_SIZE / FLOAT_SIZE);
    buf.set(vp, 0);                  // viewProj: offset 0
    buf.set(eye, 16);                // eye: offset 64
    // _pad at 19
    buf.set(LIGHT_DIR, 20);          // lightDir: offset 80
    // _pad2 at 23
    buf.set(LIGHT_COLOR, 24);        // lightColor: offset 96
    // _pad3 at 27
    buf.set(AMBIENT_COLOR, 28);      // ambientColor: offset 112
    // _pad4 at 31

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

    // Draw edges first (behind nodes)
    if (this.edgeCount > 0) {
      renderPass.setPipeline(this.edgePipeline);
      renderPass.setVertexBuffer(0, this.cylinderGeo.positionBuffer);
      renderPass.setVertexBuffer(1, this.cylinderGeo.normalBuffer);
      renderPass.setVertexBuffer(2, this.edgeInstanceBuffer);
      renderPass.setIndexBuffer(this.cylinderGeo.indexBuffer, 'uint16');
      renderPass.drawIndexed(this.cylinderGeo.indexCount, this.edgeCount);
    }

    // Draw nodes
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

// --- Instance data builders ---

/**
 * Build node instance data from positions array.
 * @param {number[]} positions - flat [x,y,z, x,y,z, ...] or from WASM
 * @param {number} nodeCount
 * @param {number} radius - sphere radius
 * @param {number[][]} colors - per-node [r,g,b,a] or single color for all
 * @returns {Float32Array}
 */
export function buildNodeInstances(positions, nodeCount, radius, colors) {
  const data = new Float32Array(nodeCount * INSTANCE_FLOATS);
  const singleColor = colors.length === 4 && typeof colors[0] === 'number';

  for (let i = 0; i < nodeCount; i++) {
    const off = i * INSTANCE_FLOATS;
    const px = positions[i * 3];
    const py = positions[i * 3 + 1];
    const pz = positions[i * 3 + 2];

    // Scale matrix: uniform radius, translated to position
    // Column 0
    data[off + 0] = radius;
    data[off + 1] = 0;
    data[off + 2] = 0;
    data[off + 3] = 0;
    // Column 1
    data[off + 4] = 0;
    data[off + 5] = radius;
    data[off + 6] = 0;
    data[off + 7] = 0;
    // Column 2
    data[off + 8] = 0;
    data[off + 9] = 0;
    data[off + 10] = radius;
    data[off + 11] = 0;
    // Column 3 (translation)
    data[off + 12] = px;
    data[off + 13] = py;
    data[off + 14] = pz;
    data[off + 15] = 1;

    // Color
    const c = singleColor ? colors : (colors[i] || [0.5, 0.5, 0.5, 1]);
    data[off + 16] = c[0];
    data[off + 17] = c[1];
    data[off + 18] = c[2];
    data[off + 19] = c[3];
  }
  return data;
}

/**
 * Build edge instance data: cylinder from point A to point B.
 * @param {number[]} positions - flat [x,y,z, ...]
 * @param {number[][]} edges - [[src, dst], ...]
 * @param {number} edgeRadius
 * @param {number[]} color - [r,g,b,a]
 * @returns {Float32Array}
 */
export function buildEdgeInstances(positions, edges, edgeRadius, color) {
  const data = new Float32Array(edges.length * INSTANCE_FLOATS);

  for (let i = 0; i < edges.length; i++) {
    const [src, dst] = edges[i];
    const off = i * INSTANCE_FLOATS;

    const ax = positions[src * 3];
    const ay = positions[src * 3 + 1];
    const az = positions[src * 3 + 2];
    const bx = positions[dst * 3];
    const by = positions[dst * 3 + 1];
    const bz = positions[dst * 3 + 2];

    // Direction vector (A -> B)
    const dx = bx - ax;
    const dy = by - ay;
    const dz = bz - az;
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-6;

    // Normalized Y axis of cylinder (along edge)
    const yx = dx / len;
    const yy = dy / len;
    const yz = dz / len;

    // Find a perpendicular vector for X axis
    // Use cross product with world up, fall back to world right if parallel
    const PARALLEL_THRESHOLD = 0.99;
    let xx, xy, xz;
    if (Math.abs(yy) < PARALLEL_THRESHOLD) {
      // cross(edge, up) where up = (0,1,0)
      xx = yz;
      xy = 0;
      xz = -yx;
    } else {
      // cross(edge, right) where right = (1,0,0)
      xx = 0;
      xy = -yz;
      xz = yy;
    }
    // Normalize X
    const xl = Math.sqrt(xx * xx + xy * xy + xz * xz) || 1e-6;
    xx /= xl; xy /= xl; xz /= xl;

    // Z axis = cross(X, Y)
    const zx = xy * yz - xz * yy;
    const zy = xz * yx - xx * yz;
    const zz = xx * yy - xy * yx;

    // Transform: scale X and Z by radius, Y by length, translate to A
    // Column 0 (X axis * radius)
    data[off + 0] = xx * edgeRadius;
    data[off + 1] = xy * edgeRadius;
    data[off + 2] = xz * edgeRadius;
    data[off + 3] = 0;
    // Column 1 (Y axis * length)
    data[off + 4] = yx * len;
    data[off + 5] = yy * len;
    data[off + 6] = yz * len;
    data[off + 7] = 0;
    // Column 2 (Z axis * radius)
    data[off + 8] = zx * edgeRadius;
    data[off + 9] = zy * edgeRadius;
    data[off + 10] = zz * edgeRadius;
    data[off + 11] = 0;
    // Column 3 (translation = point A)
    data[off + 12] = ax;
    data[off + 13] = ay;
    data[off + 14] = az;
    data[off + 15] = 1;

    // Color
    data[off + 16] = color[0];
    data[off + 17] = color[1];
    data[off + 18] = color[2];
    data[off + 19] = color[3];
  }
  return data;
}

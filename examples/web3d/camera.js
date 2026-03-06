// Arcball orbit camera with mouse drag to rotate, scroll to zoom, middle-click to pan.

export class OrbitCamera {
  constructor(canvas) {
    this.canvas = canvas;
    this.distance = 500;
    this.theta = Math.PI / 4;   // azimuth
    this.phi = Math.PI / 3;     // elevation
    this.target = [0, 0, 0];
    this.fov = Math.PI / 4;
    this.near = 1;
    this.far = 5000;
    this.aspect = canvas.width / canvas.height;

    this._dragging = false;
    this._panning = false;
    this._lastX = 0;
    this._lastY = 0;

    const ROTATION_SPEED = 0.005;
    const ZOOM_SPEED = 1.08;
    const PAN_SPEED = 0.5;
    const MIN_PHI = 0.05;
    const MAX_PHI = Math.PI - 0.05;

    canvas.addEventListener('mousedown', (e) => {
      if (e.button === 0) { this._dragging = true; }
      if (e.button === 1) { this._panning = true; e.preventDefault(); }
      this._lastX = e.clientX;
      this._lastY = e.clientY;
    });

    window.addEventListener('mouseup', () => {
      this._dragging = false;
      this._panning = false;
    });

    canvas.addEventListener('mousemove', (e) => {
      const dx = e.clientX - this._lastX;
      const dy = e.clientY - this._lastY;
      this._lastX = e.clientX;
      this._lastY = e.clientY;

      if (this._dragging) {
        this.theta -= dx * ROTATION_SPEED;
        this.phi = Math.max(MIN_PHI, Math.min(MAX_PHI, this.phi - dy * ROTATION_SPEED));
      }
      if (this._panning) {
        const right = this._right();
        const up = this._up();
        const scale = this.distance * PAN_SPEED * 0.002;
        for (let i = 0; i < 3; i++) {
          this.target[i] -= right[i] * dx * scale - up[i] * dy * scale;
        }
      }
    });

    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      if (e.deltaY > 0) this.distance *= ZOOM_SPEED;
      else this.distance /= ZOOM_SPEED;
      this.distance = Math.max(10, Math.min(4000, this.distance));
    }, { passive: false });

    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  get eye() {
    const st = Math.sin(this.phi);
    return [
      this.target[0] + this.distance * st * Math.cos(this.theta),
      this.target[1] + this.distance * Math.cos(this.phi),
      this.target[2] + this.distance * st * Math.sin(this.theta),
    ];
  }

  _right() {
    return [-Math.sin(this.theta), 0, Math.cos(this.theta)];
  }

  _up() {
    const ct = Math.cos(this.theta);
    const st = Math.sin(this.theta);
    const cp = Math.cos(this.phi);
    return [-cp * ct, Math.sin(this.phi), -cp * st];
  }

  resize(w, h) {
    this.aspect = w / h;
  }

  viewMatrix() {
    return lookAt(this.eye, this.target, [0, 1, 0]);
  }

  projectionMatrix() {
    return perspective(this.fov, this.aspect, this.near, this.far);
  }

  viewProjMatrix() {
    return multiply(this.projectionMatrix(), this.viewMatrix());
  }
}

// --- Math utilities (column-major mat4) ---

function lookAt(eye, target, up) {
  const z = normalize(sub(eye, target));
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
  ]);
}

function perspective(fov, aspect, near, far) {
  const f = 1 / Math.tan(fov / 2);
  const nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ]);
}

function multiply(a, b) {
  const r = new Float32Array(16);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      r[j * 4 + i] = a[i] * b[j * 4] + a[4 + i] * b[j * 4 + 1] + a[8 + i] * b[j * 4 + 2] + a[12 + i] * b[j * 4 + 3];
    }
  }
  return r;
}

function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function cross(a, b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
function normalize(v) {
  const l = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) || 1;
  return [v[0]/l, v[1]/l, v[2]/l];
}

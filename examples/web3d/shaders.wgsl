// Shared uniforms for all pipelines.
struct Uniforms {
    viewProj: mat4x4f,
    eye: vec3f,
    _pad: f32,
    lightDir: vec3f,
    _pad2: f32,
    lightColor: vec3f,
    _pad3: f32,
    ambientColor: vec3f,
    _pad4: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

// ---- Node pipeline (instanced spheres) ----

struct NodeInstance {
    @location(3) col0: vec4f,
    @location(4) col1: vec4f,
    @location(5) col2: vec4f,
    @location(6) col3: vec4f,
    @location(7) color: vec4f,
};

struct NodeVsOut {
    @builtin(position) pos: vec4f,
    @location(0) worldNormal: vec3f,
    @location(1) worldPos: vec3f,
    @location(2) color: vec3f,
};

@vertex fn vs_node(
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    inst: NodeInstance,
) -> NodeVsOut {
    let model = mat4x4f(inst.col0, inst.col1, inst.col2, inst.col3);
    let worldPos = (model * vec4f(position, 1.0)).xyz;
    let worldNormal = normalize((model * vec4f(normal, 0.0)).xyz);

    var out: NodeVsOut;
    out.pos = u.viewProj * vec4f(worldPos, 1.0);
    out.worldNormal = worldNormal;
    out.worldPos = worldPos;
    out.color = inst.color.rgb;
    return out;
}

@fragment fn fs_node(in: NodeVsOut) -> @location(0) vec4f {
    let N = normalize(in.worldNormal);
    let L = normalize(u.lightDir);
    let V = normalize(u.eye - in.worldPos);
    let H = normalize(L + V);

    // Blinn-Phong
    let diff = max(dot(N, L), 0.0);
    let spec = pow(max(dot(N, H), 0.0), 64.0);

    // Hemisphere ambient
    let ambient = mix(u.ambientColor * 0.4, u.ambientColor, dot(N, vec3f(0.0, 1.0, 0.0)) * 0.5 + 0.5);

    let color = in.color * (ambient + u.lightColor * diff) + u.lightColor * spec * 0.3;

    return vec4f(color, 1.0);
}

// ---- Edge pipeline (instanced cylinders) ----

struct EdgeInstance {
    @location(3) col0: vec4f,
    @location(4) col1: vec4f,
    @location(5) col2: vec4f,
    @location(6) col3: vec4f,
    @location(7) color: vec4f,
};

struct EdgeVsOut {
    @builtin(position) pos: vec4f,
    @location(0) worldNormal: vec3f,
    @location(1) worldPos: vec3f,
    @location(2) color: vec3f,
};

@vertex fn vs_edge(
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    inst: EdgeInstance,
) -> EdgeVsOut {
    let model = mat4x4f(inst.col0, inst.col1, inst.col2, inst.col3);
    let worldPos = (model * vec4f(position, 1.0)).xyz;
    let worldNormal = normalize((model * vec4f(normal, 0.0)).xyz);

    var out: EdgeVsOut;
    out.pos = u.viewProj * vec4f(worldPos, 1.0);
    out.worldNormal = worldNormal;
    out.worldPos = worldPos;
    out.color = inst.color.rgb;
    return out;
}

@fragment fn fs_edge(in: EdgeVsOut) -> @location(0) vec4f {
    let N = normalize(in.worldNormal);
    let L = normalize(u.lightDir);
    let V = normalize(u.eye - in.worldPos);
    let H = normalize(L + V);

    let diff = max(dot(N, L), 0.0);
    let spec = pow(max(dot(N, H), 0.0), 32.0);

    let ambient = mix(u.ambientColor * 0.3, u.ambientColor * 0.6, dot(N, vec3f(0.0, 1.0, 0.0)) * 0.5 + 0.5);

    let color = in.color * (ambient + u.lightColor * diff * 0.8) + u.lightColor * spec * 0.15;

    return vec4f(color, 1.0);
}

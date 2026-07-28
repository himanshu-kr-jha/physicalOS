import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Environment, Html, Line, OrbitControls, Sky } from "@react-three/drei";
import * as THREE from "three";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";

import {
  apiUrl,
  colorOf,
  isAbsence,
  isAsset,
  poseAt,
  toLocal,
  useStore,
  useVisibleFindings,
} from "./store";
import type { Pose } from "./store";
import type { Basemap, BuildingGeom, Finding, Frame, Segment } from "./types";

/**
 * The 3D twin.
 *
 * Reuses map3d's approach for buildings -- build a THREE.Shape from the
 * footprint and extrude it -- but everything here lives in local ENU metres
 * rather than map3d's single `scale = 51000` constant. One unit is one metre,
 * so marker heights, road widths and ranges are all directly comparable.
 */

const ORIGIN_FALLBACK: [number, number] = [0, 0];

// --------------------------------------------------------------------------
// Buildings
// --------------------------------------------------------------------------

function BuildingMesh({
  shape,
  height,
  tags,
}: {
  shape: THREE.Shape;
  height: number;
  tags: Record<string, string>;
}) {
  const [hovered, setHovered] = useState(false);
  const name = tags.name || tags.building || "Building";

  return (
    <mesh
      // extrudeGeometry extrudes along +Z, so lay the shape flat and lift it.
      rotation={[Math.PI / 2, 0, 0]}
      position={[0, height, 0]}
      onPointerOver={(e) => {
        e.stopPropagation();
        setHovered(true);
      }}
      onPointerOut={() => setHovered(false)}
    >
      <extrudeGeometry args={[shape, { depth: height, bevelEnabled: false }]} />
      <meshStandardMaterial
        color={hovered ? "#5b8def" : "#9aa0a6"}
        roughness={0.85}
        metalness={0.05}
      />
      {hovered && (
        <Html center distanceFactor={60}>
          <div className="tip">
            <strong>{name}</strong>
            <div>{height.toFixed(1)} m</div>
          </div>
        </Html>
      )}
    </mesh>
  );
}

function Buildings({
  buildings,
  origin,
}: {
  buildings: BuildingGeom[];
  origin: [number, number];
}) {
  const shapes = useMemo(() => {
    const out: {
      id: number;
      shape: THREE.Shape;
      height: number;
      tags: Record<string, string>;
    }[] = [];

    for (const b of buildings) {
      const pts = b.footprint.map(([lat, lon]) => {
        const [x, z] = toLocal(lat, lon, origin);
        return new THREE.Vector2(x, z);
      });
      if (pts.length < 3) continue;
      // OSM repeats the first vertex to close the way; extrude closes it for us.
      if (pts[0].distanceTo(pts[pts.length - 1]) < 0.01) pts.pop();
      if (pts.length < 3) continue;

      out.push({
        id: b.osm_id,
        shape: new THREE.Shape(pts),
        height: b.height_m,
        tags: b.tags,
      });
    }
    return out;
  }, [buildings, origin]);

  return (
    <group>
      {shapes.map((s) => (
        <BuildingMesh key={s.id} shape={s.shape} height={s.height} tags={s.tags} />
      ))}
    </group>
  );
}

// --------------------------------------------------------------------------
// Route + heatmap
// --------------------------------------------------------------------------

function RouteLine({ frames, origin }: { frames: Frame[]; origin: [number, number] }) {
  const pts = useMemo(
    () =>
      frames.map((f) => {
        const [x, z] = toLocal(f.lat, f.lon, origin);
        return new THREE.Vector3(x, 0.08, z);
      }),
    [frames, origin]
  );
  if (pts.length < 2) return null;
  return (
    <Line points={pts} color="#e6edf3" lineWidth={1.5} dashed dashSize={2} gapSize={1.4} />
  );
}

/**
 * The stretch already driven, drawn solid over the dashed full route.
 *
 * This is what makes a moving map legible: the dashed line is where the run
 * goes, the solid line is how much of it you have watched.
 *
 * Keyed on the keyframe index, not on playT. playT ticks at 60 Hz, and
 * rebuilding a few hundred Vector3s that often is pure waste when the trail
 * only ever grows one keyframe at a time.
 */
function TrailLine({
  frames,
  origin,
  index,
}: {
  frames: Frame[];
  origin: [number, number];
  index: number;
}) {
  const all = useMemo(
    () =>
      frames.map((f) => {
        const [x, z] = toLocal(f.lat, f.lon, origin);
        return new THREE.Vector3(x, 0.12, z);
      }),
    [frames, origin]
  );
  const pts = useMemo(() => all.slice(0, Math.max(index + 1, 0)), [all, index]);

  if (pts.length < 2) return null;
  return <Line points={pts} color="#38bdf8" lineWidth={3} transparent opacity={0.9} />;
}

/**
 * Where the camera vehicle is right now.
 *
 * A map that pans with no marker on it just looks like drift -- you cannot tell
 * which end of the trail is "now", or which way the camera was pointing when a
 * defect came into view.
 */
function EgoMarker({
  pose,
  origin,
  active,
}: {
  pose: Pose;
  origin: [number, number];
  active: boolean;
}) {
  const [x, z] = toLocal(pose.lat, pose.lon, origin);

  // ORIENTATION -- the sign that would silently point the arrow backwards:
  //   heading_deg is clockwise from north, and north is -Z (store.ts maps
  //   z = -north). A cone points +Y, so rotation-x of -90deg lays it along -Z,
  //   i.e. due north. Rotating the group by -heading about Y then swings it
  //   clockwise onto the bearing. Flip either convention and this mirrors.
  const yaw = -(pose.heading_deg * Math.PI) / 180;

  return (
    <group position={[x, 0, z]}>
      <mesh position={[0, 0.16, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[1.4, 28]} />
        <meshBasicMaterial color="#38bdf8" transparent opacity={active ? 0.85 : 0.4} />
      </mesh>

      <group rotation={[0, yaw, 0]}>
        <mesh position={[0, 0.75, -2.0]} rotation={[-Math.PI / 2, 0, 0]}>
          <coneGeometry args={[0.95, 2.8, 4]} />
          <meshStandardMaterial
            color="#38bdf8"
            emissive="#38bdf8"
            emissiveIntensity={active ? 0.95 : 0.25}
          />
        </mesh>
      </group>

      <mesh position={[0, 0.1, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[3.0, 3.5, 36]} />
        <meshBasicMaterial
          color="#38bdf8"
          transparent
          opacity={active ? 0.45 : 0.15}
          side={THREE.DoubleSide}
        />
      </mesh>
    </group>
  );
}

/**
 * Road segments coloured by quality index -- the heatmap.
 *
 * Each segment is a flat quad laid along the route, at a typical carriageway
 * width so the ribbon reads as a road rather than a line.
 */
function HeatmapRibbon({
  segments,
  origin,
  onPick,
}: {
  segments: Segment[];
  origin: [number, number];
  onPick: (seg: Segment) => void;
}) {
  const quads = useMemo(() => {
    const HALF_W = 4.0;
    return segments.map((seg) => {
      const [x1, z1] = toLocal(seg.start[0], seg.start[1], origin);
      const [x2, z2] = toLocal(seg.end[0], seg.end[1], origin);

      const dx = x2 - x1;
      const dz = z2 - z1;
      const len = Math.hypot(dx, dz) || 1;
      // Perpendicular within the ground plane.
      const px = (-dz / len) * HALF_W;
      const pz = (dx / len) * HALF_W;

      const y = 0.04; // just above the ground, to avoid z-fighting
      const verts = new Float32Array([
        x1 + px, y, z1 + pz,
        x2 + px, y, z2 + pz,
        x2 - px, y, z2 - pz,
        x1 + px, y, z1 + pz,
        x2 - px, y, z2 - pz,
        x1 - px, y, z1 - pz,
      ]);

      const geom = new THREE.BufferGeometry();
      geom.setAttribute("position", new THREE.BufferAttribute(verts, 3));
      geom.computeVertexNormals();
      return { seg, geom };
    });
  }, [segments, origin]);

  return (
    <group>
      {quads.map(({ seg, geom }) => (
        <mesh
          key={seg.seg_id}
          geometry={geom}
          onClick={(e) => {
            e.stopPropagation();
            onPick(seg);
          }}
        >
          <meshBasicMaterial
            color={seg.color}
            transparent
            opacity={0.82}
            side={THREE.DoubleSide}
          />
        </mesh>
      ))}
    </group>
  );
}

// --------------------------------------------------------------------------
// Findings
// --------------------------------------------------------------------------

function Markers({
  findings,
  origin,
  selectedId,
  onPick,
}: {
  findings: Finding[];
  origin: [number, number];
  selectedId: string | null;
  onPick: (f: Finding) => void;
}) {
  const classMap = useStore((s) => s.classMap);

  return (
    <group>
      {findings.map((f) => {
        if (f.lat === null || f.lon === null) return null;
        const [x, z] = toLocal(f.lat, f.lon, origin);
        const color = colorOf(classMap, f.cls);
        const isSel = f.finding_id === selectedId;
        // Taller pin for higher severity, so the worst things read first.
        const h = 1.6 + f.severity * 0.7;
        const asset = isAsset(classMap, f.cls);
        const absent = isAbsence(classMap, f.cls);

        // Absence is a property of a STRETCH, not a point, so it must not look
        // like a pin stuck at one spot. It gets a wide hovering ring plus an
        // always-visible label, so "nothing here" reads as deliberate rather
        // than as a marker that failed to place.
        if (absent) {
          return (
            <group key={f.finding_id} position={[x, 0, z]}>
              <mesh position={[0, 0.1, 0]} rotation={[-Math.PI / 2, 0, 0]}>
                <ringGeometry args={[5.5, 7.5, 40]} />
                <meshBasicMaterial
                  color={color}
                  transparent
                  opacity={isSel ? 0.55 : 0.3}
                  side={THREE.DoubleSide}
                />
              </mesh>
              <mesh
                position={[0, 3.2, 0]}
                onClick={(e) => {
                  e.stopPropagation();
                  onPick(f);
                }}
                onPointerOver={(e) => {
                  e.stopPropagation();
                  document.body.style.cursor = "pointer";
                }}
                onPointerOut={() => {
                  document.body.style.cursor = "auto";
                }}
              >
                <torusGeometry args={[1.5, 0.28, 10, 28]} />
                <meshStandardMaterial
                  color={color}
                  emissive={color}
                  emissiveIntensity={isSel ? 1.0 : 0.4}
                  transparent
                  opacity={0.9}
                />
              </mesh>
              <Html center position={[0, 6.4, 0]} distanceFactor={90}>
                <div className="tip tip-absent">
                  <strong>{f.label}</strong>
                  <div>missing along this stretch</div>
                </div>
              </Html>
            </group>
          );
        }

        return (
          <group key={f.finding_id} position={[x, 0, z]}>
            <mesh position={[0, h / 2, 0]}>
              <cylinderGeometry args={[0.09, 0.09, h, 6]} />
              <meshStandardMaterial color={color} transparent opacity={0.7} />
            </mesh>

            {/* Cone for defects (apex points down at the spot), sphere for assets. */}
            <mesh
              position={[0, h + 0.55, 0]}
              rotation={asset ? [0, 0, 0] : [Math.PI, 0, 0]}
              onClick={(e) => {
                e.stopPropagation();
                onPick(f);
              }}
              onPointerOver={(e) => {
                e.stopPropagation();
                document.body.style.cursor = "pointer";
              }}
              onPointerOut={() => {
                document.body.style.cursor = "auto";
              }}
            >
              {asset ? (
                <sphereGeometry args={[0.62, 16, 12]} />
              ) : (
                <coneGeometry args={[0.72, 1.5, 7]} />
              )}
              <meshStandardMaterial
                color={color}
                emissive={color}
                emissiveIntensity={isSel ? 1.1 : 0.35}
              />
            </mesh>

            {/* Ring on the ground marks the actual projected position. */}
            <mesh position={[0, 0.06, 0]} rotation={[-Math.PI / 2, 0, 0]}>
              <ringGeometry args={[0.7, 1.05, 24]} />
              <meshBasicMaterial
                color={color}
                transparent
                opacity={isSel ? 0.95 : 0.5}
                side={THREE.DoubleSide}
              />
            </mesh>

            {isSel && (
              <Html center position={[0, h + 2.4, 0]} distanceFactor={70}>
                <div className="tip tip-sel">
                  <strong>{f.label}</strong>
                  <div>
                    sev {f.severity} · {(f.confidence * 100).toFixed(0)}% ·{" "}
                    {f.evidence.length}x
                  </div>
                </div>
              </Html>
            )}
          </group>
        );
      })}
    </group>
  );
}

// --------------------------------------------------------------------------
// Point cloud (the lingbot-map reconstruction, when present)
// --------------------------------------------------------------------------

function PointCloud() {
  const [geom, setGeom] = useState<THREE.BufferGeometry | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    new PLYLoader().load(
      apiUrl("/api/cloud.ply"),
      (g) => {
        if (cancelled) return;
        g.computeBoundingBox();
        setGeom(g);
      },
      undefined,
      () => {
        if (!cancelled) setFailed(true);
      }
    );
    return () => {
      cancelled = true;
    };
  }, []);

  if (failed || !geom) return null;

  const hasColor = geom.hasAttribute("color");
  return (
    <points geometry={geom}>
      <pointsMaterial
        size={0.06}
        sizeAttenuation
        vertexColors={hasColor}
        color={hasColor ? "#ffffff" : "#8fb8d8"}
      />
    </points>
  );
}

// --------------------------------------------------------------------------
// Camera framing
// --------------------------------------------------------------------------

/** The bit of OrbitControls the camera code touches, without pulling in its types. */
type OrbitLike = { target: THREE.Vector3 };

/** Frame the whole route once on load, so nothing starts off-screen. */
function FitCamera({ frames, origin }: { frames: Frame[]; origin: [number, number] }) {
  const { camera } = useThree();
  const controls = useThree((s) => s.controls) as unknown as OrbitLike | null;
  const done = useRef(false);

  useEffect(() => {
    if (done.current || frames.length < 2 || !controls) return;
    done.current = true;

    let minX = Infinity;
    let maxX = -Infinity;
    let minZ = Infinity;
    let maxZ = -Infinity;
    for (const f of frames) {
      const [x, z] = toLocal(f.lat, f.lon, origin);
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minZ = Math.min(minZ, z);
      maxZ = Math.max(maxZ, z);
    }
    const cx = (minX + maxX) / 2;
    const cz = (minZ + maxZ) / 2;
    const span = Math.max(maxX - minX, maxZ - minZ, 40);

    camera.position.set(cx - span * 0.35, span * 0.55, cz + span * 0.75);
    camera.lookAt(cx, 0, cz);
    camera.updateProjectionMatrix();
    // The orbit target is set here rather than by a `target` prop on
    // OrbitControls: Scene re-renders on every playback tick, and a prop would
    // be re-applied each time, yanking the chase camera back to route centre.
    controls.target.set(cx, 0, cz);
  }, [frames, origin, camera, controls]);

  return null;
}

// Exponential ease toward the vehicle, per second. High enough to keep up at
// road speed, low enough that GPS jitter between keyframes does not shake the view.
const FOLLOW_RATE = 3.2;
// Past this, jump. Scrubbing from 0s to the end should not fly the camera the
// length of the run.
const SNAP_M = 120;
// Keep easing this long after the head stops, so a scrub still lands on target
// instead of freezing halfway there.
const SETTLE_S = 2.0;
// How far from the vehicle to watch from, in metres, once following engages.
// FitCamera frames the WHOLE route, which on a 5 km drive is kilometres up --
// panning at that altitude follows a speck. Closing in happens once per
// engagement and only ever pulls inwards, so a user who zooms out mid-drive,
// or who was already closer than this, is left alone.
const DRIVE_DIST_M = 95;

/**
 * Chase camera -- the map moving along the path as the video plays.
 *
 * It translates the orbit target and the camera by the SAME vector, so the
 * user's angle, zoom and tilt all survive: this pans, it never re-aims. That
 * also means OrbitControls' damping has nothing to fight, and a drag mid-drive
 * just changes the viewpoint you are following from.
 *
 * playT is read via getState() inside useFrame rather than subscribed to.
 * Subscribing would re-render the whole scene graph 60 times a second in order
 * to move one vector.
 *
 * When the head is parked (paused, not streaming, no recent scrub) this does
 * nothing at all -- otherwise every attempt to look around would be dragged
 * back to the vehicle.
 */
function FollowCam({ frames, origin }: { frames: Frame[]; origin: [number, number] }) {
  const camera = useThree((s) => s.camera);
  const controls = useThree((s) => s.controls) as unknown as OrbitLike | null;

  const want = useRef(new THREE.Vector3());
  const step = useRef(new THREE.Vector3());
  const arm = useRef(new THREE.Vector3());
  const prevT = useRef<number | null>(null);
  const lastMove = useRef(-Infinity);
  const wasActive = useRef(false);
  const closingIn = useRef(false);

  useFrame((state, dt) => {
    if (!controls || frames.length < 2) return;

    const { follow, playT, playing, streaming } = useStore.getState();

    if (prevT.current === null) prevT.current = playT;
    if (Math.abs(playT - prevT.current) > 1e-4) {
      prevT.current = playT;
      lastMove.current = state.clock.elapsedTime;
    }

    const idleFor = state.clock.elapsedTime - lastMove.current;
    const active = follow && (playing || streaming || idleFor <= SETTLE_S);
    if (!active) {
      wasActive.current = false;
      return;
    }
    // Rising edge: this is the moment the chase view takes over.
    if (!wasActive.current) {
      wasActive.current = true;
      closingIn.current = true;
    }

    const pose = poseAt(frames, playT);
    if (!pose) return;

    const [x, z] = toLocal(pose.lat, pose.lon, origin);
    want.current.set(x, 0, z);

    step.current.copy(want.current).sub(controls.target);
    if (step.current.length() <= SNAP_M) {
      // Framerate-independent ease: the same fraction of the gap per second
      // however fast the tab happens to be rendering.
      step.current.multiplyScalar(1 - Math.exp(-dt * FOLLOW_RATE));
    }

    controls.target.add(step.current);
    camera.position.add(step.current);

    if (closingIn.current) {
      arm.current.copy(camera.position).sub(controls.target);
      const dist = arm.current.length();
      if (dist <= DRIVE_DIST_M * 1.05) {
        closingIn.current = false;
      } else {
        // Pull in along the current view direction, so the angle the user is
        // watching from is preserved -- only the distance changes.
        const k = 1 - Math.exp(-dt * FOLLOW_RATE);
        const next = dist + (DRIVE_DIST_M - dist) * k;
        camera.position.copy(controls.target).addScaledVector(arm.current.divideScalar(dist), next);
      }
    }
  });

  return null;
}

/**
 * Real satellite imagery as the ground, from `pos basemap`.
 *
 * One texture, not a tile pyramid: a run spans a few hundred metres, so the
 * server stitches the tiles once and this just places the result. The extent
 * arrives in the same local metres as toLocal(), so there is no tile maths here.
 *
 * ORIENTATION -- the bug that would silently mirror the map north/south:
 *   planeGeometry lies in XY. rotation=[-90deg,0,0] sends plane +Y to world -Z.
 *   A texture's V=1 edge is the image's TOP row, which is the NORTHERNMOST
 *   latitude. North is the most negative z (store.ts maps z = -north). So
 *   plane +Y -> world -Z -> north, and the image lands the right way up with no
 *   flip. Change either convention and this silently inverts.
 *
 * meshBasicMaterial, not standard: this is a photograph that already contains
 * its own lighting. Lighting it again washes out the road surface.
 */
function SatelliteGround({ basemap }: { basemap: Basemap }) {
  const [tex, setTex] = useState<THREE.Texture | null>(null);

  useEffect(() => {
    let cancelled = false;
    // Imperative loader rather than drei's useTexture, which suspends -- there
    // is no Suspense boundary in this tree, and PointCloud already establishes
    // the load-then-render pattern.
    new THREE.TextureLoader().load(
      apiUrl("/api/basemap.jpg"),
      (t) => {
        if (cancelled) return;
        t.colorSpace = THREE.SRGBColorSpace;
        t.anisotropy = 8; // the plane is viewed at a grazing angle
        setTex(t);
      },
      undefined,
      () => undefined
    );
    return () => {
      cancelled = true;
    };
  }, []);

  const L = basemap.local;
  if (!tex || !L) return null;

  const w = L.x_max - L.x_min;
  const h = L.z_max - L.z_min;
  if (!(w > 0 && h > 0)) return null;

  return (
    <mesh
      rotation={[-Math.PI / 2, 0, 0]}
      position={[(L.x_min + L.x_max) / 2, -0.02, (L.z_min + L.z_max) / 2]}
    >
      <planeGeometry args={[w, h]} />
      <meshBasicMaterial map={tex} toneMapped={false} />
    </mesh>
  );
}

// --------------------------------------------------------------------------

export function Scene() {
  const manifest = useStore((s) => s.manifest);
  const frames = useStore((s) => s.frames);
  const segments = useStore((s) => s.segments);
  const twin = useStore((s) => s.twin);
  const layers = useStore((s) => s.layers);
  const basemap = useStore((s) => s.basemap);
  const selected = useStore((s) => s.selected);
  const select = useStore((s) => s.select);
  const playT = useStore((s) => s.playT);
  const playing = useStore((s) => s.playing);
  const streaming = useStore((s) => s.streaming);
  const findings = useVisibleFindings();

  const showSatellite = Boolean(layers.satellite && basemap?.available && basemap.local);

  const origin = manifest?.origin ?? ORIGIN_FALLBACK;

  // Cheap: a binary search and a lerp. This component already re-renders every
  // tick through useVisibleFindings, so it costs nothing extra to keep the ego
  // marker and the trail exact rather than a keyframe behind.
  const pose = useMemo(() => poseAt(frames, playT), [frames, playT]);

  const centre = useMemo<[number, number]>(() => {
    if (!frames.length) return [0, 0];
    const mid = frames[Math.floor(frames.length / 2)];
    return toLocal(mid.lat, mid.lon, origin);
  }, [frames, origin]);

  return (
    <Canvas
      dpr={[1, 2]}
      camera={{ fov: 55, near: 0.5, far: 4000, position: [0, 60, 90] }}
      onPointerMissed={() => select(null)}
    >
      <color attach="background" args={["#0b0f14"]} />
      <Sky sunPosition={[80, 40, -60]} turbidity={7} rayleigh={2.4} />
      <Environment preset="city" />
      <ambientLight intensity={0.55} />
      <directionalLight position={[60, 90, -40]} intensity={1.15} />

      {/* Ground, large enough that the route never runs off it. Sits below the
          satellite plane so nothing shows through to the void at the edges. */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[centre[0], -0.05, centre[1]]}>
        <planeGeometry args={[1600, 1600]} />
        <meshStandardMaterial color="#1b2129" roughness={1} />
      </mesh>
      {/* The grid exists to give an abstract plane a sense of scale. Real imagery
          does that better, so keeping both would only add noise. */}
      {!showSatellite && (
        <gridHelper
          args={[1600, 160, "#2b3440", "#222a33"]}
          position={[centre[0], 0, centre[1]]}
        />
      )}
      {showSatellite && basemap && <SatelliteGround basemap={basemap} />}

      {layers.cloud && manifest?.has_pointcloud && <PointCloud />}
      {layers.buildings && <Buildings buildings={twin.buildings} origin={origin} />}
      {layers.heatmap && (
        <HeatmapRibbon
          segments={segments}
          origin={origin}
          onPick={(seg) => {
            // Clicking a segment selects its worst finding -- why it is red.
            const members = useStore
              .getState()
              .findings.filter((f) => seg.finding_ids.includes(f.finding_id));
            if (members.length) {
              members.sort((a, b) => b.severity - a.severity);
              select(members[0]);
            }
          }}
        />
      )}
      {layers.route && <RouteLine frames={frames} origin={origin} />}
      {layers.route && pose && (
        <TrailLine frames={frames} origin={origin} index={pose.index} />
      )}
      {pose && (
        <EgoMarker pose={pose} origin={origin} active={playing || streaming} />
      )}
      {layers.markers && (
        <Markers
          findings={findings}
          origin={origin}
          selectedId={selected?.finding_id ?? null}
          onPick={select}
        />
      )}

      {/* OrbitControls first: both camera helpers read it off the R3F store, and
          FitCamera's effect only runs once it has registered itself. */}
      <FitCamera frames={frames} origin={origin} />
      <FollowCam frames={frames} origin={origin} />
      <OrbitControls
        makeDefault
        maxPolarAngle={Math.PI / 2.05}
        minDistance={5}
        maxDistance={900}
        enableDamping
        dampingFactor={0.08}
      />
    </Canvas>
  );
}

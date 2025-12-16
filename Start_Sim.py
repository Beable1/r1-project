#!/usr/bin/env python3
"""
Simplified Scene Loader + Extension Enabler

This script only:
  1. Starts Isaac Sim (optionally headless)
  2. (Optionally) enables all extensions (or filtered)
  3. Loads the given USD stage
  4. Enables ROS2 bridge extension
  5. Idles at a target FPS

Environment Variables:
  SCENE_USD_PATH               Path to USD (default: r1.usd in script directory)
  HEADLESS=1                   Headless mode
  IDLE_FPS=20                  Idle update rate
  AUTO_ENABLE_ALL_EXTENSIONS=1 Enable all extensions
  AUTO_ENABLE_EXT_PREFIX=omni.isaac.  Only enable those starting with prefix
  AUTO_ENABLE_EXT_EXCLUDE=a,b  Comma list to skip
  SAFE_RENDER_MODE=1           Lower render load (Storm delegate etc.)
  DISABLE_FORCE_ROS_BRIDGE=0   Set to 1 to skip ROS bridge enable
  ROS_BRIDGE_AUTO_GRAPH=1      Auto-enable graph extensions
  ROS_BRIDGE_VERBOSE_DIAG=1    Verbose ROS bridge diagnostics
"""

import os
import time
from pathlib import Path

# Default to r1.usd in the same directory as this script
_script_dir = Path(__file__).parent.absolute()
USD_PATH = os.getenv("SCENE_USD_PATH", str(_script_dir / "r1.usd"))
HEADLESS = os.getenv("HEADLESS", "0").lower() in ("1", "true", "yes")
IDLE_FPS = float(os.getenv("IDLE_FPS", "25"))
AUTO_ENABLE_ALL_EXT = os.getenv("AUTO_ENABLE_ALL_EXTENSIONS", "1").lower() in ("1", "true", "yes")
SAFE_RENDER = os.getenv("SAFE_RENDER_MODE", "1").lower() in ("1", "true", "yes")
EXT_PREFIX = os.getenv("AUTO_ENABLE_EXT_PREFIX", "")
EXT_EXCLUDE = {x.strip() for x in os.getenv("AUTO_ENABLE_EXT_EXCLUDE", "").split(',') if x.strip()}
EXT_ALLOWLIST = {x.strip() for x in os.getenv("EXTENSION_ALLOWLIST", "").split(',') if x.strip()}
EXT_BLOCKLIST = {x.strip() for x in os.getenv("EXTENSION_BLOCKLIST", "").split(',') if x.strip()}
DISABLE_RTX = os.getenv("DISABLE_RTX", "0").lower() in ("1", "true", "yes")

os.environ.setdefault("OMNI_FETCH_ASSETS", "1")

try:
    from isaacsim import SimulationApp  # noqa: E402
except Exception as e:
    raise SystemExit(f"SimulationApp import failed: {e}")

simulation_app = SimulationApp({"headless": HEADLESS})

CRITICAL_ROS_EXT_NAMES = [
    "isaacsim.ros2.bridge",      # New preferred name per deprecation warning
    "omni.isaac.ros2_bridge",    # Legacy name
]

DISABLE_FORCE_ROS_BRIDGE = os.getenv("DISABLE_FORCE_ROS_BRIDGE", "0").lower() in ("1", "true", "yes")
ROS_BRIDGE_AUTO_GRAPH = os.getenv("ROS_BRIDGE_AUTO_GRAPH", "1").lower() in ("1", "true", "yes")
ROS_BRIDGE_VERBOSE_DIAG = os.getenv("ROS_BRIDGE_VERBOSE_DIAG", "1").lower() in ("1", "true", "yes")

def _env_dump_for_ros():
    if not ROS_BRIDGE_VERBOSE_DIAG:
        return
    keys = [
        'PYTHONPATH', 'OMNI_EXTENSIONS_PATH', 'OMNI_KIT_PRIORITIZED_EXTENSIONS', 'LD_LIBRARY_PATH',
        'ROS_DOMAIN_ID', 'RMW_IMPLEMENTATION'
    ]
    print('[ROS2_BRIDGE][ENV] ----')
    for k in keys:
        v = os.getenv(k)
        if v:
            print(f'  {k}={v}')
    print('[ROS2_BRIDGE][ENV] ----')

def _dump_all_ros_extensions():
    if not ROS_BRIDGE_VERBOSE_DIAG:
        return
    try:
        import omni.kit.app
        app = omni.kit.app.get_app()
        mgr = app.get_extension_manager()
        all_exts = mgr.get_extensions()
        print('[ROS2_BRIDGE][EXT] Listing extensions containing "ros" substring:')
        for ext_id, meta in sorted(all_exts.items()):
            if 'ros' in ext_id.lower():
                enabled = False
                for m in ("is_extension_enabled", "is_enabled"):
                    if hasattr(mgr, m):
                        try:
                            if getattr(mgr, m)(ext_id):
                                enabled = True
                                break
                        except Exception:
                            pass
                path = meta.get('path') or meta.get('location') or '?'
                print(f'   - {ext_id} enabled={enabled} version={meta.get("version", "?")} path={path}')
    except Exception as e:
        print(f'[ROS2_BRIDGE][EXT] Dump failed: {e}')

def _force_ros2_bridge(verbose=True):
    if DISABLE_FORCE_ROS_BRIDGE:
        if verbose:
            print("[ROS2_BRIDGE] Force enable skipped (DISABLE_FORCE_ROS_BRIDGE=1)")
        return False
    try:
        import omni.kit.app
        app_obj = omni.kit.app.get_app()
        mgr = app_obj.get_extension_manager() if hasattr(app_obj, 'get_extension_manager') else None
        if not mgr:
            return False
        
        ext_list = []
        try:
            if hasattr(mgr, 'get_extensions'):
                ext_list = list(getattr(mgr, 'get_extensions')().keys())
            else:
                # fallback union method
                for attr in ("get_enabled_extension_ids", "get_disabled_extension_ids"):
                    if hasattr(mgr, attr):
                        try:
                            ext_list.extend(getattr(mgr, attr)())
                        except Exception:
                            pass
            ext_list = list(set(ext_list))
        except Exception:
            pass

        for target in CRITICAL_ROS_EXT_NAMES:
            present = (target in ext_list) if ext_list else True
            if not present:
                if verbose:
                    print(f"[ROS2_BRIDGE] {target} not yet discovered")
                continue
            already = False
            for m in ("is_extension_enabled", "is_enabled"):
                if hasattr(mgr, m):
                    try:
                        if getattr(mgr, m)(target):
                            already = True
                            break
                    except Exception:
                        continue
            if already:
                if verbose:
                    print(f"[ROS2_BRIDGE] {target} already enabled")
                return True
            for m in ("set_extension_enabled_immediate", "set_extension_enabled"):
                if hasattr(mgr, m):
                    try:
                        getattr(mgr, m)(target, True)
                        if verbose:
                            print(f"[ROS2_BRIDGE] Enabled {target}")
                        return True
                    except Exception:
                        continue
            if verbose:
                print(f"[ROS2_BRIDGE] Could not enable {target}")
        return False
    except Exception as e:
        if verbose:
            print(f"[ROS2_BRIDGE] Force enable failed: {e}")
    return False

def _wait_for_ros2_node_types(timeout=6.0, poll=0.2, required=None, verbose=True):
    """Wait until required ros2 bridge OmniGraph node types are registered.

    Env override:
      ROS_BRIDGE_REQUIRED_NODES=comma,list (full type names or short names without prefix)
      ROS_BRIDGE_NODE_WAIT_TIMEOUT=seconds
    """
    if DISABLE_FORCE_ROS_BRIDGE:
        return False
    try:
        import omni.graph.core as og
    except Exception as e:
        if verbose:
            print(f"[ROS2_BRIDGE] OmniGraph core import failed (cannot wait nodes): {e}")
        return False

    # Build required list
    env_req = os.getenv("ROS_BRIDGE_REQUIRED_NODES", "")
    if required is None:
        if env_req.strip():
            raw = [x.strip() for x in env_req.split(',') if x.strip()]
        else:
            # Default commonly used nodes
            raw = [
                "ROS2Context",
                "ROS2PublishTransformTree",
                "ROS2CameraInfoHelper",
                "ROS2SubscribeJointState",
                "ROS2CameraHelper",
            ]
    else:
        raw = list(required)
    # Normalize to full names
    pref_new = "isaacsim.ros2.bridge."
    full_required = []
    for name in raw:
        if name.startswith(pref_new):
            full_required.append(name)
        elif name.startswith("omni.isaac.ros2_bridge."):
            # legacy style fully qualified
            full_required.append(name)
        else:
            full_required.append(pref_new + name)

    timeout = float(os.getenv("ROS_BRIDGE_NODE_WAIT_TIMEOUT", str(timeout)))
    deadline = time.time() + timeout
    missing_prev = set(full_required)
    # Some nodes may be registered under legacy prefix; include mapping attempts
    legacy_map = {r: r.replace("isaacsim.ros2.bridge.", "omni.isaac.ros2_bridge.") for r in full_required}

    def _is_registered(tn):
        try:
            nt = og.get_node_type_registry().get_node_type(tn)
            return nt is not None
        except Exception:
            return False

    while time.time() < deadline:
        missing = []
        for full in full_required:
            if _is_registered(full):
                continue
            # check legacy alias
            legacy = legacy_map.get(full)
            if legacy and _is_registered(legacy):
                continue
            missing.append(full)
        if not missing:
            if verbose:
                print(f"[ROS2_BRIDGE] All required node types registered: {len(full_required)}")
            return True
        # Print only if changed to reduce spam
        s_missing = set(missing)
        if verbose and s_missing != missing_prev:
            short = [m.split('.')[-1] for m in missing]
            print(f"[ROS2_BRIDGE] Waiting node types ({len(missing)} missing): {', '.join(short)}")
            missing_prev = s_missing
        simulation_app.update()
        time.sleep(poll)
    if verbose:
        print("[ROS2_BRIDGE] Node type registration timeout; still missing:")
        for m in missing_prev:
            print("  -", m)
    return False

def _dump_ros2_node_types(prefix_filters=("ros2", "ROS2", "isaacsim.ros2", "omni.isaac.ros2")):
    """Diagnostic: list currently registered OmniGraph node types containing any of the prefixes."""
    try:
        import omni.graph.core as og
        names = []
        reg = og.get_node_type_registry()
        # get_registered_node_type_names may differ across versions; try common APIs
        for attr in ("get_node_type_names", "get_registered_node_type_names", "get_all_node_type_names"):
            if hasattr(reg, attr):
                try:
                    names = list(getattr(reg, attr)())
                    break
                except Exception:
                    continue
        if not names:
            print("[ROS2_BRIDGE][DIAG] Could not enumerate node type names.")
            return
        filt = []
        for n in names:
            low = n.lower()
            if any(p.lower() in low for p in prefix_filters):
                filt.append(n)
        print(f"[ROS2_BRIDGE][DIAG] Registered ROS-related node types ({len(filt)}):")
        for n in sorted(filt):
            print("   -", n)
    except Exception as e:
        print(f"[ROS2_BRIDGE][DIAG] Dump failed: {e}")

def _post_enable_ros_bridge_check():
    """After initial enable attempt & wait, try fallback strategy if nodes missing."""
    if os.getenv("ROS_BRIDGE_WAIT_NODES", "1").lower() not in ("1", "true", "yes"):
        return
    # First attempt already executed outside; we just verify presence
    success = _wait_for_ros2_node_types(timeout=float(os.getenv("ROS_BRIDGE_NODE_WAIT_TIMEOUT", "6")),
                                        poll=0.3, verbose=False)
    if success:
        print("[ROS2_BRIDGE] Node types present after initial wait.")
        _dump_ros2_node_types()
        return
    print("[ROS2_BRIDGE] Initial node wait failed.")
    # Try auto graph extension enable if allowed
    if ROS_BRIDGE_AUTO_GRAPH:
        try:
            import omni.kit.app
            app = omni.kit.app.get_app()
            mgr = app.get_extension_manager()
            graph_like = [e for e in mgr.get_extensions().keys() if 'ros2.bridge' in e and 'graph' in e]
            enabled_any = False
            for ext_id in graph_like:
                already = False
                for m in ("is_extension_enabled", "is_enabled"):
                    if hasattr(mgr, m):
                        try:
                            if getattr(mgr, m)(ext_id):
                                already = True
                                break
                        except Exception:
                            pass
                if already:
                    continue
                for m in ("set_extension_enabled_immediate", "set_extension_enabled"):
                    if hasattr(mgr, m):
                        try:
                            getattr(mgr, m)(ext_id, True)
                            print(f"[ROS2_BRIDGE][AUTO_GRAPH] Enabled graph extension: {ext_id}")
                            enabled_any = True
                            break
                        except Exception as e:
                            print(f"[ROS2_BRIDGE][AUTO_GRAPH] Failed enable {ext_id}: {e}")
            if enabled_any:
                # Warm frames and re-check
                for _ in range(60):
                    simulation_app.update()
                    time.sleep(0.01)
                if _wait_for_ros2_node_types(timeout=5, poll=0.4, verbose=True):
                    print("[ROS2_BRIDGE] Node types appeared after auto graph enable.")
                    _dump_ros2_node_types()
                    return
        except Exception as e:
            print(f"[ROS2_BRIDGE][AUTO_GRAPH] Error during auto graph attempt: {e}")
    print("[ROS2_BRIDGE] Some node types may not be available. Dumping what exists:")
            _dump_ros2_node_types()

if SAFE_RENDER or DISABLE_RTX:
    try:
        import carb.settings
        s = carb.settings.get_settings()
        for k, v in [
            ("/rtx/enabled", False),
            ("/ngx/enabled", False),
            ("/app/renderer/resolution/width", 640),
            ("/app/renderer/resolution/height", 360),
            ("/app/renderer/vsync", False),
            ("/app/hydraEngine/hydraEngineDelegate", "Storm"),
            ("/app/asyncRendering/enabled", False),
            ("/app/renderer/multiGpu/enabled", False),
            ("/rtx/rendermode", 0),  # basic
        ]:
            try:
                s.set(k, v)
            except Exception:
                pass
        print("[SAFE_RENDER] Low render settings applied (RTX disabled)." if SAFE_RENDER else "[SAFE_RENDER] RTX forced OFF.")
    except Exception:
        pass

def enable_extensions():
    # Skip mass enabling if in minimal diagnostic mode to avoid GPU spike
    if os.getenv('MINIMAL_TEST', '0').lower() in ('1', 'true', 'yes'):
        print('[EXT] Skipping auto-enable (MINIMAL_TEST=1).')
        return
    if not AUTO_ENABLE_ALL_EXT:
        print('[EXT] AUTO_ENABLE_ALL_EXTENSIONS disabled.')
        return
    try:
        import omni.kit.app
        app_obj = omni.kit.app.get_app()
        mgr = None
        if app_obj and hasattr(app_obj, 'get_extension_manager'):
            try:
                mgr = app_obj.get_extension_manager()
            except Exception:
                mgr = None
        if mgr is None:
            try:
                import omni.ext
                mgr = getattr(omni.ext, 'get_extension_manager', lambda: None)()
            except Exception:
                mgr = None
        if mgr is None:
            print('[EXT] Extension manager not available.')
            return
        try:
            ext_dict = mgr.get_extensions()
            names = list(ext_dict.keys())
        except Exception:
            names = []
            for attr in ("get_enabled_extension_ids", "get_disabled_extension_ids"):
                if hasattr(mgr, attr):
                    try:
                        names.extend(getattr(mgr, attr)())
                    except Exception:
                        pass
            names = list(set(names))
        force_first = ["omni.isaac.ros2_bridge"]
        ordered = []
        for n in force_first + names:
            if n not in ordered:
                ordered.append(n)

        def _is_enabled(n):
            for m in ("is_extension_enabled", "is_enabled"):
                if hasattr(mgr, m):
                    try:
                        return bool(getattr(mgr, m)(n))
                    except Exception:
                        continue
            return False

        def _enable(n):
            for m in ("set_extension_enabled_immediate", "set_extension_enabled"):
                if hasattr(mgr, m):
                    try:
                        getattr(mgr, m)(n, True)
                        return True
                    except Exception:
                        continue
            return False

        new_cnt = 0
        skipped = 0
        for name in ordered:
            if EXT_PREFIX and not name.startswith(EXT_PREFIX):
                continue
            if name in EXT_EXCLUDE:
                skipped += 1
                continue
            if EXT_ALLOWLIST and name not in EXT_ALLOWLIST:
                skipped += 1
                continue
            if EXT_BLOCKLIST and name in EXT_BLOCKLIST:
                skipped += 1
                continue
            if not _is_enabled(name):
                if _enable(name):
                    new_cnt += 1
                else:
                    skipped += 1
        print(f"[EXT] enabled_new={new_cnt} skipped={skipped} allowlist={'ON' if EXT_ALLOWLIST else 'OFF'} blocklist={'ON' if EXT_BLOCKLIST else 'OFF'}")
    except Exception as e:
        print(f"[EXT] enable failed: {e}")

ROS_BRIDGE_ENABLE_STRATEGY = os.getenv("ROS_BRIDGE_ENABLE_STRATEGY", "post_stage")
ROS_BRIDGE_WARMUP_FRAMES = int(os.getenv("ROS_BRIDGE_WARMUP_FRAMES", "30"))
ROS_BRIDGE_RETRY_SEC = float(os.getenv("ROS_BRIDGE_RETRY_SEC", "3"))
ROS_BRIDGE_WITH_STAGE_DELAY_FRAMES = int(os.getenv("ROS_BRIDGE_WITH_STAGE_DELAY_FRAMES", "5"))
ROS_BRIDGE_PRE_STAGE_WARMUP_FRAMES = int(os.getenv("ROS_BRIDGE_PRE_STAGE_WARMUP_FRAMES", "10"))

# Early extension enabling (but not forcing ros bridge unless strategy says so)
enable_extensions()
_env_dump_for_ros()
_dump_all_ros_extensions()

if ROS_BRIDGE_ENABLE_STRATEGY in ("early", "pre_stage") and not DISABLE_FORCE_ROS_BRIDGE:
    if ROS_BRIDGE_ENABLE_STRATEGY == "pre_stage":
        print(f"[ROS2_BRIDGE] Strategy=pre_stage: warming {ROS_BRIDGE_PRE_STAGE_WARMUP_FRAMES} frames, then enabling BEFORE stage load.")
        for i in range(max(0, ROS_BRIDGE_PRE_STAGE_WARMUP_FRAMES)):
            simulation_app.update()
            time.sleep(0.01)
    else:
        print("[ROS2_BRIDGE] Strategy=early: attempting enable immediately BEFORE stage load.")
    deadline = time.time() + ROS_BRIDGE_RETRY_SEC
    tries = 0
    while time.time() < deadline:
        if _force_ros2_bridge(verbose=(tries == 0)):
            if os.getenv("ROS_BRIDGE_WAIT_NODES", "1").lower() in ("1", "true", "yes"):
                _wait_for_ros2_node_types()
            break
        # Give a couple of frames so internal registries & extension machinery settle
        simulation_app.update()
        time.sleep(0.2)
        tries += 1
else:
    if ROS_BRIDGE_ENABLE_STRATEGY == "with_stage":
        print("[ROS2_BRIDGE] Strategy=with_stage: will attempt enable during stage load loop after delay frames.")
    elif ROS_BRIDGE_ENABLE_STRATEGY == "pre_stage":
        # pre_stage handled above (only executes if not DISABLE_FORCE_ROS_BRIDGE)
        pass
    else:
        print(f"[ROS2_BRIDGE] Strategy={ROS_BRIDGE_ENABLE_STRATEGY}: will defer forcing until after stage load.")

# ----------------------------------------------------------

import omni.usd
usd_ctx = omni.usd.get_context()
SKIP_STAGE = os.getenv("SKIP_STAGE", "0").lower() in ("1", "true", "yes")
if not SKIP_STAGE:
    print(f"[SCENE] Opening USD: {USD_PATH}")
    usd_ctx.open_stage(USD_PATH)
else:
    print("[SCENE] SKIP_STAGE=1 -> Stage will not be opened here.")

if not SKIP_STAGE:
    start = time.time()
    _with_stage_frame = 0
    _ros_bridge_attempted_midload = False
    _ros_bridge_midload_ok = False
    while True:
        st = usd_ctx.get_stage()
        if st:
            try:
                if hasattr(usd_ctx, 'is_stage_loading') and not usd_ctx.is_stage_loading():
                    break
                if time.time() - start > 5:
                    print("[SCENE] Stage load wait timeout (5s) -> continuing")
                    break
            except Exception:
                break
        # Attempt mid-load enable if strategy = with_stage
        if ROS_BRIDGE_ENABLE_STRATEGY == "with_stage" and not DISABLE_FORCE_ROS_BRIDGE and not _ros_bridge_attempted_midload:
            if _with_stage_frame >= ROS_BRIDGE_WITH_STAGE_DELAY_FRAMES:
                _ros_bridge_attempted_midload = True
                print(f"[ROS2_BRIDGE] (with_stage) Enabling ROS bridge during load at frame {_with_stage_frame}...")
                _ros_bridge_midload_ok = bool(_force_ros2_bridge(verbose=True))
                if _ros_bridge_midload_ok and os.getenv("ROS_BRIDGE_WAIT_NODES", "1").lower() in ("1", "true", "yes"):
                    _wait_for_ros2_node_types(timeout=float(os.getenv("ROS_BRIDGE_NODE_WAIT_TIMEOUT", "4")), poll=0.25)
            else:
                _with_stage_frame += 1
        simulation_app.update()
        time.sleep(0.02)
    print("[SCENE] Stage ready.")
    if ROS_BRIDGE_ENABLE_STRATEGY == "with_stage" and not DISABLE_FORCE_ROS_BRIDGE:
        if not _ros_bridge_midload_ok:
            print("[ROS2_BRIDGE] (with_stage) Mid-load enable failed or skipped; retrying post-stage.")
        else:
            print("[ROS2_BRIDGE] (with_stage) Bridge already enabled during load; post-stage retry skipped.")
    if ROS_BRIDGE_ENABLE_STRATEGY != "early" and not DISABLE_FORCE_ROS_BRIDGE and (ROS_BRIDGE_ENABLE_STRATEGY != "with_stage" or not _ros_bridge_midload_ok):
        # Warmup a few frames first to reduce race with renderer / physx init
        for _ in range(max(0, ROS_BRIDGE_WARMUP_FRAMES)):
            simulation_app.update()
            time.sleep(0.005)
        deadline = time.time() + ROS_BRIDGE_RETRY_SEC
        attempt = 0
        while time.time() < deadline:
            ok_bridge = _force_ros2_bridge(verbose=(attempt == 0))
            if ok_bridge:
                if os.getenv("ROS_BRIDGE_WAIT_NODES", "1").lower() in ("1", "true", "yes"):
                    _wait_for_ros2_node_types()
                break
            simulation_app.update()
            time.sleep(0.3)
            attempt += 1
        else:
            print("[ROS2_BRIDGE] Could not enable bridge within retry window.")
    # List sublayers to aid diagnostics
    try:
        st = usd_ctx.get_stage()
        if st:
            root = st.GetRootLayer()
            subs = list(root.subLayerPaths)
            print(f"[SCENE] Root layer: {root.identifier}")
            if subs:
                print("[SCENE] Sublayers (order):")
                for i, p in enumerate(subs):
                    print(f"  [{i}] {p}")
            else:
                print("[SCENE] No sublayers detected.")
    except Exception as e:
        print(f"[SCENE] Could not list sublayers: {e}")
else:
    print("[SCENE] Stage load loop skipped (SKIP_STAGE=1).")

# Post-stage ROS bridge check
if not DISABLE_FORCE_ROS_BRIDGE:
    _post_enable_ros_bridge_check()

# Main idle loop
dt = 1.0 / max(1.0, IDLE_FPS)
try:
    while True:
        simulation_app.update()
        time.sleep(dt)
except KeyboardInterrupt:
    pass
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass
    print("[EXIT] Closed.")

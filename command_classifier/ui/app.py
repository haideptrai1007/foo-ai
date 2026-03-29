from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from command_classifier.config import (
    AUDIO_DURATION_S,
    CHECKPOINT_DIR,
    EXPORT_DIR,
    F_MAX,
    F_MIN,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_gradio():
    try:
        import gradio as gr
        return gr
    except Exception as e:
        raise ImportError("gradio is required for the UI.") from e


def _save_clips_to_dir(clips: List[Tuple[int, np.ndarray]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (sr, arr) in enumerate(clips):
        if arr.ndim > 1:
            arr = arr.mean(axis=-1)
        arr = arr.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 32768.0
        sf.write(str(out_dir / f"clip_{i:04d}.wav"), arr, sr)


def _trim_silence(arr: np.ndarray, sr: int, top_db: float = 30.0) -> np.ndarray:
    mono = arr.mean(axis=-1) if arr.ndim > 1 else arr
    frame_len = max(1, int(sr * 0.02))
    pad = (-len(mono)) % frame_len
    padded = np.concatenate([mono, np.zeros(pad, dtype=mono.dtype)])
    frames = padded.reshape(-1, frame_len)
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    peak_rms = rms.max()
    if peak_rms == 0:
        return arr
    threshold = peak_rms * (10.0 ** (-top_db / 20.0))
    active = np.where(rms >= threshold)[0]
    if len(active) == 0:
        return arr
    start = int(active[0]) * frame_len
    end = min(len(mono), int(active[-1] + 1) * frame_len)
    min_len = max(1, int(sr * 0.1))
    if end - start < min_len:
        center = (start + end) // 2
        start = max(0, center - min_len // 2)
        end = min(len(mono), start + min_len)
    return arr[start:end] if arr.ndim == 1 else arr[start:end]


def _commands_table(state: Dict) -> List[List]:
    """Render state as a list-of-rows for gr.Dataframe."""
    rows = []
    for cmd, clips in state.get("commands", {}).items():
        rows.append([cmd, len(clips)])
    return rows or [["—", 0]]


def _similarity_table(similarities: Dict[str, float]) -> List[List]:
    rows = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return [[cmd, f"{sim:.3f}"] for cmd, sim in rows]


def _build_prototype(method: str, command_waveforms: Dict[str, Any]) -> Any:
    """Instantiate and fit the selected prototype method."""
    from command_classifier.prototype.logmel import LogMelPrototype
    from command_classifier.prototype.logmel_delta import LogMelDeltaPrototype
    from command_classifier.prototype.pretrained import PretrainedEmbeddingPrototype

    if method == "logmel":
        proto = LogMelPrototype()
    elif method == "logmel_delta":
        proto = LogMelDeltaPrototype()
    elif method == "pretrained":
        proto = PretrainedEmbeddingPrototype()
    else:
        raise ValueError(f"Unknown method: {method}")

    proto.fit(command_waveforms)
    return proto


def _waveforms_from_state(state: Dict) -> Dict[str, Any]:
    """Convert recorded clips in state to waveform tensors per command."""
    import torch
    from command_classifier.preprocessing.audio import load_from_gradio

    result: Dict[str, Any] = {}
    for cmd, clips in state.get("commands", {}).items():
        wavs = []
        for clip in clips:
            try:
                wavs.append(load_from_gradio(clip))
            except Exception:
                pass
        if wavs:
            result[cmd] = wavs
    return result


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def create_app():
    gr = _load_gradio()

    with gr.Blocks(title="Few-Shot Multi-Command Classifier") as demo:
        # State must live inside Blocks context to avoid KeyError: 0
        clips_state = gr.State({"commands": {}})
        proto_state = gr.State(None)
        method_state = gr.State("logmel_delta")
        threshold_state = gr.State(0.75)

        gr.Markdown("# Few-Shot Multi-Command Classifier")
        gr.Markdown(
            "Record 3–5 clips per command → Build prototype → "
            "Identify commands in real time → Export"
        )

        # ── Tab 1: Record ────────────────────────────────────────────────
        with gr.Tab("1 · Record"):
            gr.Markdown(
                "Type a command name, record a clip, then click **Add Clip**. "
                "Repeat for every command you want to recognise."
            )

            with gr.Row():
                cmd_name_box = gr.Textbox(
                    label="Command name",
                    placeholder='e.g. "lights on"',
                    scale=2,
                )
                n_target = gr.Number(label="Target clips / command", value=5, precision=0, scale=1)

            mic_audio = gr.Audio(sources=["microphone"], type="numpy", label="Microphone")

            with gr.Row():
                btn_add = gr.Button("Add Clip", variant="primary")
                btn_remove = gr.Button("Remove Command", variant="secondary")
                btn_clear_all = gr.Button("Clear All", variant="stop")

            commands_table = gr.Dataframe(
                headers=["Command", "Clips"],
                datatype=["str", "number"],
                label="Recorded commands",
                interactive=False,
            )

            def _counter_text(state, target):
                n_cmds = len(state.get("commands", {}))
                total_clips = sum(len(v) for v in state.get("commands", {}).values())
                return f"{n_cmds} command(s), {total_clips} clip(s) recorded"

            def _cmd_choices(state):
                return gr.update(choices=list(state.get("commands", {}).keys()))

            def add_clip(cmd_name, audio, state, target):
                cmd = cmd_name.strip().lower() if cmd_name else ""
                if not cmd or audio is None:
                    return state, _commands_table(state), _counter_text(state, target), _cmd_choices(state)
                sr, arr = audio
                arr = _trim_silence(arr.copy(), sr)
                new_state = {**state, "commands": dict(state["commands"])}
                clips = list(new_state["commands"].get(cmd, []))
                if len(clips) < int(target):
                    clips = clips + [(sr, arr)]
                new_state["commands"][cmd] = clips
                return new_state, _commands_table(new_state), _counter_text(new_state, target), _cmd_choices(new_state)

            def remove_command(cmd_name, state, target):
                cmd = cmd_name.strip().lower() if cmd_name else ""
                new_state = {**state, "commands": dict(state["commands"])}
                new_state["commands"].pop(cmd, None)
                return new_state, _commands_table(new_state), _counter_text(new_state, target), _cmd_choices(new_state)

            def clear_all(state, target):
                new_state = {"commands": {}}
                return new_state, _commands_table(new_state), _counter_text(new_state, target), _cmd_choices(new_state)

            counter_display = gr.Textbox(
                label="Status", value="0 command(s), 0 clip(s) recorded", interactive=False
            )

            gr.Markdown("### Playback — hear your recordings")
            with gr.Row():
                preview_cmd = gr.Dropdown(
                    label="Command", choices=[], interactive=True, scale=3,
                )
                preview_idx = gr.Number(
                    label="Clip # (1-based)", value=1, precision=0, minimum=1, scale=1,
                )
            btn_preview = gr.Button("Play Clip")
            preview_audio = gr.Audio(label="Playback", interactive=False)

            def play_clip(state, cmd, idx):
                clips = state.get("commands", {}).get(cmd, [])
                if not clips:
                    return None
                i = max(0, min(int(idx) - 1, len(clips) - 1))
                sr, arr = clips[i]
                arr = arr.astype(np.float32)
                if arr.max() > 1.0:
                    arr = arr / 32768.0
                return (sr, arr)

            btn_preview.click(
                fn=play_clip,
                inputs=[clips_state, preview_cmd, preview_idx],
                outputs=[preview_audio],
            )

            btn_add.click(
                fn=add_clip,
                inputs=[cmd_name_box, mic_audio, clips_state, n_target],
                outputs=[clips_state, commands_table, counter_display, preview_cmd],
            )
            btn_remove.click(
                fn=remove_command,
                inputs=[cmd_name_box, clips_state, n_target],
                outputs=[clips_state, commands_table, counter_display, preview_cmd],
            )
            btn_clear_all.click(
                fn=clear_all,
                inputs=[clips_state, n_target],
                outputs=[clips_state, commands_table, counter_display, preview_cmd],
            )

        # ── Tab 2: Build ─────────────────────────────────────────────────
        with gr.Tab("2 · Build Prototypes"):
            gr.Markdown(
                "Choose an embedding method and build a prototype for each command. "
                "No training loop — this completes in seconds (except pretrained on first run)."
            )

            method_radio = gr.Radio(
                choices=[
                    ("Log-Mel  (fast, 40-dim)", "logmel"),
                    ("Log-Mel + Delta  (robust, 120-dim)", "logmel_delta"),
                    ("Pretrained wav2vec2  (best, 768-dim — downloads ~360 MB once)", "pretrained"),
                ],
                value="logmel_delta",
                label="Embedding method",
            )
            threshold_slider = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.75, step=0.01,
                label="Rejection threshold  (similarity below this → 'none')",
            )
            build_btn = gr.Button("Build Prototypes", variant="primary")
            build_log = gr.Textbox(label="Log", lines=6, interactive=False)
            build_stats = gr.JSON(label="Prototype summary")

            def on_build(state, method, threshold):
                import queue as _q

                if not state.get("commands"):
                    yield "No commands recorded.", None, None, threshold
                    return

                log_q: "queue.Queue[Optional[str]]" = _q.Queue()
                result: list = [None, None]

                def _run():
                    try:
                        log_q.put(f"Converting clips to waveforms...")
                        wavs = _waveforms_from_state(state)
                        if not wavs:
                            log_q.put("No valid audio found.")
                            result[1] = "No valid audio."
                            return
                        log_q.put(f"Building {method} prototype for {len(wavs)} command(s)...")
                        proto = _build_prototype(method, wavs)
                        stats = {
                            "method": method,
                            "commands": {cmd: len(state["commands"][cmd]) for cmd in proto.commands},
                            "embedding_dim": proto.embedding_dim,
                            "threshold": threshold,
                        }
                        result[0] = proto
                        result[1] = stats
                        log_q.put(f"Done. Embedding dim: {proto.embedding_dim}")
                        for cmd in proto.commands:
                            log_q.put(f"  {cmd}: {len(state['commands'][cmd])} clips")
                    except Exception as exc:
                        log_q.put(f"Error: {exc}")
                        result[1] = str(exc)
                    finally:
                        log_q.put(None)

                logs = []
                thread = threading.Thread(target=_run, daemon=True)
                thread.start()

                while True:
                    msg = log_q.get()
                    if msg is None:
                        break
                    logs.append(msg)
                    yield "\n".join(logs), None, None, threshold

                thread.join()

                proto_obj = result[0]
                stats = result[1]
                if proto_obj is None:
                    yield "\n".join(logs), None, None, threshold
                    return
                yield "\n".join(logs), stats, proto_obj, threshold

            build_btn.click(
                fn=on_build,
                inputs=[clips_state, method_radio, threshold_slider],
                outputs=[build_log, build_stats, proto_state, threshold_state],
            )
            method_radio.change(
                fn=lambda m: m,
                inputs=[method_radio],
                outputs=[method_state],
            )

        # ── Tab 3: Test ──────────────────────────────────────────────────
        with gr.Tab("3 · Test"):
            gr.Markdown(
                "Record a clip. The classifier scores it against every command prototype "
                "and returns the best match (or 'none' if below threshold)."
            )

            test_threshold = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.75, step=0.01, label="Threshold"
            )
            test_audio = gr.Audio(sources=["microphone"], type="numpy", label="Microphone")
            test_btn = gr.Button("Identify Command", variant="primary")

            with gr.Row():
                test_result = gr.Textbox(label="Best match", interactive=False, scale=2)
                test_similarity = gr.Number(label="Similarity", interactive=False, scale=1)
                test_latency = gr.Number(label="Latency (ms)", interactive=False, scale=1)

            test_table = gr.Dataframe(
                headers=["Command", "Similarity"],
                datatype=["str", "str"],
                label="All scores (sorted)",
                interactive=False,
            )

            def on_test(audio, proto, threshold):
                if audio is None:
                    return "No audio recorded.", 0.0, 0.0, []
                if proto is None:
                    return "Build prototypes first (Tab 2).", 0.0, 0.0, []
                from command_classifier.preprocessing.audio import load_from_gradio

                try:
                    waveform = load_from_gradio(audio)
                except Exception as e:
                    return f"Audio error: {e}", 0.0, 0.0, []

                t0 = time.perf_counter()
                best_cmd, best_sim, all_sims = proto.predict(waveform, threshold=float(threshold))
                latency_ms = (time.perf_counter() - t0) * 1000.0

                label = f"COMMAND: {best_cmd}" if best_cmd != "none" else "none (below threshold)"
                table = _similarity_table(all_sims)
                return label, round(best_sim, 4), round(latency_ms, 1), table

            test_btn.click(
                fn=on_test,
                inputs=[test_audio, proto_state, test_threshold],
                outputs=[test_result, test_similarity, test_latency, test_table],
            )

        # ── Tab 4: Export ────────────────────────────────────────────────
        with gr.Tab("4 · Export"):
            gr.Markdown(
                "Export prototypes as a `.npz` bundle. "
                "Inference on-device needs only `numpy` + the mel pipeline — no model file."
            )

            export_dir_box = gr.Textbox(
                label="Output directory", value=str(EXPORT_DIR / "prototype")
            )
            export_threshold_slider = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.75, step=0.01,
                label="Threshold to embed in config"
            )
            export_btn = gr.Button("Export", variant="primary")
            export_result = gr.JSON(label="Export result")

            def on_export(proto, method, threshold, out_dir):
                if proto is None:
                    return {"error": "Build prototypes first (Tab 2)."}
                out = Path(out_dir)
                out.mkdir(parents=True, exist_ok=True)
                npz_path = out / "prototypes.npz"
                audio_config = {
                    "sample_rate": SAMPLE_RATE,
                    "audio_duration_s": AUDIO_DURATION_S,
                    "n_fft": N_FFT,
                    "hop_length": HOP_LENGTH,
                    "n_mels": N_MELS,
                    "f_min": F_MIN,
                    "f_max": F_MAX,
                }
                proto.save(
                    npz_path=npz_path,
                    method_name=method,
                    threshold=float(threshold),
                    audio_config=audio_config,
                )
                return {
                    "prototypes_npz": str(npz_path),
                    "config_json": str(npz_path.with_suffix(".json")),
                    "commands": proto.commands,
                    "embedding_dim": proto.embedding_dim,
                    "method": method,
                    "threshold": float(threshold),
                }

            export_btn.click(
                fn=on_export,
                inputs=[proto_state, method_state, export_threshold_slider, export_dir_box],
                outputs=[export_result],
            )

    return demo


def main() -> None:  # pragma: no cover
    demo = create_app()
    demo.launch()


if __name__ == "__main__":  # pragma: no cover
    main()

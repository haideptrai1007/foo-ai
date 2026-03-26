from __future__ import annotations

import argparse

from command_classifier.ui.app import create_app


def main() -> None:
    ap = argparse.ArgumentParser(description="Few-Shot Audio Command Classifier UI")
    ap.add_argument(
        "--ckpt",
        default=None,
        help="Path to a Stage A pretrained backbone .pt file. If omitted, uses raw ImageNet MobileNetV3 weights.",
    )
    ap.add_argument("--port", type=int, default=7860, help="Gradio server port (default: 7860).")
    ap.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    args = ap.parse_args()

    demo = create_app(backbone_ckpt=args.ckpt)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

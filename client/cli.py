"""Minimal CLI to call the FastAPI server."""
from __future__ import annotations
import argparse, requests

def main() -> None:
    """Entry point for the simple client CLI."""
    p = argparse.ArgumentParser("pv-client")
    p.add_argument("--server", default="http://127.0.0.1:8000")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("health")

    p1 = sub.add_parser("train-cnn")
    p1.add_argument("--data-root", default=None)
    p1.add_argument("--batch-size", type=int, default=32)
    p1.add_argument("--epochs", type=int, default=30)
    p1.add_argument("--lr", type=float, default=1e-4)
    p1.add_argument("--l2", type=float, default=0.0)

    p2 = sub.add_parser("train-vgg19")
    p2.add_argument("--data-root", default=None)
    p2.add_argument("--batch-size", type=int, default=32)
    p2.add_argument("--phase1-epochs", type=int, default=8)
    p2.add_argument("--phase2-epochs", type=int, default=30)
    p2.add_argument("--fine-tune-from-block", type=int, default=4, choices=[4,5])
    p2.add_argument("--lr1", type=float, default=1e-3)
    p2.add_argument("--lr2", type=float, default=1e-5)
    p2.add_argument("--dropout", type=float, default=0.3)
    p2.add_argument("--weight-decay", type=float, default=1e-4)

    args = p.parse_args()
    base = args.server.rstrip("/")

    if args.cmd == "health":
        print(requests.get(f"{base}/health").json()); return

    if args.cmd == "train-cnn":
        payload = {
            "data_root": args.data_root, "batch_size": args.batch_size,
            "epochs": args.epochs, "lr": args.lr, "l2": args.l2
        }
        print(requests.post(f"{base}/train/cnn", json=payload).json()); return

    if args.cmd == "train-vgg19":
        payload = {
            "data_root": args.data_root, "batch_size": args.batch_size,
            "phase1_epochs": args.phase1_epochs, "phase2_epochs": args.phase2_epochs,
            "fine_tune_from_block": args.fine_tune_from_block, "lr1": args.lr1, "lr2": args.lr2,
            "dropout": args.dropout, "weight_decay": args.weight_decay
        }
        print(requests.post(f"{base}/train/vgg19", json=payload).json()); return

    p.print_help()

if __name__ == "__main__":
    main()

import argparse
from DDGLNET import DDGLConfig, DDGLTrainer
from DDGLNET.data import create_daily_loader, load_pickle

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate DDGL-Net.")
    parser.add_argument("--train", required=True, help="Path to training pickle file.")
    parser.add_argument("--valid", required=True, help="Path to validation pickle file.")
    parser.add_argument("--test", required=True, help="Path to test pickle file.")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--device", default=None, help="Override device, e.g. cpu or cuda.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    config = DDGLConfig(checkpoint_dir=args.checkpoint_dir)
    if args.device is not None:
        config.device = args.device
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.seed is not None:
        config.seed = args.seed

    print("Loading datasets...")
    train_data = load_pickle(args.train)
    valid_data = load_pickle(args.valid)
    test_data = load_pickle(args.test)

    train_loader = create_daily_loader(train_data, shuffle=True, drop_last=True)
    valid_loader = create_daily_loader(valid_data, shuffle=False, drop_last=True)
    test_loader = create_daily_loader(test_data, shuffle=False, drop_last=False)

    trainer = DDGLTrainer(config)
    trainer.fit(train_loader, valid_loader)

    predictions, metrics = trainer.predict(test_loader, test_data.get_index())
    predictions.to_pickle("ddgl_predictions.pkl")

    print("Test metrics")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()

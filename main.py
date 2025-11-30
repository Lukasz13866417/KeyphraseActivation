import sys

from pipeline import TrainingConfig, TrainingResult, run_training_pipeline


def main():
    key_phrase = input("Keyphrase to model: ").strip()
    if not key_phrase:
        print("No keyphrase provided, exiting.")
        sys.exit(1)

    print("Starting data generation and model training...")
    config = TrainingConfig(
        key_phrase=key_phrase,
        num_confusers=100,
        num_positives=100,
        num_inbetween=150,
        num_plain_negatives=100,
        num_piper_per=10,
        num_bark_per=0,
        num_kokoro_per=3,
        num_eleven_per=2,
        num_tps_random=0,
        growth_constant=5,
        epochs=1000,
        batch_size=8,
    )

    try:
        result: TrainingResult = run_training_pipeline(config)
    except Exception as exc:  # pragma: no cover - CLI feedback
        print(f"Training run failed: {exc}")
        sys.exit(1)

    print(
        f"\nTraining complete!\n"
        f"- Model saved to: {result.model_path}\n"
        f"- Train samples: {result.train_samples}, Val samples: {result.val_samples}\n"
        f"- Final metrics: train_loss={result.train_loss:.4f} "
        f"val_loss={result.val_loss:.4f} val_f1={result.val_f1:.4f}"
    )


if __name__ == "__main__":
    main()

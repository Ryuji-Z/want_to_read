import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def main() -> None:
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(1,)),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(
        optimizer="sgd",
        loss="mean_squared_error",
        metrics=["mae"],
    )

    x_train = np.array([-1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32).reshape(-1, 1)
    y_train = np.array([-3.0, -1.0, 1.0, 3.0, 5.0], dtype=np.float32).reshape(-1, 1)

    x_test = np.array([4.0, 5.0, 10.0], dtype=np.float32).reshape(-1, 1)
    y_test = np.array([7.0, 9.0, 19.0], dtype=np.float32).reshape(-1, 1)

    print("Start training...")
    history = model.fit(x_train, y_train, epochs=100, verbose=2)

    print("\nLoss function:", model.loss)
    print(f"Final training loss: {history.history['loss'][-1]:.6f}")

    eval_loss, eval_mae = model.evaluate(x_test, y_test, verbose=0)
    print(f"Evaluation loss: {eval_loss:.6f}")
    print(f"Evaluation MAE : {eval_mae:.6f}")

    prediction = model.predict(np.array([[10.0]], dtype=np.float32), verbose=0)
    print(f"Prediction for x=10.0: {prediction[0][0]:.4f}")

    x_line = np.linspace(-1.0, 10.0, 200, dtype=np.float32).reshape(-1, 1)
    y_line = model.predict(x_line, verbose=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], color="tab:blue")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(
        x_train.flatten(),
        y_train.flatten(),
        label="Train Data",
        color="tab:orange",
    )
    axes[1].plot(
        x_line.flatten(),
        y_line.flatten(),
        label="Model Prediction",
        color="tab:green",
    )
    axes[1].set_title("Linear Regression Fit")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    output_file = "tensorflow_training_plot.png"
    fig.savefig(output_file, dpi=150)
    print(f"Saved graph: {output_file}")
    plt.show()


if __name__ == "__main__":
    main()

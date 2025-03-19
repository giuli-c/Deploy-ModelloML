import mlflow
import mlflow.pytorch

class LoggerMLFLOW:
    @staticmethod
    def log_training_run(model, tokenizer, metrics, df_results, config):
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        with mlflow.start_run():
            mlflow.log_params({
                "model": config["model"]["pre_trained"],
                "epochs": config["training"].get("num_train_epochs", 10),
                "batch_size": config["training"].get("batch_size", 8),
                "early_stopping": config["training"].get("early_stopping", False)
            })
            mlflow.log_metrics(metrics)
            mlflow.pytorch.log_model(model, "model")
            tokenizer.save_pretrained("models/tokenizer")
            mlflow.log_artifacts("models/tokenizer", artifact_path="tokenizer")
            df_results.to_csv("predizioni.csv", index=False)
            mlflow.log_artifact("predizioni.csv")
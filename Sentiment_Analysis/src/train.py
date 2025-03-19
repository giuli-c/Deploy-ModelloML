
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback, TrainerCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class AccuracyThresholdCallback(TrainerCallback):
    def __init__(self, threshold=0.90):
        self.threshold = threshold

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and metrics.get("eval_accuracy", 0) >= self.threshold:
            print(f"Accuracy soglia raggiunta ({metrics['eval_accuracy']:.2f} ≥ {self.threshold}) - STOP training")
            control.should_early_stop = True
            control.should_save = True
        return control

class TrainerManager:
    def __init__(self, config, model, tokenizer, train_dataset, test_dataset):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def compute_metrics(self, pred):
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(pred.label_ids, preds, average="weighted")
        acc = accuracy_score(pred.label_ids, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    def tokenize_function(example):
        return tokenizer(example["text"], padding="max_length", truncation=True)

    def train(self):
        training_args = TrainingArguments(
            output_dir=self.config["training"].get("output_dir", "./results"),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=self.config["training"].get("num_train_epochs", 80),
            per_device_train_batch_size=self.config["training"].get("batch_size", 8),
            per_device_eval_batch_size=self.config["training"].get("batch_size", 8),
            logging_dir="./logs",
            logging_steps=10,
            report_to="none",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy"
        )

        callbacks = [AccuracyThresholdCallback()]
        if self.config["training"].get("early_stopping", False):
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config["training"].get("early_stopping_patience", 2)
                )
            )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            compute_metrics=self.compute_metrics,
            callbacks=callbacks
        )
        trainer.train()

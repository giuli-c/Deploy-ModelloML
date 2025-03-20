from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback, TrainerCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import configparser
from datasets import load_dataset
from preprocess_data import PreprocessData
from load_config import LoadConfig
from huggingface_hub import login
import os

class AccuracyThresholdCallback(TrainerCallback):
    def __init__(self, threshold=0.90):
        self.threshold = threshold

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and metrics.get("eval_accuracy", 0) >= self.threshold:
            print(f"Accuracy soglia raggiunta ({metrics['eval_accuracy']:.2f} ≥ {self.threshold}) - STOP training")
            control.should_early_stop = True
            control.should_save = True
        return control

class SentimentTrainer:
    def __init__(self, config):
        self.config_loader = LoadConfig("Sentiment_Analysis/config.yaml")
        self.config = config_loader.load_config()

        login(token=os.environ["HF_TOKEN"])

        self.dataset = load_dataset("cardiffnlp/tweet_sentiment_multilingual", "italian")

        self.model_name = self.config["model"].get("pre_trained", "cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        #config = AutoConfig.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        self.preprocess_dataset()

        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

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
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks
        )

    def compute_metrics(self, pred):
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(pred.label_ids, preds, average="weighted")
        acc = accuracy_score(pred.label_ids, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    def preprocess_function(self, examples):
        """
        Preprocesses the given examples using the tokenizer.

        Args:
            examples (dict): The examples to be preprocessed.

        Returns:
            dict: The preprocessed examples.
        """
        return tokenizer(example["text"], padding="max_length", truncation=True)
    def preprocess_dataset(self):
        """
        Preprocesses the dataset.

        This function tokenizes the dataset using the `preprocess_function` and stores the tokenized dataset in the `tokenized_dataset` attribute.

        Parameters:
            self (object): The instance of the class.
        
        Returns:
            None
        """
        # Il preprocessing vero e proprio lo faccio sul Dataset HuggingFace
        train_dataset = train_dataset.map(lambda x: {"text": PreprocessData.preprocess(x["text"])})
        test_dataset = test_dataset.map(lambda x: {"text": PreprocessData.preprocess(x["text"])})

        self.train_dataset = self.test_dataset.map(self.preprocess_function, batched=True)

    def train(self):
        """
        Train the model using the trainer object.

        Parameters:
            None

        Returns:
            None
        """
        self.trainer.train()

    def push_to_hub(self):
        """
        Pushes the current state of the trainer to the hub.

        This function calls the `push_to_hub` method of the `trainer` object.

        Parameters:
            None

        Returns:
            None
        """
        self.trainer.push_to_hub()


if __name__ == "__main__":
    trainer = SentimentTrainer("config.conf")
    trainer.train()
     #trainer.push_to_hub()
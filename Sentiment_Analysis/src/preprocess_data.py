class PreprocessData:
    @staticmethod
    def preprocess(text):
        """
        Pulisce il testo:
        - Sostituisce i tag @utente con '@user'
        - Sostituisce i link con 'http'
        """
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    @staticmethod
    def mapping_label():
        """
        Restituisce la mappatura delle etichette per FastText.
        """
        return {0: "negative", 1: "neutral", 2: "positive"}

    @staticmethod
    def check_dataset(train_df, test_df):
        df_train = train_df.to_pandas()
        df_test = test_df.to_pandas()
        print(" ---- DATASET TRAIN ---- ")
        df_train.info()
        print("------------------------------------")
        print("Elementi null: ", df_train.isnull().sum())
        print("Elementi duplicati: ", df_train.duplicated().sum())
        print("\n\n")
        print(" ---- DATASET TEST ---- ")
        df_train.info()
        print("------------------------------------")
        print("Elementi null: ", df_test.isnull().sum())
        print("Elementi duplicati: ", df_test.duplicated().sum())
        print(" ---- DATASET TRAIN shape: ", df_train.shape)
        print(" ---- DATASET TEST shape: ", df_test.shape)
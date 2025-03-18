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

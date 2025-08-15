from deepface.models.demography.Emotion import load_model


def load_emotion_model(config,weight_path=None):
    """
    Return an instance of EmotionClient with loaded model.
    """
    print(config)
    return load_model()

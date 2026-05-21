from keras.models import load_model

MODEL_PATH = r"C:\Users\joshk\OneDrive\Documents\GitHub_Strath\PolyVision\models\local\EfficientNetB0\best_model.keras"

m = load_model(MODEL_PATH)
print("Last layer type:", type(m.layers[-1]).__name__)
print("Last layer activation:", getattr(m.layers[-1], "activation", None))
m.summary()
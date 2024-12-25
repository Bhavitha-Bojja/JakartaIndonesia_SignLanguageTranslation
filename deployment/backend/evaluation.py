from app.preprocessing import VideoProcessor
from app.model import VideoClassifier
from tqdm import tqdm
import os
import json

# Initialize model
processor = VideoProcessor()
model = VideoClassifier()

total = 0
correct = 0
misclassified = []

for word in tqdm(os.listdir('test_videos'), desc='Processing words'):
    for video in os.listdir(f'test_videos/{word}'):
        video_path = f'test_videos/{word}/{video}'
        landmarks, angles = processor.process_video(video_path)
        prediction = model.predict(landmarks, angles)

        total += 1
        if prediction["label"] == word:
            correct += 1
        else:
            misclassified.append({
                'path': video_path,
                'actual': word,
                'predicted': prediction["label"],
                'confidence': round(prediction["confidence"], 2)
            })

eval_results = {
    "accuracy": f"{round(correct / total * 100)}%",
    "correct": correct,
    "total": total,
    "misclassified": misclassified
}

with open("eval_results.json", "w") as f:
    json.dump(eval_results, f, indent=4)

print(f"\nAccuracy: {eval_results['accuracy']}")
print(f"Correct predictions: {correct}/{total}")

print("\nMisclassified examples:")
for error in misclassified:
    print(f"Video: {error['path']}")
    print(f"Actual: {error['actual']}")
    print(f"Predicted: {error['predicted']} (confidence: {error['confidence']:.2f})")
    print("-" * 50)

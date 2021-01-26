import numpy as np
import datasets
bert_score = datasets.load_metric('bertscore')

test_data = np.load('data_for_bertscore.npy', allow_pickle=True)
score = bert_score.compute(predictions=test_data[0], references=test_data[1], lang='en'
                           )
print("Bert score F1 mean", score['f1'].mean().item())

for i in range(test_data.shape[1]):
    text = test_data[:, i]
    print(str(i) + '. Prediction: ', text[0])
    print(str(i) + '. Truth: ', text[1])

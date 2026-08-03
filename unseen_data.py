import json 
from transformers import pipeline
from dotenv import load_dotenv
import os
from pathlib import Path
load_dotenv()

class UnseenDataTests:
    '''Tests for evaluating the model on unseen data.to record '''
    def __init__(self, p_true=0, p_false=0):
        self.p_true = p_true
        self.p_false = p_false

    def add_true(self):
        self.p_true += 1

    def add_false(self):
        self.p_false += 1
    def current_states(self):
        return f"Values True{self.p_true}----Value False {self.p_false}, "
    def accurecy_percentage(self):
        return (((self.p_true)/(self.p_false+self.p_true))*100)

    def main(self, unseen_data_path, current_model_path):
        with open(unseen_data_path, "r") as f:
            data = json.load(f)

        nlp = pipeline("text-classification", model=current_model_path)
        for item in data:
            true_intent = item["label"]
            query = item["text"]
            result = nlp(query)[0]
            predicted_intent = result['label']
            if predicted_intent == true_intent:
                self.add_true()
            else:
                self.add_false()
            confidence = result['score']

            # we need to add a confusion matrix to evaluate the model's performance on this unseen dataset. The confusion matrix will help us understand how well the model is classifying the intents and where it might be making mistakes.#
            # a final confusion matrix here 
        accurecy_percentage = self.accurecy_percentage()
        return accurecy_percentage
    def read_and_decide(self, accuracy_comparison_file, current_acc_percentage, current_model_path):

        if (not os.path.exists(accuracy_comparison_file)or os.path.getsize(accuracy_comparison_file) == 0):
            self.write_to_json(
                accuracy_comparison_file,
                current_acc_percentage,
                current_model_path
            )
            return True, current_model_path


        try:
            with open(accuracy_comparison_file, "r") as file:
                data = json.load(file)

        except json.JSONDecodeError:
            print("Corrupted JSON file. Recreating...")
            self.write_to_json(
                accuracy_comparison_file,
                current_acc_percentage,
                current_model_path
            )
            return True, current_model_path


        previous_accuracy = data["current"]["accuracy"]

        if current_acc_percentage >= previous_accuracy:
            self.write_to_json(
                accuracy_comparison_file,
                current_acc_percentage,
                current_model_path
            )
            return True, current_model_path
        return False
    def write_to_json(
        self,
        accuracy_comparison_file,
        current_acc_percentage,
        current_model_path
    ):

        model_directory_name = Path(current_model_path).name

        json_obj = {
            "current": {
                "accuracy": current_acc_percentage,
                "model_path": model_directory_name
            }
        }

        with open(accuracy_comparison_file, "w") as f:
            json.dump(json_obj, f, indent=4)

            
# if __name__ == '__main__':
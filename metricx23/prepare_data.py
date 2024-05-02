import json

def process_json(input_file, output_file,flag):
    with open(input_file, 'r', encoding='utf-8') as file:
        updated_data = []
        for line in file:
            try:
                data = json.loads(line)
                if flag:
                    
                    entry = {
                        "source": data["src"],
                        "hypothesis": data["translation"],
                        "mqm":float(data["mqm_norm_score"]),
                        "da": float(data["da_norm_score"])
                    }
                else:
                    entry = {
                        "reference": data["ref"],
                        "hypothesis": data["translation"],
                        "mqm": float(data["mqm_norm_score"]),
                        "da": float(data["da_norm_score"])
                    }
                        
                updated_data.append(entry)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON in file: {input_file}")

    with open(output_file, 'w', encoding='utf-8') as output:
        for entry in updated_data:
            json.dump(entry, output, ensure_ascii=False)
            output.write('\n')

# Replace 'input.json' and 'output.json' with your input and output file paths
process_json('LLM_FT/extracted_spans/Indic_MT_eval_dataset/val_data.json', 'metricx/data/val_data_qe.jsonl',1)
print("done")
# python metricx/metricx23/prepare_data.py
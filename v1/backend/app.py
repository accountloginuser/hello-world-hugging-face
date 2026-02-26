
def my_routine(adata,atoken):

#
    import torch

#
    from datasets import load_dataset
    imdb = load_dataset("imdb")
    
#
#    small_train_dataset = imdb["train"].shuffle(seed=42).select([i for i in list(range(3000))])
#    small_test_dataset = imdb["test"].shuffle(seed=42).select([i for i in list(range(300))])
    small_train_dataset = imdb["train"].shuffle(seed=42).select([i for i in list(range(30))])
    small_test_dataset = imdb["test"].shuffle(seed=42).select([i for i in list(range(3))])
    
#    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
#
    def preprocess_function(examples):
       return tokenizer(examples["text"], truncation=True)
     
    tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
    tokenized_test = small_test_dataset.map(preprocess_function, batched=True)

#
    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

#
    import numpy as np
#    from datasets import load_metric
    import evaluate
     
    def compute_metrics(eval_pred):
#       load_accuracy = load_metric("accuracy")
#       load_f1 = load_metric("f1")
       load_accuracy = evaluate.load("accuracy")
       load_f1 = evaluate.load("f1")
       logits, labels = eval_pred
       predictions = np.argmax(logits, axis=-1)
       accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
       f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
       return {"accuracy": accuracy, "f1": f1}

#
#    from huggingface_hub import notebook_login
#    notebook_login()

#  
    from transformers import TrainingArguments, Trainer
     
    repo_name = "finetuning-sentiment-model-3000-samples"
 
    import os
    
    training_args = TrainingArguments(
       output_dir=repo_name,
       learning_rate=2e-5,
       per_device_train_batch_size=16,
       per_device_eval_batch_size=16,
       num_train_epochs=2,
       weight_decay=0.01,
       save_strategy="epoch",
#       push_to_hub=True,
       push_to_hub=False,
        dataloader_pin_memory=torch.cuda.is_available(),
        disable_tqdm=False,  # Set to True to disable progress bars
        dataloader_num_workers=os.cpu_count(),
    )
    
    trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=tokenized_train,
       eval_dataset=tokenized_test,
#       tokenizer=tokenizer,
        processing_class=tokenizer,
       data_collator=data_collator,
       compute_metrics=compute_metrics,
    )

#
    trainer.train()
    
#
    trainer.evaluate()
    
#
#    trainer.push_to_hub()
    trainer.push_to_hub(token=atoken)

#
    from transformers import pipeline
     
    sentiment_model = pipeline(model="username/"+repo_name)
#    sentiment_model(["I love this movie", "This movie sucks!"])
    aresults = sentiment_model([adata,])
    
    return aresults

from fastapi import FastAPI, Request

app = FastAPI()

@app.post("/items/")
async def create_item(request: Request):
    raw_body = await request.body()
    json_body = await request.json() # or use other methods based on content type
    aresponse = my_routine(adata=json_body["data"][0],atoken=json_body["accesstoken"],)

    import json
    astr = json.dumps(aresponse)

    return astr

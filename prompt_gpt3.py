import json
import requests
import csv
import argparse
import os
import time
import openai
import tqdm



def open_file(file_name):
  file = open(file_name, "r")
  data = json.load(file)
  file.close()
  return data

    
def get_answers_gpt3(args, question):
    prompt = 'Write 10 new wildly diverse questions with different words that have the same answer as "{0}"'
    response = openai.Completion.create(
                                        engine="text-davinci-002",
                                        prompt=prompt.format(question),
                                        temperature=0.8,
                                        max_tokens=350,
                                        top_p=1,
                                        frequency_penalty=0,
                                        presence_penalty=0
                                        )
    return response["choices"][0]["text"], response["usage"]["total_tokens"]
        

def get_answers_all_gpt3(args, data, dataset_name):
    all_generations = []
    total_tokens = 0
    for index, data_point in enumerate(tqdm.tqdm(data)):
        time.sleep(1)
        # if index:
        #     break
        generations, tokens = get_answers_gpt3(args, data_point["question"])
        total_tokens += tokens
        print(f"New tokens used: {tokens} for a total of {total_tokens} or an average of {total_tokens/(index+1)}")
        data_point["generated"] = generations
        all_generations.append(generations)
        
    file = open("data/" + dataset_name.upper() + "/" + dataset_name + "_w_generations.json", "w")
    json.dump(data, file, indent = 4)
    file.close()
    return all_generations
                

def generate_answers(args, dataset_name, API_TOKEN):
    file = "data/" + dataset_name.upper() + "/" + dataset_name + ".json"
    data = open_file(file)
    openai.api_key = API_TOKEN
    responses = get_answers_all_gpt3(args, data, dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation of Question Decomposition')
    parser.add_argument('--dataset_name',help='dataset to be used')
    parser.add_argument('--API_TOKEN',help='API token for the model to be used')
    args = parser.parse_args()
    
    generate_answers(args, args.dataset_name, args.API_TOKEN)
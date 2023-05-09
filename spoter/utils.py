
import logging
import torch
import csv
import wandb

def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):

    pred_correct, pred_all = 0, 0
    running_loss = 0.0

    data_length = len(dataloader)

    for i, data in enumerate(dataloader):
        inputs, labels, _ = data
        inputs = inputs.squeeze(0).to(device)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(inputs).expand(1, -1, -1)

        loss = criterion(outputs[0], labels[0])
        loss.backward()
        optimizer.step()
        running_loss += loss

        # Statistics
        if int(torch.argmax(torch.nn.functional.softmax(outputs, dim=2))) == int(labels[0][0]):
            pred_correct += 1
        pred_all += 1

    if scheduler:
        #scheduler.step(running_loss.item() / len(dataloader))
        scheduler.step()

    return running_loss/data_length, pred_correct, pred_all, (pred_correct / pred_all)


def evaluate(model, dataloader, cel_criterion, device, print_stats=False):

    pred_correct, pred_top_5,  pred_all = 0, 0, 0
    running_loss = 0.0
    
    stats = {i: [0, 0] for i in range(302)}

    data_length = len(dataloader)

    k = 5 # top 5 (acc)

    for i, data in enumerate(dataloader):
        inputs, labels, _ = data
        inputs = inputs.squeeze(0).to(device)
        labels = labels.to(device, dtype=torch.long)
        #print(f"iteration {i} in evaluate")
        outputs = model(inputs).expand(1, -1, -1)

        loss = cel_criterion(outputs[0], labels[0])
        running_loss += loss

        # Statistics
        if int(torch.argmax(torch.nn.functional.softmax(outputs, dim=2))) == int(labels[0][0]):
            stats[int(labels[0][0])][0] += 1
            pred_correct += 1
        
        if int(labels[0][0]) in torch.topk(torch.reshape(outputs, (-1,)), k).indices.tolist():
            pred_top_5 += 1

        stats[int(labels[0][0])][1] += 1
        pred_all += 1

    if print_stats:
        stats = {key: value[0] / value[1] for key, value in stats.items() if value[1] != 0}
        print("Label accuracies statistics:")
        print(str(stats) + "\n")
        logging.info("Label accuracies statistics:")
        logging.info(str(stats) + "\n")

    return running_loss/data_length, pred_correct, pred_all, (pred_correct / pred_all), (pred_top_5 / pred_all)


def evaluate_top_k(model, dataloader, device, k=5):

    pred_correct, pred_all = 0, 0

    for i, data in enumerate(dataloader):
        inputs, labels, _ = data
        inputs = inputs.squeeze(0).to(device)
        labels = labels.to(device, dtype=torch.long)

        outputs = model(inputs).expand(1, -1, -1)

        if int(labels[0][0]) in torch.topk(outputs, k).indices.tolist():
            pred_correct += 1

        pred_all += 1

    return pred_correct, pred_all, (pred_correct / pred_all)

import pandas as pd

def evaluate_with_features(model, dataloader, cel_criterion, device, print_stats=False, save_results=False):

    pred_correct, pred_top_5, pred_all = 0, 0, 0
    running_loss = 0.0
    
    stats = {i: [0, 0] for i in range(302)}

    data_length = len(dataloader)

    k = 5 # top 5 (acc)

    # create a list to store the results
    results = []

    for i, data in enumerate(dataloader):
        inputs, labels, video_name, false_seq,percentage_group,max_consec = data
        inputs = inputs.squeeze(0).to(device)
        labels = labels.to(device, dtype=torch.long)
        #print(f"iteration {i} in evaluate, video name {video_name}, max_consec {max_consec[i]}")
        outputs = model(inputs).expand(1, -1, -1)

        loss = cel_criterion(outputs[0], labels[0])
        running_loss += loss

        # Statistics
        if int(torch.argmax(torch.nn.functional.softmax(outputs, dim=2))) == int(labels[0][0]):
            stats[int(labels[0][0])][0] += 1
            pred_correct += 1
        
        if int(labels[0][0]) in torch.topk(torch.reshape(outputs, (-1,)), k).indices.tolist():
            pred_top_5 += 1

        stats[int(labels[0][0])][1] += 1
        pred_all += 1

        # calculate the accuracy per instance
        acc = 1 if int(torch.argmax(torch.nn.functional.softmax(outputs, dim=2))) == int(labels[0][0]) else 0

        # append the results to the list
        results.append({
            'video_name': video_name,
            'in_range_sequences': false_seq[i].numpy()[0],
            'percentage_group': percentage_group[i].numpy()[0],
            'max_percentage': max_consec[i].numpy()[0],
            'accuracy': acc
        })

    if print_stats:
        stats = {key: value[0] / value[1] for key, value in stats.items() if value[1] != 0}
        print("Label accuracies statistics:")
        print(str(stats) + "\n")
        logging.info("Label accuracies statistics:")
        logging.info(str(stats) + "\n")

    # convert the list to a DataFrame
    df_results = pd.DataFrame(results)

    # # save the DataFrame to a CSV file if save_results is True
    # if save_results:
    #     save_path = 'results.csv'
    #     df_results.to_csv(save_path, index=False)

    return running_loss/data_length, pred_correct, pred_all, (pred_correct / pred_all), (pred_top_5 / pred_all), df_results


def generate_csv_result(run, model, dataloader, folder_path, meaning, device):

    model.train(False)
    
    submission = dict()
    trueLabels = dict()
    meaningLabels = dict()

    for i, data in enumerate(dataloader):
        inputs, labels, video_name = data
        inputs = inputs.squeeze(0).to(device)
        labels = labels.to(device, dtype=torch.long)
        outputs = model(inputs).expand(1, -1, -1)

        pred = int(torch.argmax(torch.nn.functional.softmax(outputs, dim=2)))
        trueLab = int(labels[0][0])

        submission[video_name] = pred
        trueLabels[video_name] = trueLab
        meaningLabels[video_name] = meaning[trueLab]

    diccionarios = [submission, trueLabels, meaningLabels]

    # Define the row names
    headers = ['videoName', 'prediction', 'trueLabel', 'class']

    full_path = folder_path+'/submission.csv'

    # create the csv and define the headers
    with open(full_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        # write the acummulated data
        for key in diccionarios[0].keys():
            row = [key[0]]
            for d in diccionarios:
                row.append(d[key])
            writer.writerow(row)
    
    #artifact = wandb.Artifact(f'predicciones_{run.id}.csv', type='dataset')
    #artifact.add_file(full_path)
    #run.log_artifact(artifact)
    wandb.save(full_path)

import os

import torch

from my_dataset import Dataset

def predict(model, batch, device):
        
        x, ys, lengths = batch
        
        x.to(device)
        ys.to(device)
        lengths.to(device)
        
        predictions = model(x, lengths)
        
        return predictions, ys

def train(
        model,
        iterator,
        optimizer,
        criterion,
        print_every=10,
        epoch=0,
        device="cpu",
    ):

        print(f"epoch {epoch}")

        epoch_loss = 0

        model.train()

        for i, batch in enumerate(iterator):
            
            optimizer.zero_grad()

            predictions, ys = predict(model, batch, device)

            loss = criterion(predictions.view(-1, predictions.size(-1)), ys.view(-1))
            loss.backward()

            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss

            if not (i + 1) % print_every:
                print(f"step {i} from {len(iterator)} at epoch {epoch}")
                print(f"Loss: {batch_loss}")

        return epoch_loss / len(iterator)

def evaluate(model, 
             iterator, 
             criterion, 
             epoch=0, 
             device="cpu", 
             save_checkpoints=True, 
             timestamp=None):

    print(f"epoch {epoch} evaluation")

    epoch_loss = 0

    #    model.train(False)
    model.eval()

    with torch.no_grad():
        for batch in iterator:

            predictions, ys = predict(model, batch, device)

            loss = criterion(predictions.view(-1, predictions.size(-1)), ys.view(-1))
            
            epoch_loss += loss.item()

    overall_loss = epoch_loss / len(iterator)

    if save_checkpoints:
        file_name = f'{timestamp}_epoch_{str(epoch)}.pt'
        folder = 'logs/checkpoint/'
        path = os.path.expanduser(folder +  file_name)
        torch.save(model.state_dict(), path)

    print(f"epoch loss {overall_loss}")
    print(
        "========================================================================================================"
    )

    return overall_loss

def generate(model, seed_text, tokenizer, pad_index=0, eos_index=3, max_sequence=512, max_len=25, device='cpu'):
    
    tokenized = tokenizer.encode([seed_text])
        
    model.eval()
    
    with torch.no_grad():

        pred = []

        for step in range(max_sequence):
            
            length = len(tokenized)
            
            inference_dataset = Dataset(data=tokenized, max_len=max_len,
                                       pad_index=pad_index, eos_index=eos_index)            
            inference_loader = torch.utils.data.DataLoader(inference_dataset, batch_size=1)

            prediction, _ = predict(model, next(iter(inference_loader)), device)
            next_token_prediction = prediction[:, length-1]
            pred_token_id = next_token_prediction.argmax(dim=1).item()

            pred.append(pred_token_id)
            tokenized[0].append(pred_token_id)
            if len(tokenized) > max_len:
                tokenized = tokenized[-max_len:]

            if pred_token_id == eos_index:
                break
    
    predicted_text = tokenizer.decode(pred)[0]
    predicted_text += ' <EOS>'    
    return predicted_text[:predicted_text.index('<EOS>')]
    
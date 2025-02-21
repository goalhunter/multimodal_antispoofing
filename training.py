import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from multimodal import MultiModalDetector
from file_reader import load_data_files

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiModalDetector()
model.to(device)

train_dataset, test_dataset, train_loader, test_loader = load_data_files()

criterion = nn.CrossEntropyLoss()

num_epochs = 100

total_steps = len(train_loader) * num_epochs

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-5
)

scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-4,
    total_steps=total_steps,
    pct_start=0.1,  # Use first 10% of training for warmup
    div_factor=25,  # Start with lr/25
    final_div_factor=1e4  # End with lr/10000
)

FINE_TUNED_MODEL_PATH = 'Wav2vec.pt'

def evaluate(model, dataloader, device):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    running_loss = 0.0
    
    # No gradient computation needed
    with torch.no_grad():
        for audio, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            audio, labels = audio.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(audio)
            
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * audio.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

best_loss = 1  # Initialize best accuracy

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar
    loop = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]', leave=False)

    for batch_idx, (audio, labels) in enumerate(loop):
        # Move data to device
        audio, labels = audio.to(device), labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(audio)
        
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()

        optimizer.step()
        scheduler.step()
        
        # Update running loss
        running_loss += loss.item() * audio.size(0)
        
        # Compute accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        loop.set_postfix(loss=loss.item(), accuracy=100. * correct / total)
        
    
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = 100. * correct / total
    
    print(f'Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%')
    
    # Optional: Evaluate on test set after each epoch
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f'Validation - Loss: {test_loss:.4f} - Accuracy: {test_acc:.2f}%')
    
    # Optional: Save the best model
    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), FINE_TUNED_MODEL_PATH)
        print(f'Best model saved at epoch {epoch+1} with accuracy {epoch_acc:.2f}%')
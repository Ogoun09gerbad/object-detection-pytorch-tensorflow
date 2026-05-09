import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, test_dataloader,
                 lr, wd, epochs, device):
        self.epochs           = epochs
        self.model            = model
        self.train_dataloader = train_dataloader
        self.val_dataloader   = val_dataloader   
        self.test_dataloader  = test_dataloader
        self.device           = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)
        self.criterion = nn.CrossEntropyLoss()
        self.patience   = 10
        self.no_improve = 0
        self.best_acc   = 0

    def train(self, save=True, plot=False):
        self.train_acc   = []
        self.train_loss  = []
        self.val_accs    = []
        self.val_losses  = [] 

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            total_correct = 0
            total_samples = 0
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=False)

            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss    = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                _, preds = outputs.max(1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
                total_loss    += loss.item() * labels.size(0)
                avg_acc  = 100.0 * total_correct / total_samples
                avg_loss = total_loss / total_samples
                progress_bar.set_postfix({'Acc': f'{avg_acc:.2f}%', 'Loss': f'{avg_loss:.4f}'})

            self.scheduler.step()
            self.train_acc.append(avg_acc)
            self.train_loss.append(avg_loss)

            # VALIDATION
            val_acc, val_loss = self.evaluate(self.val_dataloader)
            self.val_accs.append(val_acc)
            self.val_losses.append(val_loss)

            print(f"\nEpoch {epoch+1}/{self.epochs}")
            print(f"Train Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.2f}%")
            print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.no_improve = 0
                if save:
                    torch.save(self.model.state_dict(), "geraud_model.pth")
                    print(f" Best model saved (val_acc={val_acc:.2f}%)")
            else:
                self.no_improve += 1
                print(f"No improvement ({self.no_improve}/{self.patience})")
                if self.no_improve >= self.patience:
                    print(" Early stopping triggered")
                    break

        if plot:
            self.plot_training_history()

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss, total_correct, total_samples = 0, 0, 0
        for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            loss    = self.criterion(outputs, labels)
            _, preds = outputs.max(1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            total_loss    += loss.item() * labels.size(0)
        return 100.0 * total_correct / total_samples, total_loss / total_samples

    def test(self):
        print("\n Final Test Evaluation:")
        return self.evaluate(self.test_dataloader)

    def plot_training_history(self):
        epochs = range(1, len(self.train_loss) + 1)
        plt.figure(figsize=(12, 5))
        
        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_acc, label="Train Accuracy")
        plt.plot(epochs, self.val_accs, label="Validation Accuracy")
        plt.title("Accuracy")
        plt.legend()
        
        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_loss, label="Train Loss")
        plt.plot(epochs, self.val_losses, label="Validation Loss")
        plt.title("Loss")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("training_history.png") 
        plt.show()

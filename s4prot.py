import pandas as pd
import numpy as np
from sklearn import model_selection
import torch.nn as nn
from models.s4.s4 import S4Block as S4  # Can use full version instead of minimal S4D standalone below
from models.s4.s4d import S4D
import torch.optim as optim

df=pd.read_csv('SVMdata.csv')
y=df['Labels']
X=df['2']

def alphabet_to_number(alphabet):
    alphabet = alphabet.upper()  # Convert alphabet to uppercase
    return ord(alphabet) - 64  # Subtract 64 to get the numerical value

# Convert alphabets in a sequence to numbers
def sequence_to_numbers(sequence):
    numbers = []
    for alphabet in sequence:
        number = alphabet_to_number(alphabet)
        numbers.append(number)
    return numbers

seq=[]
Sequences=X.values.tolist()

for i in range(len(Sequences)):
    nseq=sequence_to_numbers(Sequences[i])
    #print(nseq)
    seq.append(nseq)
    #print("Hi")
seq=pd.DataFrame(seq)

seq= seq.replace('', np.nan) # Replace empty strings with NaN


seq = seq.fillna(0) # Fill NaN values with zeroes

''' Count the number of elements in each row
row_element_counts = seq.apply(lambda row: len(row), axis=1)
print(row_element_counts.max()) #5537 is longest sequence
seq.to_csv('Seqtest.csv')'''


X=seq
X_train, X_test, y_train, test = model_selection.train_test_split(X, y,
                                    train_size=0.80, test_size=0.20, random_state=4)

X_test, X_val, y_test, y_val = model_selection.train_test_split(X_test, test, train_size=0.50, test_size=0.50, random_state=4)

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(y_train)     # Fit the encoder on the training labels and transform both training and test labels
y_test = label_encoder.transform(y_test)
y_val = label_encoder.transform(y_val)

X_train=torch.tensor(X_train.values,  dtype=torch.float32)
X_train=X_train.unsqueeze(-1)

X_test=torch.tensor(X_test.values,  dtype=torch.float32)
X_test=X_test.unsqueeze(-1)

X_val=torch.tensor(X_val.values,  dtype=torch.float32)
X_val=X_val.unsqueeze(-1)

y_train=torch.tensor(y_train,  dtype=torch.float32)
y_test=torch.tensor(y_test,  dtype=torch.float32)
y_val=torch.tensor(y_val,  dtype=torch.float32)

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

# Create an instance of your custom dataset
trainset = CustomDataset(X_train, y_train)
valset = CustomDataset(X_val, y_val)

bts=1
n_classes=5
d_input=1
d_output = 1

# Dataloaders
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=bts, shuffle=True, num_workers=5)
valloader = torch.utils.data.DataLoader(valset, batch_size=bts, shuffle=False, num_workers=5)
#testloader = torch.utils.data.DataLoader(testset, batch_size=bts, shuffle=False, num_workers=5)

class S4Model(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        n_layers=10,
        dropout=0.2,
        prenorm=False,
    ):
        super().__init__()

        self.prenorm = prenorm
        print(d_input, 'pause', d_model)

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, 0.01))
            )
            self.norms.append(nn.LayerNorm(d_model))
            #self.dropouts.append(dropout_fn(dropout))
        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)
    def forward(self, x):
            """
            Input x is shape (B, L, d_input)
            """
            x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

            x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
            for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
                # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

                z = x
                if self.prenorm:
                    # Prenorm
                    z = norm(z.transpose(-1, -2)).transpose(-1, -2)

                # Apply S4 block: we ignore the state input and output
                z, _ = layer(z)

                # Dropout on the output of the S4 block
                z = dropout(z)

                # Residual connection
                x = z + x

                if not self.prenorm:
                    # Postnorm
                    x = norm(x.transpose(-1, -2)).transpose(-1, -2)

            x = x.transpose(-1, -2)

            # Pooling: average pooling over the sequence length
            x = x.mean(dim=1)

            # Decode the outputs
            x = self.decoder(x)  # (B, d_model) -> (B, d_output)

            return x
    
# Model
print('==> Building model..')
model = S4Model(
    d_input=d_input,
    d_output=n_classes,
    d_model=10,
    n_layers=10,
    #dropout=args.dropout,
    #prenorm=args.prenorm,
)


def setup_optimizer(model, lr, weight_decay, epochs):
    """
    S4 requires a specific optimizer setup.
    The S4 layer (A, B, C, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.
    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    # Create a lr scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler

criterion = use nn.BCEWithLogitsLoss()
accuracy=[]
optimizer, scheduler = setup_optimizer(
    model, lr=0.001, weight_decay=0.001, epochs=10
)
for epoch in range(10):
    model.train()
    for inputs, targets in trainloader:
            #Perform forward pass
            targets= targets.long()
            onelabels = F.one_hot(targets, num_classes=500)
            onelabels=onelabels.float()
            output=model(inputs)
            loss = criterion((output), onelabels)
            optimizer.zero_grad()

            #Perform backward pass
            loss.backward()
            optimizer.step()
            #print('training')

    model.eval()
    from torcheval.metrics import MulticlassAccuracy
    import torch.nn.functional as F

    for inputs, targets in valloader:
            #Perform forward pass
            targets= targets.long()            
            onelabels = F.one_hot(targets, num_classes=5)
            onelabels=onelabels.float()
            #print(onelabels)
            onelabels = onelabels.squeeze()

            output=model(inputs)

            predicted_labels= torch.round(torch.sigmoid(output)) #sigmoid produces probabilities that are rounded to 0 or 1
            predicted_labels = predicted_labels.squeeze()
            metric= MulticlassAccuracy()
            metric.update(predicted_labels, onelabels)
            a=metric.compute()
    accuracy.append(a)
    print(a)


torch.save(model, 's4.pth')

import matplotlib.pyplot as plt
plt.plot(accuracy)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

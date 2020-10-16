import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from typing import Callable


from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()  # datasets are stored in a dictionary containing an array of features and targets
iris.keys()
preprocessed_features = (iris['data'] - iris['data'].mean(axis=0)) / iris['data'].std(axis=0)
labels = iris['target']
# train_test_split takes care of the shuffling and splitting process
train_features, test_features, train_labels, test_labels = train_test_split(preprocessed_features, labels, test_size=1/3)
features = {
    'train': torch.tensor(train_features, dtype=torch.float32),
    'test': torch.tensor(test_features, dtype=torch.float32),
}
labels = {
    'train': torch.tensor(train_labels, dtype=torch.long),
    'test': torch.tensor(test_labels, dtype=torch.long),
}

class MLP(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_layer_size: int,
                 output_size: int,
                 activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.relu):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_layer_size)
        self.l2 = nn.Linear(hidden_layer_size, output_size)
        self.activation_fn = activation_fn
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.l1(inputs)
        x = relu(x)
        # x = self.activation_fn(x)
        x = self.l2(x)
        return x

def accuracy(probs: torch.FloatTensor, targets: torch.LongTensor) -> float:
    """
    Args:
        probs: A float32 tensor of shape ``(batch_size, class_count)`` where each value 
            at index ``i`` in a row represents the score of class ``i``.
        targets: A long tensor of shape ``(batch_size,)`` containing the batch examples'
            labels.
    """
    predicted = torch.argmax(probs,dim=1)
    count = torch.eq(predicted,targets).sum()
    return count.item()/targets.shape[0]

# def check_accuracy(probs: torch.FloatTensor,
#                    labels: torch.LongTensor,
#                    expected_accuracy: float):
#     actual_accuracy = float(accuracy(probs, labels))
#     assert actual_accuracy == expected_accuracy, f"Expected accuracy to be {expected_accuracy} but was {actual_accuracy}"

# check_accuracy(torch.tensor([[0, 1],
#                             [0, 1],
#                             [0, 1],
#                             [0, 1],
#                             [0, 1]]),
#             torch.ones(5, dtype=torch.long),
#             1.0)
# check_accuracy(torch.tensor([[1, 0],
#                             [0, 1],
#                             [0, 1],
#                             [0, 1],
#                             [0, 1]]),
#             torch.ones(5, dtype=torch.long),
#             0.8)
# check_accuracy(torch.tensor([[1, 0],
#                             [1, 0],
#                             [0, 1],
#                             [0, 1],
#                             [0, 1]]),
#             torch.ones(5, dtype=torch.long),
#             0.6)
# check_accuracy(torch.tensor([[1, 0],
#                             [1, 0],
#                             [1, 0],
#                             [1, 0],
#                             [1, 0]]),
#             torch.ones(5, dtype=torch.long),
#             0.0)
# print("All test cases passed")

def relu(inputs: torch.Tensor) -> torch.Tensor:
    # We take a copy of the input as otherwise we'll be modifying it in
    # place which makes it harder to debug.
    outputs = inputs.clone()  
    outputs[inputs < 0] = 0
    return outputs

def softmax(logits: torch.Tensor) -> torch.Tensor:
    x_exp = torch.exp(logits)
    x_exp_sum = torch.sum(torch.exp(logits),dim=1,keepdim=True)
    return x_exp/x_exp_sum


def deep_learn():
    feature_count = 4
    hidden_layer_size = 100
    class_count = 3

    device = torch.device('cuda')
    summary_writer = SummaryWriter('logs', flush_secs=5)
    model = MLP(feature_count, hidden_layer_size, class_count)
    model = model.to(device)
    features["train"] = features["train"].to(device)
    features["test"] = features["test"].to(device)
    labels["train"] = labels["train"].to(device)
    labels["test"] = labels["test"].to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    CEL = nn.CrossEntropyLoss()
    criterion = lambda logits, ys: CEL(softmax(logits), ys)

    for epoch in range(1000):

        logits = model.forward(features['train'])
        # logits.shape is (100,3)
        loss = criterion(logits,  labels['train'])
        print("epoch: {} train accuracy: {:2.2f}, loss: {:5.5f}".format(
            epoch,
            accuracy(logits, labels['train']) * 100,
            loss.item()
        ))
        train_accuracy = accuracy(logits, labels['train']) * 100
        summary_writer.add_scalar('accuracy/train', train_accuracy, epoch)
        summary_writer.add_scalar('loss/train', loss.item(), epoch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


    # Finally we can test our model on the test set and get an unbiased estimate of its performance.    
    summary_writer.close()
    logits = model.forward(features['test'])    
    test_accuracy = accuracy(logits, labels['test']) * 100
    print("test accuracy: {:2.2f}".format(test_accuracy))

deep_learn()
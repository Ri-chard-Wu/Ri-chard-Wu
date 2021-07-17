import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        """
        e.g.:

        state:  tensor([0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1.])

        next_state:  tensor([0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1.])

        action:  tensor([0, 1, 0])

        reward:  tensor(0.)

        """
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

##        print("\n\nstate: ", state)
##        print("\nnext_state: ", next_state)
##        print("\naction: ", action)
##        print("\nreward: ", reward)

        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)
##        print("pred: ",pred)
        """
        pred:  tensor([ [-0.0244,  0.0611, -0.0317],
                        [-0.0002,  0.0428,  0.0645],
                        [-0.0244,  0.0611, -0.0317],
                        [ 0.0086,  0.0607,  0.0073],
                        [-0.0289,  0.0701, -0.0988],
                        [-0.0002,  0.0428,  0.0645],
                        [-0.0244,  0.0611, -0.0317],
                        [ 0.0086,  0.0607,  0.0073],], grad_fn=<AddmmBackward>)
        
        Which means, for example:
        
        Q(state[0], go_straigt) == -0.0244
        Q(state[0], go_left) == 0.0611
        Q(state[0], go_right) == -0.0317
        """

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:# terminal state
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()





































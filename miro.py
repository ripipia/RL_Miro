from Gridworld import Gridworld

game = Gridworld(size=4, mode='static')

game.display()

import numpy as np

import torch

from Gridworld import Gridworld

import random

from matplotlib import pylab as plt

print(torch.__version__) # 파이 토치 버전 확인
USE_CUDA = torch.cuda.is_available()
print(USE_CUDA) # GPU 사용 확인 True가 출력 될 시 GPU가 사용가능
device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print(device) # GPU가 사용가능할 경우 cuda:0가 출력되며 사용 불가능할 경우 cpu가 출력됨



l1 = 64  # 4X4X4

l2 = 150

l3 = 100

l4 = 4  # action

model = torch.nn.Sequential(

    torch.nn.Linear(l1, l2),

    torch.nn.ReLU(),

    torch.nn.Linear(l2, l3),

    torch.nn.ReLU(),

    torch.nn.Linear(l3, l4)

)

loss_fn = torch.nn.MSELoss()

learning_rate = 1e-3

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

gamma = 0.9

epsilon = 1.0

action_set = {

    0: 'u',

    1: 'd',

    2: 'l',

    3: 'r',

}

epochs = 100

losses = []  # A

for i in range(epochs):  # B

    game = Gridworld(size=4, mode='static')  # C

    state_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0  # D

    state1 = torch.from_numpy(state_).float()  # E

    status = 1  # F

    while (status == 1):  # G

        qval = model(state1)  # H

        qval_ = qval.data.numpy()

        if (random.random() < epsilon):  # I

            action_ = np.random.randint(0, 4)

        else:

            action_ = np.argmax(qval_)

        action = action_set[action_]  # J

        game.makeMove(action)  # K

        state2_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0

        state2 = torch.from_numpy(state2_).float()  # L

        reward = game.reward()

        with torch.no_grad():

            newQ = model(state2.reshape(1, 64))

        maxQ = torch.max(newQ)  # M

        if reward == -1:  # N

            Y = reward + (gamma * maxQ)

        else:

            Y = reward

        Y = torch.Tensor([Y]).detach()

        X = qval.squeeze()[action_]  # O

        loss = loss_fn(X, Y)  # P

        print(i, loss.item())

        # clear_output(wait=True)

        optimizer.zero_grad()

        loss.backward()

        losses.append(loss.item())

        optimizer.step()

        state1 = state2

        if reward != -1:  # Q

            status = 0

    if epsilon > 0.1:  # R

        epsilon -= (1 / epochs)


def test_model(model, mode='static', display=True):
    i = 0

    test_game = Gridworld(mode=mode)

    state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0

    state = torch.from_numpy(state_).float()

    if display:
        print("Initial State:")

        print(test_game.display())

    status = 1

    while (status == 1):  # A

        qval = model(state)

        qval_ = qval.data.numpy()

        action_ = np.argmax(qval_)  # B

        action = action_set[action_]

        if display:
            print('Move #: %s; Taking action: %s' % (i, action))

        test_game.makeMove(action)

        state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0

        state = torch.from_numpy(state_).float()

        if display:
            print(test_game.display())

        reward = test_game.reward()

        if reward != -1:

            if reward > 0:

                status = 2

                if display:
                    print("Game won! Reward: %s" % (reward,))

            else:

                status = 0

                if display:
                    print("Game LOST. Reward: %s" % (reward,))

        i += 1

        if (i > 15):

            if display:
                print("Game lost; too many moves.")

            break

    win = True if status == 2 else False

    return win


test_model(model)

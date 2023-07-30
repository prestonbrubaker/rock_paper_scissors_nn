import random

# Variables
iteration_count = 0
player_choices = [[1, 0, 0]]
computer_choices = [[0, 1, 0]]
correct_choices = [[0, 1, 0]]
fitness = 0
debug_mode = False
global computer_wins
global player_wins
computer_wins = 0
player_wins = 0


nodes = [

    1, 1, 1, 1,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0

]

weights = [
    0.7636036909135714, -1.0516711072721012, -0.05650303890134151, 0.17907383497437696, -0.4900763406419367,
     0.7759543914527869, -0.5580978643583527, -1.9870741375706877, 1.9732432782576614, 1.4498755884097076,
     -1.9750793545223695, 1.993855863153414, -2, 2, -1.9999999999999636, 1.6638235331351285, -2, -0.8583422369005256,
     1.8752735357651313, 1.999979139720976, 2, 1.9875978693327034, -1.999999999999916, -1.9976625116833457,
     -1.9999089166920752, 1.9874091058752683, 1.9999999948895626, -1.9746572966409381, 1.8499325086601865,
     -1.985806873227798, -2, 1.7485092253832561, -1.1165923834552711, -2, 2, 1.924750992280907, -1.9985647456021374,
     1.8914995533578016, -1.2289548655122635, -0.7153989591394668, -0.4758961436300868, 1.9999999999999478, -2,
     0.985705682440192, 1.999999999999793, -1.8206693606779356, -1.6820858931390767, -1.2965320328349517

]

def initialize_weights(weights):
    for i in range(len(weights)):
        weights[i] = random.uniform(-1,1)

    print("\nInitial value of weights: ")
    print(weights)

def reset_nodes(nodes):
    for i in range(len(nodes)):
        nodes[i] = 0


def sigmoid(number):
    sigmoid_num = 1 / ( 1 + 2.711 ** ( -1 * number ))
    return sigmoid_num

#initialize_weights(weights)

def forward_propogation(nodes, weights, inputs):
    reset_nodes(nodes)
    nodes[0] = inputs[0]
    nodes[1] = inputs[1]
    nodes[2] = inputs[2]
    nodes[3] = inputs[3]

    for i in list(range(4,16)):
        if(debug_mode == True):
            print("\nNode: ")
            print(i)

        node_layer = int((i - i % 4) / 4)
        if (debug_mode == True):
            print("   Node Layer: ")
            print(node_layer)

        # Sum weights from previous layer
        for j in list(range(0,4)):
            node_accessed = (node_layer - 1) * 4 + j
            weight_used = node_accessed * 4 + i % 4
            if (debug_mode == True):
                print("Node " + str(i) + " accessed " + "node " + str(node_accessed) + " and used weight " + str(weight_used))
            nodes[i] += weights[weight_used] * nodes[node_accessed]
        nodes[i] = sigmoid(nodes[i])
    if (debug_mode == True):
        print("\nFinal state of the nodes:")
        print(nodes)
    outputs = [nodes[12], nodes[13], nodes[14], nodes[15]]
    return outputs

def one_hot_encode(one_hot):
    j = one_hot.index(max((one_hot)))
    for i in range(len(one_hot)):
        one_hot[i] = 0

    one_hot[j] = 1
    return one_hot


def battle(player, computer):
    if(len(player) == 4):
        player.pop(3)
    player_choices.append(player)
    computer_choices.append(computer)
    if(player == [1, 0, 0]):
        correct_choices.append([0, 1, 0])
    elif(player == [0, 1, 0]):
        correct_choices.append([0, 0, 1])
    elif(player == [0, 0, 1]):
        correct_choices.append([1, 0, 0])

    if(player == computer):
        print("\nTie!")
        score = 0
    elif((player == [1, 0, 0] and computer == [0, 0, 1]) or (player == [0, 1, 0] and computer == [1, 0, 0]) or (player == [0, 0, 1] and computer == [0, 1, 0])):
        print("\nPlayer wins!")
        score = -1
    elif ((player == [1, 0, 0] and computer == [0, 1, 0]) or (player == [0, 1, 0] and computer == [0, 0, 1]) or (player == [0, 0, 1] and computer == [1, 0, 0])):
        print("\nComputer wins!")
        score = 1
    return score

def test(nodes, weights):
    fitness = 0
    decay_rate = .97     # Rate of drop off of old data
    # evaluate fitness
    for i in range(len(player_choices)):
        #inputs = player_choices[i].copy()
        inputs = player_choices[i - 1].copy()
        inputs.append(i % 6 / 6)
        outputs = forward_propogation(nodes, weights, inputs)
        fitness += ((outputs[0] - correct_choices[i][0]) ** 2 )
        fitness += ((outputs[1] - correct_choices[i][1]) ** 2 )
        fitness += ((outputs[2] - correct_choices[i][2]) ** 2 )
    fitness = fitness / len(player_choices)
    return fitness

def train(nodes, weights):
    mut_frac_i = .1
    annealing_co = 0.0001
    fitness_mother = test(nodes, weights)
    print("Fitness of Mother (lower is better): " + str(fitness_mother))

    for k in range(10000):
        mut_frac = mut_frac_i * 2.711 ** ( -1 * annealing_co * k)
        nodes_c = nodes.copy()
        weights_c = weights.copy()
        for j in range(len(weights_c)):
            weights_c[j] += random.uniform(-1 * mut_frac, mut_frac)
            if(weights_c[j] > 2):
                weights_c[j] = 2
            elif(weights_c[j] < -2):
                weights_c[j] = -2
        fitness_off = test(nodes_c, weights_c)
        if(debug_mode == True):
            print("Fitness of Offspring: " + str(fitness_off))
        if(fitness_off < fitness_mother):
            nodes = nodes_c
            weights = weights_c
            fitness_mother = fitness_off
            if(debug_mode == True):
                print("Offspring " + str(k) + " has replaced Mother!")
    return [nodes, weights]





# Initialize
# initialize_weights(weights)

# Main Loop
inputs = [0, 1, 0, 0]
outputs = forward_propogation(nodes, weights, inputs)
print("\nOutput of neural network: ")
print(outputs)

while True:
    iteration_count += 1
    reset_nodes(nodes)
    print("\nIteration count: " + str(iteration_count))

    player_choice_s = input("Rock (r), Paper(p), or Scissors(s)?: ")
    if(player_choice_s == "r"):
        player_choice_a = [1, 0, 0]
    elif(player_choice_s == "p"):
        player_choice_a = [0, 1, 0]
    elif(player_choice_s == "s"):
        player_choice_a = [0, 0, 1]

    timer = iteration_count % 6 / 6

    #inputs = player_choice_a   mode of getting player's current hand
    inputs = player_choices[iteration_count - 1].copy()

    inputs.append(timer)

    outputs = forward_propogation(nodes, weights, inputs)   # Retrieve guess from neural network in confidence such as [.9343, .244, .8244]
    outputs.pop(3)

    if(len(player_choice_a) == 4):
        player_choice_a.pop(3)

    print("\nConfidence of computer: ")
    print(outputs)

    one_hot = one_hot_encode(outputs)   #obtain actual guess from computer in form such as [1, 0, 0]

    print(one_hot)

    if(one_hot == [1, 0, 0]):
        print("\nComputer chooses rock!")
    elif (one_hot == [0, 1, 0]):
        print("\nComputer chooses paper!")
    elif (one_hot == [0, 0, 1]):
        print("\nComputer chooses scissors!")

    score = battle(player_choice_a, one_hot)
    if(score == 1):
        computer_wins += 1
    elif(score == -1):
        player_wins += 1

    if(debug_mode == True):
        print("\nSummary of choices so Far:")
        print("Player: " + str(player_choices))
        print("Computer: " + str(computer_choices))
        print("Correct choices for computer: " + str(correct_choices))
    if(computer_wins + player_wins > 0):
        print("\nPercent of games won by computer: " + str( round(computer_wins / (computer_wins + player_wins) * 100, 2) ) + "%")

    if(iteration_count % 20 == 0):
        print(weights)

    new_M = train(nodes, weights)
    nodes = new_M[0]
    weights = new_M[1]


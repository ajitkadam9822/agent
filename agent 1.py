import pygame
import constants as c
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from torch.utils.tensorboard import SummaryWriter

# Global Variables Initialization
WIDTH, HEIGHT = c.WIDTH, c.HEIGHT
WHITE = (255, 255, 255)
RED = (255, 0, 0)
RESOURCE_REGEN_INTERVAL = c.RES_REGEN_INTERVAL

# Initialize essential components
pygame.init()
writer = SummaryWriter()
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Resource Collection Agent")
tree_image = pygame.image.load('assets/tree.png')
tree_image = pygame.transform.scale(tree_image, (20, 20))

def display_message(message):
    """
    For displaying messages (for future uses).
    """
    print(message)

def initialize_neural_network():
    """
    Initialize the Neural Network model for the agent.
    """
    return NeuralNetwork()

def random_position(width, height):
    """
    Returns a random position within the given width and height.
    """
    x = random.randint(0, width)
    y = random.randint(0, height)
    return [x, y]

def calculate_distance(point1, point2):
    """
    Calculates the Euclidean distance between two points.
    """
    delta_x = point1[0]-point2[0]
    delta_y = point1[1]-point2[1]
    return math.sqrt(delta_x**2 + delta_y**2)

def check_agent_eats_resource(agent_position, resource_position):
    """
    Check if the agent has collected the resource.
    """
    dist = calculate_distance(agent_position, resource_position)
    return dist < 15

def save_neural_network_model(model, filename="agent_model.pth"):
    """
    Save the trained model.
    """
    display_message("Saving model...")
    torch.save(model.state_dict(), filename)
    display_message("Model saved successfully!")

def load_neural_network_model(model, filename="agent_model.pth"):
    """
    Load a pre-trained model.
    """
    display_message("Loading model...")
    model.load_state_dict(torch.load(filename))
    model.eval()
    display_message("Model loaded successfully!")

class NeuralNetwork(nn.Module):
    """
    Neural network class defining the architecture.
    """
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.initialize_layers()

    def initialize_layers(self):
        self.fc = nn.Sequential(
            nn.Linear(4, 128),  # Input layer
            nn.ReLU(),
            nn.Linear(128, 64),  # Hidden layer 1
            nn.ReLU(),
            nn.Linear(64, 32),   # Hidden layer 2
            nn.ReLU(),
            nn.Linear(32, 2)     # Output layer
        )

    def forward(self, x):
        output = self.fc(x)
        return output

def regenerate_resource_position(resources):
    """
    Adds a new resource position based on the regeneration interval.
    """
    if len(resources) > 0 and random.randint(0, RESOURCE_REGEN_INTERVAL) == 0:
        resources.append(random_position(WIDTH, HEIGHT))

def move_agent_based_on_prediction(agent_position, predicted_movement):
    """
    Updates the agent position based on the neural network prediction.
    """
    delta_x = int(predicted_movement[0].item() * WIDTH)
    delta_y = int(predicted_movement[1].item() * HEIGHT)
    agent_position[0] += delta_x
    agent_position[1] += delta_y
    return agent_position

def display_agent_and_resource(agent_position, resources):
    """
    Draws the agent and resource on the screen.
    """
    win.fill(WHITE)
    pygame.draw.circle(win, RED, (agent_position[0], agent_position[1]), 10)
    for resource in resources:
        win.blit(tree_image, (resource[0] - 10, resource[1] - 10))
    pygame.display.flip()

def main_simulation():
    """
    Main simulation function.
    """
    running = True
    resources = [random_position(WIDTH, HEIGHT) for _ in range(10)]
    agent_position = [WIDTH // 2, HEIGHT // 2]
    agent_neural_net = initialize_neural_network()
    optimizer = optim.Adam(agent_neural_net.parameters())
    loss_function = nn.MSELoss()

    exploration_rate = 0.1
    EXPLORATION_DECAY = 0.0995

    load_model_decision = input("Do you want to load a pre-trained model? (yes/no): ")
    if load_model_decision.lower() == "yes":
        load_neural_network_model(agent_neural_net)

    for episode_number in range(100):
        total_loss_for_episode = 0.0
        current_score = 0
        action_count = 0
        while running and action_count < 100:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if random.random() < exploration_rate:
                chosen_resource = random.choice(resources)
            else:
                chosen_resource = min(resources, key=lambda resource: calculate_distance(agent_position, resource))

            target_tensor = torch.FloatTensor([(chosen_resource[0]-agent_position[0])/WIDTH, (chosen_resource[1]-agent_position[1])/HEIGHT])
            input_tensor = torch.FloatTensor([agent_position[0]/WIDTH, agent_position[1]/HEIGHT, chosen_resource[0]/WIDTH, chosen_resource[1]/HEIGHT])
            
            predicted_movement = agent_neural_net(input_tensor)
            loss = loss_function(predicted_movement, target_tensor)

            total_loss_for_episode += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            agent_position = move_agent_based_on_prediction(agent_position, predicted_movement)
            if(action_count!=0):
             average_loss = total_loss_for_episode / action_count   # Calculate average loss
            else:
                average_loss=0
            writer.add_scalar("Average Loss per Episode", average_loss, episode_number)
            writer.add_scalar("Loss", loss, action_count)
            writer.add_scalar("Score", current_score, action_count)

            if check_agent_eats_resource(agent_position, chosen_resource):
                resources.remove(chosen_resource)
                current_score += 1

            regenerate_resource_position(resources)
            display_agent_and_resource(agent_position, resources)

            if not resources:
                break
            action_count += 1

            exploration_rate *= EXPLORATION_DECAY
            pygame.time.delay(10)
           
        efficiency = current_score / action_count
        print(f"Episode {episode_number + 1}, Score: {current_score}, Actions: {action_count}, Efficiency: {current_score / action_count}")
        resources = [random_position(WIDTH, HEIGHT) for _ in range(10)]
        writer.add_scalar("Actions per Episode", action_count, episode_number) 
        writer.add_scalar("Efficiency", efficiency, episode_number)
        writer.add_scalar("Total Score per Episode", current_score, episode_number) # <- This line logs the total score for each episode

    save_neural_network_model(agent_neural_net)
    writer.close()
    pygame.quit()

if __name__ == "__main__":
    main_simulation()

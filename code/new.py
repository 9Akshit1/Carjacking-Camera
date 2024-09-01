import pygame
import sys
import os
import random

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 800  # Screen dimensions
ROWS, COLS = 20, 20  # Number of rows and columns
NODE_RADIUS = 5  # Radius of each node circle
GAP = 40  # Gap between nodes (20px node, 20px gap)
START_POS = ('J', 10)  # Example starting position (row, column)
DIRECTIONS = ['right', 'down', 'down', 'right', 'up']  # Example path

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
BLUE = (0, 0, 255)

# Create the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simple Map Graph")

# Font for labels
font = pygame.font.SysFont(None, 24)
title_font = pygame.font.SysFont(None, 48)

# Load images
image_paths = ["house.png", "building1.png", "building2.png", "park.png", "shop.png"]
images = [pygame.image.load(os.path.join('data', img)) for img in image_paths]

# Resize images to fit within a grid square
image_size = GAP - 10
images = [pygame.transform.scale(img, (image_size, image_size)) for img in images]

car_image = pygame.image.load(os.path.join('data', 'car.png'))
car_image = pygame.transform.scale(car_image, (GAP - 10, GAP - 10))

# Function to convert (letter, number) to (x, y) grid coordinates
def get_grid_position(pos):
    row = ord(pos[0]) - 65  # Convert letter to index (A=0, B=1, ...)
    col = pos[1] - 1  # Convert number to index (1=0, 2=1, ...)
    x = col * GAP + 50
    y = row * GAP + 50
    return x, y

# Function to draw the grid
def draw_grid(traveled_path, traveled_lines, current_pos, highlight_next=False, next_positions=[]):
    screen.fill(WHITE)
    
    # Draw title
    title = title_font.render("Simple Map Graph", True, BLACK)
    screen.blit(title, (WIDTH//2 - title.get_width()//2, 10))
    
    # Draw the alphabet (A to T) on the left side
    for i in range(ROWS):
        label = font.render(chr(65 + i), True, BLACK)
        screen.blit(label, (10, i * GAP + 50))
    
    # Draw the numbers (1 to 20) on the top
    for j in range(COLS):
        label = font.render(str(j + 1), True, BLACK)
        screen.blit(label, (j * GAP + 50, 10))
    
    # Draw the grid squares with images and connections
    for i in range(ROWS):
        for j in range(COLS):
            x = j * GAP + 50
            y = i * GAP + 50
            pos = (65 + i, j + 1)
            
            # Draw green grass squares
            pygame.draw.rect(screen, GREEN, (x - NODE_RADIUS, y - NODE_RADIUS, GAP, GAP))
            
            # Place images in the grid squares
            screen.blit(images[random.randint(0, 4)], (x - image_size//2, y - image_size//2))
            
            # Determine the color of the node
            if (chr(pos[0]), pos[1]) in traveled_path:
                color = BLUE
            elif (chr(pos[0]), pos[1]) == current_pos:
                color = RED
            else:
                color = BLACK
            
            # Draw the node
            pygame.draw.circle(screen, color, (x, y), NODE_RADIUS)
            
            # Draw connections (up, down, left, right) and highlight traveled lines
            if j < COLS - 1:  # Right connection possible
                right_pos = (chr(pos[0]), pos[1] + 1)
                if ((chr(pos[0]), pos[1]), right_pos) in traveled_lines:
                    pygame.draw.line(screen, BLUE, (x + NODE_RADIUS, y), (x + GAP - NODE_RADIUS, y), 2)
                else:
                    pygame.draw.line(screen, BLACK, (x + NODE_RADIUS, y), (x + GAP - NODE_RADIUS, y), 1)
            
            if i < ROWS - 1:  # Down connection
                down_pos = (chr(pos[0] + 1), pos[1])
                if ((chr(pos[0]), pos[1]), down_pos) in traveled_lines:
                    pygame.draw.line(screen, BLUE, (x, y + NODE_RADIUS), (x, y + GAP - NODE_RADIUS), 2)
                else:
                    pygame.draw.line(screen, BLACK, (x, y + NODE_RADIUS), (x, y + GAP - NODE_RADIUS), 1)
    
    # Draw the car at the current position
    car_x, car_y = get_grid_position(current_pos)
    screen.blit(car_image, (car_x - image_size//2, car_y - image_size//2))

# Main function to run the game loop
def main():
    traveled_path = set()
    traveled_lines = set()
    current_pos = START_POS
    square_size = NODE_RADIUS * 2

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Move the car smoothly (for simplicity, just moving to the next position in this example)
        if DIRECTIONS:
            draw_grid(traveled_path, traveled_lines, current_pos)
            pygame.display.flip()
            pygame.time.wait(500)
            direction = DIRECTIONS.pop(0)
            prev_pos = current_pos

            if direction == 'up' and ord(current_pos[0]) > 65:
                current_pos = (chr(ord(current_pos[0]) - 1), current_pos[1])
            elif direction == 'down' and ord(current_pos[0]) < 65 + ROWS - 1:
                current_pos = (chr(ord(current_pos[0]) + 1), current_pos[1])
            elif direction == 'left' and current_pos[1] > 1:
                current_pos = (current_pos[0], current_pos[1] - 1)
            elif direction == 'right' and current_pos[1] < COLS:
                current_pos = (current_pos[0], current_pos[1] + 1)
            
            traveled_path.add(prev_pos)
            traveled_lines.add((prev_pos, current_pos))

        draw_grid(traveled_path, traveled_lines, current_pos)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

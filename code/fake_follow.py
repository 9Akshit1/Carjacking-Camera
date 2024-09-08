import pygame
import sys, os
import random
import time

pygame.init()

WIDTH, HEIGHT = 850, 800  
ROWS, COLS = 20, 20  
NODE_RADIUS = 5  
GAP = 40  
START_POS = ('J', 10)  # Example starting position (row, column)
DIRECTIONS = ['right', 'down', 'down', 'right', 'up']  # Example path

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 150, 0)
GRASS_GREEN = (124, 252, 0)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
BLUE = (0, 0, 255)
SKY_BLUE = (135, 206, 235)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Visualized Tracker Map")

font = pygame.font.SysFont(None, 24)
title_font = pygame.font.SysFont(None, 48)

image_paths = ["house.png", "building1.png", "building2.png", "park.png", "shop.png"]
images = [pygame.image.load(os.path.join('data', img)) for img in image_paths]

image_sizes = [GAP - 10, GAP - 15, GAP - 15, GAP - 10, GAP - 10]
images = [pygame.transform.scale(img, (image_sizes[i], image_sizes[i])) for i, img in enumerate(images)]

car_image = pygame.image.load(os.path.join('data', 'car.png'))
car_image = pygame.transform.scale(car_image, (GAP - 10, GAP - 10))

road_image = pygame.image.load(os.path.join('data', 'road.png'))
road_image = pygame.transform.scale(road_image, (GAP, NODE_RADIUS * 2))

def get_grid_position(pos):
    row = ord(pos[0]) - 65  # Convert letter to index (A=0, B=1, ...)
    col = pos[1] - 1  # Convert number to index (1=0, 2=1, ...)
    x = col * GAP + 50
    y = row * GAP + 50
    return x, y

def draw_grid(traveled_path, traveled_lines, current_pos, highlight_next=False, next_positions=[]):
    screen.fill(SKY_BLUE)
    
    for i in range(ROWS):
        label = font.render(chr(65 + i), True, WHITE)
        screen.blit(label, (10, i * GAP + 50))
    
    for j in range(COLS):
        label = font.render(str(j + 1), True, WHITE)
        screen.blit(label, (j * GAP + 50, 10))
    
    for i in range(ROWS):
        for j in range(COLS):
            x = j * GAP + 50
            y = i * GAP + 50
            pos = (65 + i, j + 1)

            if j != COLS-1:
                pygame.draw.rect(screen, GRASS_GREEN, (x, y, GAP, GAP))
                if images_on:
                    screen.blit(images[j % len(images)], (x + image_sizes[j % len(images)]//4, y + image_sizes[j % len(images)]//4))
            
            if (chr(pos[0]), pos[1]) in traveled_path:
                color = GREEN
            elif (chr(pos[0]), pos[1]) == current_pos:
                color = RED
            else:
                color = BLACK
            
            pygame.draw.circle(screen, color, (x, y), NODE_RADIUS)   #the node
            
            if j < COLS - 1:  # Right connection possible
                right_pos = (chr(pos[0]), pos[1] + 1)
                if ((chr(pos[0]), pos[1]), right_pos) in traveled_lines:
                    pygame.draw.line(screen, GREEN, (x + NODE_RADIUS, y), (x + GAP - NODE_RADIUS, y), 2)
                elif highlight_next and right_pos in next_positions:
                    pass
                    #pygame.draw.line(screen, YELLOW, (x + NODE_RADIUS, y), (x + GAP - NODE_RADIUS, y), 2)
                else:
                    pygame.draw.line(screen, BLACK, (x + NODE_RADIUS, y), (x + GAP - NODE_RADIUS, y), 1)
            
            if i < ROWS - 1:  # Down connection
                down_pos = (chr(pos[0] + 1), pos[1])
                if ((chr(pos[0]), pos[1]), down_pos) in traveled_lines:
                    pygame.draw.line(screen, GREEN, (x, y + NODE_RADIUS), (x, y + GAP - NODE_RADIUS), 2)
                elif highlight_next and down_pos in next_positions:
                    pass
                    #pygame.draw.line(screen, YELLOW, (x, y + NODE_RADIUS), (x, y + GAP - NODE_RADIUS), 2)
                else:
                    pygame.draw.line(screen, BLACK, (x, y + NODE_RADIUS), (x, y + GAP - NODE_RADIUS), 1)

            if j > 0:  # Left connection
                left_pos = (chr(pos[0]), pos[1] - 1)
                if ((chr(pos[0]), pos[1]), left_pos) in traveled_lines:
                    pygame.draw.line(screen, GREEN, (x - GAP + NODE_RADIUS, y), (x - NODE_RADIUS, y), 2)
                elif highlight_next and left_pos in next_positions:
                    pass
                    #pygame.draw.line(screen, YELLOW, (x - GAP + NODE_RADIUS, y), (x - NODE_RADIUS, y), 2)
                else:
                    pygame.draw.line(screen, BLACK, (x - GAP + NODE_RADIUS, y), (x - NODE_RADIUS, y), 1)
            
            if i > 0:  # Up connection
                up_pos = (chr(pos[0] - 1), pos[1])
                if ((chr(pos[0]), pos[1]), up_pos) in traveled_lines:
                    pygame.draw.line(screen, GREEN, (x, y - GAP + NODE_RADIUS), (x, y - NODE_RADIUS), 2)
                elif highlight_next and up_pos in next_positions:
                    pass
                    #pygame.draw.line(screen, YELLOW, (x, y - GAP + NODE_RADIUS), (x, y - NODE_RADIUS), 2)
                else:
                    pygame.draw.line(screen, BLACK, (x, y - GAP + NODE_RADIUS), (x, y - NODE_RADIUS), 1)
    
    for n in next_positions:
        x, y = get_grid_position(current_pos) 
        if n[1] != current_pos[1]:  
            if n[1] > current_pos[1]:  #right happened   n[1] = col + 2
                pygame.draw.line(screen, YELLOW, (x + NODE_RADIUS, y), (x + GAP - NODE_RADIUS, y), 2)
            else:   #left happened  n[1] = col
                x -= GAP
                pygame.draw.line(screen, YELLOW, (x + NODE_RADIUS, y), (x + GAP - NODE_RADIUS, y), 2)
        else:       # n[1] = col + 1
            if ord(n[0]) > ord(current_pos[0]):    #down happened  ord(n[0]) = 65 + row + 1
                pygame.draw.line(screen, YELLOW, (x, y + NODE_RADIUS), (x, y + GAP - NODE_RADIUS), 2)
            else:          #up happened  ord(n[0]) = 65 + row - 1
                pygame.draw.line(screen, YELLOW, (x, y - GAP + NODE_RADIUS), (x, y - NODE_RADIUS), 2)

def highlight_next_paths(traveled_path, current_pos, traveled_lines, start_pos, current_directions):
    row, col = ord(current_pos[0]) - 65, current_pos[1] - 1
    next_positions = []
    next_directions = []
    
    if row > 0:  # Up
        next_positions.append((chr(65 + row - 1), col + 1))
        next_directions.append('up')
    if row < ROWS - 1:  # Down
        next_positions.append((chr(65 + row + 1), col + 1))
        next_directions.append('down')
    if col > 0:  # Left
        next_positions.append((chr(65 + row), col))
        next_directions.append('left')
    if col < COLS - 1:  # Right
        next_positions.append((chr(65 + row), col + 2))
        next_directions.append('right')
    
    draw_grid(traveled_path, traveled_lines, current_pos, highlight_next=True, next_positions=next_positions)
    pygame.display.flip()

    print("Next possible positions:", next_positions)

    activate_camera(next_positions, next_directions, start_pos, current_directions)


def activate_camera(positions, next_directions, start_pos, current_directions):
    # Pseudocode:
    # 1. Capture the node position (e.g., A3, B5).
    # 2. Activate the camera at this position, so it can run its code and then return if it has foudn the target
    # 3. Optionally, log or print that the camera was activated.
    for pos in positions:
        print(f"Camera activated at node {pos}")

    # Since all roads are equidistant on our map/graph, and assuming the target travels at constant same speed for all roads
    # the target will arrive at each next_positions node at the same time as the other same instance next_position nodes
    # so, given this, all the newly activated cameras would give a detection signal at the same time if they detected the target car (or a similar car)
    # obviously, this would be solved, if the license plate detector always worked with 100% accuracy, but that it isn't realistic
    # anyways, if we receive a signal at the same time, then we can determine which cameras have detected and which ones have not
    # so, we can leave the detected cameras alone and let them run independently with all of this shared data (find_this folder)
    # and luckily, we can immediately deactivate the non-detected cameras which is good obviously
    print("Waiting for cameras' signals...")
    pygame.time.wait(1000)   #wait for target to reach the next nodes
    signals_from_cameras = []
    for i in range(len(positions)):
        signals_from_cameras.append(random.randint(0, 1))  #example values for signals. 0 means not-detected, 1 means detected
    for i in range(0, len(signals_from_cameras)):
        x, y = get_grid_position(positions[i])
        if signals_from_cameras[i] == 1:
            print(f"Camera at node {positions[i]} detected target-like vehicle. Camera will continue searching independently")
            pygame.draw.circle(screen, (0, 0, 255), (x, y), NODE_RADIUS)    #color the activated camera nodes blue
            #make blink button  ----------------------------------------------------------
        else:
            print(f"Camera deactivated at node {positions[i]} because it did not detect target's vehicle")
            pygame.draw.circle(screen, (128, 128, 128), (x, y), NODE_RADIUS)    #color the deactivated camera nodes grey

def move_surface_smoothly(start_pos, directions):
    traveled_path = set()
    traveled_lines = set()
    current_pos = start_pos
    square_size = NODE_RADIUS * 2  

    clock = pygame.time.Clock()

    for direction in directions:
        x, y = get_grid_position(current_pos)
        traveled_path.add(current_pos)
        prev_pos = current_pos
        
        if direction == 'up' and ord(current_pos[0]) > 65:
            next_pos = (chr(ord(current_pos[0]) - 1), current_pos[1])
        elif direction == 'down' and ord(current_pos[0]) < 65 + ROWS - 1:
            next_pos = (chr(ord(current_pos[0]) + 1), current_pos[1])
        elif direction == 'left' and current_pos[1] > 1:
            next_pos = (current_pos[0], current_pos[1] - 1)
        elif direction == 'right' and current_pos[1] < COLS:
            next_pos = (current_pos[0], current_pos[1] + 1)
        else:
            next_pos = current_pos
        
        next_x, next_y = get_grid_position(next_pos)
        traveled_lines.add((prev_pos, next_pos))
        traveled_lines.add((next_pos, prev_pos))

        for i in range(10):
            intermediate_x = x + (next_x - x) * i / 10
            intermediate_y = y + (next_y - y) * i / 10
            
            draw_grid(traveled_path, traveled_lines, current_pos, highlight_next=False, next_positions=[])
            pygame.draw.rect(screen, RED, (intermediate_x - square_size // 2, intermediate_y - square_size // 2, square_size, square_size))
            if images_on:
                screen.blit(car_image, (intermediate_x - square_size // 2 - 10, intermediate_y - square_size // 2 - 8))
            pygame.display.flip()
            clock.tick(20)  # speed of movement
        
        current_pos = next_pos

    highlight_next_paths(traveled_path, current_pos, traveled_lines, start_pos, directions)

def main_follow(start_pos, directions):
    running = True
    paused = False
    global images_on
    user_input = input("Images on? (True/False): ").lower()
    if user_input == 'true':
        images_on = True
    elif user_input == 'false':
        images_on = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:  # Press SPACE to unpause if needed
                    paused = False

        if not paused:        #For testing purposes only. If not for testing purposes, then delete and un-indent the next 11 lines
            move_surface_smoothly(start_pos, directions)
            #All info will be sent to user and maybe police

            paused = True

            pygame.display.flip()

            # The next 3 lines are commented only for testing purposes. If not for testing, remove the comment hastag
            #print("Deactivating this camera soon...")
            #print("The other activated cameras will send their info/videos to user and maybe police...")
            #running = False

    # Quit Pygame
    pygame.quit()
    sys.exit()

#main_follow(START_POS, DIRECTIONS)    #turn off when not testing
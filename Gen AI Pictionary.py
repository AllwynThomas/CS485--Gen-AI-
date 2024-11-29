import sys
import pygame
import pygame_gui
import pygame_gui.elements.ui_button
import ctypes
import time
import pandas as pd

# Sharper Window
ctypes.windll.shcore.SetProcessDpiAwareness(True)

# Pygame Config
pygame.init()
fps = 0
fpsClock = pygame.time.Clock()
width, height = 640, 480
screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
font = pygame.font.SysFont('Helvetica', 30)
ui_manager = pygame_gui.UIManager((1080, 1080), "themes/button_themes.json")

# Data Preprocessing
categories = pd.read_csv("data/categories.txt", sep="/n", header=None, names=["Categories"], engine="python") 

# Picking Categories
random_categories = categories.sample(n=12)['Categories'].tolist()
player_categories = {}
ai_categories = {}
for round in range(1, 4):
    ai_categories[round] = random_categories[round-1]
    player_categories[round] = (random_categories[3*round], random_categories[3*round+1], random_categories[3*round+2])

# Variables
buttons = {}
colorMappings = {"#1,1": "#FFFFFF", "#1,2": "#C1C1C1", "#1,3": "#EF130B", "#1,4": "#FF7100", "#1,5": "#FFE400", "#1,6": "#00CC00", "#1,7": "#00FF91", "#1,8": "#00B2FF", "#1,9": "#231FD3", "#1,10": "#A300BA", "#1,11": "#DF69A7", "#1,12": "#FFAC8E", "#1,13": "#A0522D",
                 "#2,1": "#000000", "#2,2": "#505050", "#2,3": "#740B07", "#2,4": "#C23800", "#2,5": "#E8A200", "#2,6": "#004619", "#2,7": "#00785D", "#2,8": "#00569E", "#2,9": "#0E0865", "#2,10": "#550069", "#2,11": "#873554", "#2,12": "#CC774D", "#2,13": "#63300D"}
drawColor = [0,0,0]
brushSize = 10
canvasSize = [800, 800]
playerTurn = True               # 1 = Player; 0 = AI Opponent
round = 1
gameOver = False

# Round Timer
roundTime = 60
counter, timerText = roundTime, str(roundTime).rjust(3)
TIMEREVENT = pygame.USEREVENT+3
pygame.time.set_timer(TIMEREVENT, 1000)
timerFont = pygame.font.SysFont('Courier New', 30)

# Canvas Setup
canvas = pygame.Surface(canvasSize)
canvas.fill((255, 255, 255))

# Text Input
textInputRect = pygame.Rect(1210, 798, 270, 40)
enteredTextRect = pygame.Rect(1210, 798-40, 270, 40)
userText = "Type your guess here..."
enteredText = ""
inputActive = pygame.Color('white')
inputPassive = pygame.Color('#B8FF9B')
inputColor = inputPassive
inputTextColor = pygame.Color('darkgray')
isTextInputActive = False
enteredTextColor = pygame.Color('lightskyblue')

# Color Swatches (Buttons)                                                       
button_row_width = 50
button_row_height = 50
start_x = 200
start_y = 110
spacing = 5
for i in range(1, 3):
    for j in range(1, 14):
        position =  (start_x + i * spacing + ((i - 1) * button_row_width),
                    (start_y + j * spacing + ((j - 1) * button_row_height)))
        button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect( position,
                                                                (button_row_width,
                                                                button_row_height)),
                                                                text='',
                                                                manager=ui_manager,
                                                                object_id='#' + str(i) + ',' + str(j))
        buttons[str(button.object_ids)] = button

# Handler Functions
def changeColor(color):
    global drawColor
    drawColor = color
        
def save(name, playerType):
    pygame.image.save(canvas, f"images/{playerType}/{time.strftime('%Y-%m-%d_%H-%M-%S_', time.localtime())}{name}.png")

# Main Game Loop
while True:
    screen.fill((200,200,200))
    time_delta = fpsClock.tick(fps) / 1000
    for event in pygame.event.get():
        if (event.type == pygame.QUIT):
            pygame.quit()
            sys.exit()
        if (playerTurn and event.type == pygame_gui.UI_BUTTON_PRESSED):
            buttonID = str(event.ui_element.object_ids)[2:-2]
            if (buttonID in colorMappings):
                changeColor(colorMappings[buttonID])
        if (event.type == pygame.MOUSEBUTTONDOWN):
            if (textInputRect.collidepoint(event.pos)):
                isTextInputActive = True
            else:
                isTextInputActive = False
        if (event.type == pygame.KEYDOWN and isTextInputActive):
            if (event.key == pygame.K_BACKSPACE):
                userText = userText[:-1]
            elif (event.key == pygame.K_RETURN):
                enteredText = userText
                userText = ""
            else:
                userText += event.unicode
        if (event.type == TIMEREVENT):
            counter -= 1
            timerText = str(counter).rjust(3)
            # Decrementing timer
            if (counter >= 0):
                timerText = str(counter).rjust(3)
            # Switching between player/ai turns when timer hits 0 and incrementing rounds
            elif ((counter < 0) and not(round == 3 and not playerTurn)):
                playerTurn = not playerTurn
                if (playerTurn == True):
                    timerText = "Player's Turn!"
                    save(f"round_{round}_ai_drawing", "ai")
                    round += 1
                else:
                    timerText = "AI's Turn!"
                    save(f"round_{round}_player_drawing", "player")
                drawColor = "#000000"
                screen.blit(timerFont.render(timerText, True, (0, 0, 0)), (32, 48))
                canvas.fill("#FFFFFF")
                pygame.display.flip()
                time.sleep(3)
                counter = roundTime + 3
            # Game finished after round 3
            # TODO: show points per round, total points for each team, and the winner
            else:
                gameOver = True
                timerText = "Game Over!"
                screen.blit(timerFont.render(timerText, True, (0, 0, 0)), (32, 48))
                pygame.display.flip()
        ui_manager.process_events(event)
    ui_manager.update(time_delta)
    
    if (not gameOver):
    
        x, y= screen.get_size()
        screen.blit(canvas, [x/2 - canvasSize[0]/2, y/2 - canvasSize[1]/2])
            
        # Drawing
        if (playerTurn and pygame.mouse.get_pressed()[0]):
            mx, my = pygame.mouse.get_pos()
            dx = mx - x/2 + canvasSize[0]/2
            dy = my - y/2 + canvasSize[1]/2
            pygame.draw.circle(
                canvas,
                drawColor,
                [dx, dy],
                brushSize,
            )
         
        # Text Input
        if (isTextInputActive):
            inputColor = inputActive
            inputTextColor = pygame.Color('black')
            if (not userText or userText == "Type your guess here..."):
                userText = ""
        else:
            inputColor = inputPassive
            if (not userText):
                userText = "Type your guess here..."
                inputTextColor = pygame.Color('darkgray')
        pygame.draw.rect(screen, inputColor, textInputRect)
        textSurface = font.render(userText, True, inputTextColor)
        screen.blit(textSurface, (textInputRect.x+5, textInputRect.y))
        textInputRect.w = max(270, textSurface.get_width()+10)
        
        # Entered Text
        pygame.draw.rect(screen, enteredTextColor, enteredTextRect)
        enteredTextSurface = font.render(enteredText, True, (0, 0, 0))
        screen.blit(enteredTextSurface, (enteredTextRect.x+5, enteredTextRect.y))
        enteredTextRect.w = max(270, enteredTextSurface.get_width()+10) 
            
        # Color Indicator
        pygame.draw.rect(
            screen,
            drawColor,
            pygame.Rect(235, 40, 50, 50),
            border_radius=5
        )
        pygame.draw.rect(
            screen,
            "#000000",
            pygame.Rect(235, 40, 50, 50),
            width=2,
            border_radius=5
        )
        ui_manager.draw_ui(screen)
        
    # Timer
    screen.blit(timerFont.render(timerText, True, (0, 0, 0)), (32, 48))
        
    pygame.display.flip()
    fpsClock.tick(fps)
import sys
import os
#import pygame
#import pygame_gui
#import pygame_gui.elements.ui_button
import ctypes
import time, datetime
import random
import quickdraw
#import pandas as pd
import numpy as np
import tensorflow as tf
#import imageio
from matplotlib import pyplot as plt
#tf.config.set_visible_devices([], 'GPU')                    # Does not use GPU for processing
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from PIL import Image, ImageOps
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Checking if TFlite model exists
CNNmodelExists = False
GANmodelExists = False
if Path("models/CNN/tflite_model_40.tflite").exists():
    CNNmodelExists = True
if Path("models/cGAN/RMSProp_1000_Epochs_model.tflite").exists():
    GANmodelExists = True

# Setting Parameters
batch_size = 64
num_channels = 1
num_classes = 345
image_size = 40
latent_dim = 100
generator_in_channels = latent_dim + num_classes
discriminator_in_channels = num_channels + num_classes
#print(generator_in_channels, discriminator_in_channels)

# Getting dataset for training cGAN
if (not GANmodelExists):
    # Loading images
    dataset_dir = Path("data/quickdraw_images")
    GAN_data = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        seed=42,
        color_mode="grayscale",
        image_size=(image_size, image_size),
        batch_size=64
    )
    
    class_names = GAN_data.class_names
    n_classes = len(class_names)
    AUTOTUNE = tf.data.AUTOTUNE
    
    # Preprocessing with Normalizing Images and 1-Hot Encoding Labels
    def GANpreprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.one_hot(label, depth=n_classes)
        return image, label
    GAN_data = GAN_data.map(GANpreprocess, num_parallel_calls=AUTOTUNE)
    #for img_batch, label_batch in GAN_data.take(1):
        #print(f"Image batch shape: {img_batch.shape}")
        #print(f"Label batch shape: {label_batch.shape}")
    GAN_data = GAN_data.shuffle(buffer_size=4096).prefetch(buffer_size=AUTOTUNE)

# Creating cGAN Model
if (not GANmodelExists):
    # Creating Discriminator
    discriminator = keras.Sequential(
        [
            keras.layers.InputLayer((image_size, image_size, discriminator_in_channels)),
            keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
            keras.layers.LeakyReLU(negative_slope=0.2),
            keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
            keras.layers.LeakyReLU(negative_slope=0.2),
            keras.layers.GlobalMaxPooling2D(),
            keras.layers.Dense(1),
        ],
        name="discriminator",
    )

    # Creating Generator
    generator = keras.Sequential(
        [
            keras.layers.InputLayer((generator_in_channels,)),
            # We want to generate 128 + num_classes coefficients to reshape into a
            # 10x10x(128 + num_classes) map.
            keras.layers.Dense(10 * 10 * generator_in_channels),
            keras.layers.LeakyReLU(negative_slope=0.2),
            keras.layers.Reshape((10, 10, generator_in_channels)),
            keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
            keras.layers.LeakyReLU(negative_slope=0.2),
            keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
            keras.layers.LeakyReLU(negative_slope=0.2),
            keras.layers.Conv2D(1, (10, 10), padding="same", activation="sigmoid"),
        ],
        name="generator",
    )
    
    # Creating cGAN Model
    class ConditionalGAN(keras.Model):
        def __init__(self, discriminator, generator, latent_dim):
            super().__init__()
            self.discriminator = discriminator
            self.generator = generator
            self.latent_dim = latent_dim
            self.seed_generator = keras.random.SeedGenerator(42)
            self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
            self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

        @property
        def metrics(self):
            return [self.gen_loss_tracker, self.disc_loss_tracker]

        def compile(self, d_optimizer, g_optimizer, loss_fn):
            super().compile()
            self.d_optimizer = d_optimizer
            self.g_optimizer = g_optimizer
            self.loss_fn = loss_fn

        def train_step(self, data):
            # Unpack the data.
            real_images, one_hot_labels = data

            # Add dummy dimensions to the labels so that they can be concatenated with
            # the images. This is for the discriminator.
            image_one_hot_labels = one_hot_labels[:, :, None, None]
            image_one_hot_labels = keras.ops.repeat(
                image_one_hot_labels, repeats=[image_size * image_size]
            )
            image_one_hot_labels = keras.ops.reshape(
                image_one_hot_labels, (-1, image_size, image_size, num_classes)
            )

            # Sample random points in the latent space and concatenate the labels.
            # This is for the generator.
            batch_size = keras.ops.shape(real_images)[0]
            random_latent_vectors = keras.random.normal(
                shape=(batch_size, self.latent_dim), seed=self.seed_generator
            )
            random_vector_labels = keras.ops.concatenate(
                [random_latent_vectors, one_hot_labels], axis=1
            )

            # Decode the noise (guided by labels) to fake images.
            generated_images = self.generator(random_vector_labels)

            # Combine them with real images. Note that we are concatenating the labels
            # with these images here.
            fake_image_and_labels = keras.ops.concatenate(
                [generated_images, image_one_hot_labels], -1
            )
            real_image_and_labels = keras.ops.concatenate([real_images, image_one_hot_labels], -1)
            combined_images = keras.ops.concatenate(
                [fake_image_and_labels, real_image_and_labels], axis=0
            )

            # Assemble labels discriminating real from fake images.
            labels = keras.ops.concatenate(
                [keras.ops.ones((batch_size, 1))*0.9, keras.ops.zeros((batch_size, 1))], axis=0
            )

            # Train the discriminator.
            with tf.GradientTape() as tape:
                predictions = self.discriminator(combined_images)
                d_loss = self.loss_fn(labels, predictions)
            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_weights)
            )

            # Sample random points in the latent space.
            random_latent_vectors = keras.random.normal(
                shape=(batch_size, self.latent_dim), seed=self.seed_generator
            )
            random_vector_labels = keras.ops.concatenate(
                [random_latent_vectors, one_hot_labels], axis=1
            )

            # Assemble labels that say "all real images".
            misleading_labels = keras.ops.zeros((batch_size, 1))

            # Train the generator (note that we should *not* update the weights
            # of the discriminator)!
            with tf.GradientTape() as tape:
                fake_images = self.generator(random_vector_labels)
                fake_image_and_labels = keras.ops.concatenate(
                    [fake_images, image_one_hot_labels], -1
                )
                predictions = self.discriminator(fake_image_and_labels)
                g_loss = self.loss_fn(misleading_labels, predictions)
            grads = tape.gradient(g_loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

            # Monitor loss.
            self.gen_loss_tracker.update_state(g_loss)
            self.disc_loss_tracker.update_state(d_loss)
            return {
                "g_loss": self.gen_loss_tracker.result(),
                "d_loss": self.disc_loss_tracker.result(),
            }
    
    # Training cGAN model with callbacks for logging
    cond_gan = ConditionalGAN(
    discriminator=discriminator, generator=generator, latent_dim=latent_dim
    )
    cond_gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0003, clipvalue=1.0, beta_1=0.5),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0003, clipvalue=1.0, beta_1=0.5),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    )
    def generate_initial_noise(batch_size, latent_dim, num_classes):
        noise = tf.random.normal(shape=(batch_size, latent_dim))
        labels = tf.one_hot(np.random.randint(0, num_classes, size=batch_size), num_classes)
        noise_with_labels = tf.concat([noise, labels], axis=1)
        return noise_with_labels
    noise_with_labels = generate_initial_noise(batch_size, latent_dim, num_classes)
    def log_images(epoch, _):
        if (epoch + 1) % 1 == 0:
            gen_imgs = generator.predict(noise_with_labels[:10])

            # Create directory for images if it doesn't exist
            img_log_dir = "figures/Keras_Adam_20_Epochs"
            if not os.path.exists(img_log_dir):
                os.makedirs(img_log_dir)

            # Create a single plot for all images
            fig, axs = plt.subplots(2, 5, figsize=(10, 5))
            fig.suptitle(f'Epoch {epoch + 1}', fontsize=16)

            for i, ax in enumerate(axs.flat):
                ax.imshow(gen_imgs[i, :, :, 0], cmap='gray')
                ax.set_title(f"Label: {class_names[np.argmax(noise_with_labels[i, latent_dim:latent_dim+num_classes])]}")
                ax.axis('off')

            # Save the plot
            img_path = os.path.join(img_log_dir, f"epoch_{epoch + 1}_images.png")
            plt.savefig(img_path)
            plt.close()
    
    log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    cond_gan.fit(GAN_data, epochs=20, callbacks=[tensorboard_callback, tf.keras.callbacks.LambdaCallback(on_epoch_end=log_images)])
    # Saving the model
    try:
        cond_gan.save("./models/cGAN/Keras_Adam_20_Epoch_model.keras")
    except Exception:
        pass
    cond_gan.export('./models/cGAN/Keras_Adam_20_Epoch_model_lite')
    
quit()
# Getting QuickDraw Images from API
def generate_class_images(name, max_drawings, recognized):
    directory = Path("data/quickdraw_images/" + name)
    if directory.exists():
        return
    
    directory.mkdir(parents=True)
    try:
        images = quickdraw.QuickDrawDataGroup(name, max_drawings=max_drawings, recognized=recognized, cache_dir="data/.quickdrawcache")
        for img in images.drawings:
            filename = directory.as_posix() + "/" + str(img.key_id) + ".png"
            img.get_image(stroke_width=3).resize((image_size, image_size)).save(filename)
    except:
        return

def download_images_parallel(labels, max_drawings, recognized):
    # Use the number of CPUs or threads available
    max_threads = os.cpu_count() or 4  # Fallback to 4 if os.cpu_count() returns None
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [
            executor.submit(generate_class_images, label, max_drawings, recognized)
            for label in labels
        ]
        for future in futures:
            future.result()

if __name__ == "__main__":
    categories = quickdraw.QuickDrawData().drawing_names
    #start = time.time()
    if (not CNNmodelExists or not GANmodelExists):
        download_images_parallel(categories, max_drawings=2000, recognized=True)
    #end = time.time()
    #print(f"Image Download Time: {(end-start)//60}m {(end-start)%60}s")

# Picking Categories
random_categories = random.sample(categories, 6)
player_categories = {}
ai_categories = {}
ai_guesses = {}
for r in range(1, 4):
    ai_categories[r] = random_categories[r-1]
    player_categories[r] = random_categories[r+2]
#print(player_categories, ai_categories)

if (not CNNmodelExists or not Path("data/categories.txt").exists()):
    # Splitting Dataset into Train/Validate Sets    
    dataset_dir = Path("data/quickdraw_images")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        color_mode="grayscale",
        image_size=(image_size, image_size),
        batch_size=32
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        color_mode="grayscale",
        image_size=(image_size, image_size),
        batch_size=32
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(2000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    categories = train_ds.class_names
    if (not Path("data/categories.txt").exists()):
        with open("data/categories.txt", "w") as f:
            f.write("\n".join(categories))

# Load correctly ordered categories for predictions
with open("data/categories.txt", "r") as f:
    categories = f.read().splitlines()

if (not CNNmodelExists):
    # Building CNN Model
    n_classes = 345
    input_shape = (40, 40, 1)

    model = Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Rescaling(1./255),
        layers.BatchNormalization(),

        layers.Conv2D(6, kernel_size=3, padding="same", activation="relu"),
        layers.Conv2D(8, kernel_size=3, padding="same", activation="relu"),
        layers.Conv2D(10, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2),
        
        layers.Flatten(),
        
        layers.Dense(700, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(500, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(400, activation="relu"),
        layers.Dropout(0.2),

        layers.Dense(n_classes, activation="softmax")
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

    model.summary()

    # Training and Saving CNN Base Model
    epochs = 10

    #logdir = os.path.join("models/CNN/logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    #tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1,
        #callbacks=[tensorboard_callback]
    )
    model.save("./models/CNN/bigger_model_40")

    # Converting the Model
    model.export('./models/CNN/bigger_model_40_lite')
    converter = tf.lite.TFLiteConverter.from_saved_model("models/CNN/bigger_model_40_lite")
    tflite_model = converter.convert()

    # Saving the Tflite Model.
    with open('models/CNN/tflite_model_40.tflite', 'wb') as f:
        f.write(tflite_model)

# Function for preprocessing player drawings to make them similar to training data
def imagePreprocessing(image):
    # Converting to grayscale
    img = image.convert('L')
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Find bounding box of non-white pixels
    non_white_pixels = np.argwhere(img_array < 255)
    try:
        y_min, x_min = non_white_pixels.min(axis=0)
        y_max, x_max = non_white_pixels.max(axis=0)
    except:
        # Return white image if no drawing
        return Image.new('L', (image_size, image_size), color=255)    
    
    # Crop to bounding box
    cropped_img = img_array[y_min:y_max + 1, x_min:x_max + 1]
    
    # Resize to fit within output_size while maintaining aspect ratio
    cropped_img_pil = Image.fromarray(cropped_img)
    cropped_img_resized = ImageOps.fit(cropped_img_pil, (image_size, image_size), method=Image.Resampling.LANCZOS)
    
    # Pad to ensure it's centered on a 40x40 canvas
    padded_img = Image.new('L', (image_size, image_size), color=255)  # Create white canvas
    padded_img.paste(cropped_img_resized, (0, 0))  # Paste resized image
    
    # Final resizing to ensure size compatibility
    final_img = padded_img.resize((image_size, image_size))
    return final_img

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="models/CNN/tflite_model_40.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function for CNN to guess player's drawing
def CNNguess(img_name):
    global ai_guesses
    # Preprocessing player's drawing
    player_img = Image.open(f"images/temp/player_temp/{img_name}.png")
    processed_img = imagePreprocessing(player_img)
    #processed_img.save(f"images/temp/test/{img_name}.png")

    input_data = tf.convert_to_tensor(np.array(processed_img, dtype=np.float32))
    input_data = np.expand_dims(input_data, axis=(0, -1))
    '''
    kerasModel = tf.keras.models.load_model("models/CNN/bigger_model_40")
    #kerasModel.summary()
    '''
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Finding top 3 possible categories based on 5 highest probabilities
    output_data = interpreter.get_tensor(output_details[0]['index'])[0] #kerasModel.predict(input_data)[0]
    top5_categories = np.argpartition(output_data, -5)[-5:]
    top5_categories_sorted = top5_categories[np.argsort(output_data[top5_categories])[::-1]]
    for i in top5_categories_sorted:
        ai_guesses[categories[i]] = round(output_data[i]*100, 2)
    print(f"\nPredicted Categories: {ai_guesses}\n")
    
    # Deleting temp drawing
    current_dir = os.getcwd()
    file_name = f"images/temp/player_temp/{img_name}.png"
    file_path = os.path.join(current_dir, file_name)
    try:
        os.remove(file_path)
    except Exception:
        return

# Pygame Config
ctypes.windll.shcore.SetProcessDpiAwareness(True)                   # Sharper Window
pygame.init()
fps = 0
fpsClock = pygame.time.Clock()
width, height = 640, 480
screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
font = pygame.font.SysFont('Helvetica', 30)
ui_manager = pygame_gui.UIManager((1080, 1080), "themes/button_themes.json")

# Variables
buttons = {}
colorMappings = {"#1,1": "#FFFFFF", "#1,2": "#C1C1C1", "#1,3": "#EF130B", "#1,4": "#FF7100", "#1,5": "#FFE400", "#1,6": "#00CC00", "#1,7": "#00FF91", "#1,8": "#00B2FF", "#1,9": "#231FD3", "#1,10": "#A300BA", "#1,11": "#DF69A7", "#1,12": "#FFAC8E", "#1,13": "#A0522D",
                 "#2,1": "#000000", "#2,2": "#505050", "#2,3": "#740B07", "#2,4": "#C23800", "#2,5": "#E8A200", "#2,6": "#004619", "#2,7": "#00785D", "#2,8": "#00569E", "#2,9": "#0E0865", "#2,10": "#550069", "#2,11": "#873554", "#2,12": "#CC774D", "#2,13": "#63300D"}
drawColor = [0,0,0]
brushSize = 3
canvasSize = [800, 800]
playerTurn = True               # 1 = Player; 0 = AI Opponent
roundNum = 1
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
    pygame.image.save(canvas, f"images/{playerType}/{name}.png")

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
            if (roundNum == 1 and playerTurn and counter == 60):
                targetWord = player_categories[roundNum]
            counter -= 1
            timerText = str(counter).rjust(3)
            # Decrementing timer
            if (counter >= 0):
                timerText = str(counter).rjust(3)
                # CNN guessing player's image every 10s
                if (playerTurn and 60-counter > 5 and(counter % 60 - 5) % 10 == 0):
                    imgName = f"{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}_round_{roundNum}_{60-counter}s_player_drawing"
                    save(imgName, "temp/player_temp")
                    CNNguess(imgName)
                    ai_guesses = {}
            # Switching between player/ai turns when timer hits 0 and incrementing rounds
            elif ((counter < 0) and not(roundNum == 3 and not playerTurn)):
                playerTurn = not playerTurn
                if (playerTurn == True):
                    timerText = "Player's Turn!"
                    save(f"{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}_round_{roundNum}_ai_drawing", "ai")
                    roundNum += 1
                    targetWord = player_categories[roundNum]
                else:
                    timerText = "AI's Turn!"
                    save(f"{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}_round_{roundNum}_player_drawing", "player")
                    targetWord = ai_categories[roundNum]
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
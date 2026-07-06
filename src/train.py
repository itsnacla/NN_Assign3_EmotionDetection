import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode",help="train/display")
mode = ap.parse_args().mode

# plots accuracy and loss curves
def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history and save to output/plots/
    """
    plot_dir = os.path.join('output', 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xticks(np.arange(1, len(model_history.history['accuracy']) + 1, max(1, len(model_history.history['accuracy']) // 10)))
    axs[0].legend(['train', 'val'], loc='best')
    axs[0].grid(True, linestyle='--', alpha=0.6)
    
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1, max(1, len(model_history.history['loss']) // 10)))
    axs[1].legend(['train', 'val'], loc='best')
    axs[1].grid(True, linestyle='--', alpha=0.6)
    
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'plot.png'), dpi=300)
    plt.show()

    # Save individual high-quality accuracy plot
    fig_acc, ax_acc = plt.subplots(figsize=(8, 5))
    ax_acc.plot(range(1, len(model_history.history['accuracy']) + 1), model_history.history['accuracy'], label='train', color='#1f77b4', linewidth=2)
    ax_acc.plot(range(1, len(model_history.history['val_accuracy']) + 1), model_history.history['val_accuracy'], label='val', color='#ff7f0e', linewidth=2)
    ax_acc.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax_acc.set_ylabel('Accuracy', fontsize=12)
    ax_acc.set_xlabel('Epoch', fontsize=12)
    ax_acc.set_xticks(np.arange(1, len(model_history.history['accuracy']) + 1, max(1, len(model_history.history['accuracy']) // 10)))
    ax_acc.legend(loc='best')
    ax_acc.grid(True, linestyle='--', alpha=0.6)
    fig_acc.savefig(os.path.join(plot_dir, 'accuracy.png'), dpi=300)
    plt.close(fig_acc)

    # Save individual high-quality loss plot
    fig_loss, ax_loss = plt.subplots(figsize=(8, 5))
    ax_loss.plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'], label='train', color='#d62728', linewidth=2)
    ax_loss.plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'], label='val', color='#2ca02c', linewidth=2)
    ax_loss.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax_loss.set_ylabel('Loss', fontsize=12)
    ax_loss.set_xlabel('Epoch', fontsize=12)
    ax_loss.set_xticks(np.arange(1, len(model_history.history['loss']) + 1, max(1, len(model_history.history['loss']) // 10)))
    ax_loss.legend(loc='best')
    ax_loss.grid(True, linestyle='--', alpha=0.6)
    fig_loss.savefig(os.path.join(plot_dir, 'loss.png'), dpi=300)
    plt.close(fig_loss)


# Define data generators
train_dir = 'data/train'
val_dir = 'data/test'

num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = 70

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

# Create the model
model = Sequential()
model.add(Input(shape=(48, 48, 1)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# model output directory
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
models_dir = os.path.join(repo_root, 'models')
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, 'model.h5')

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# If you want to train the same model or try other models, go for this
if mode == "train":
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )
    model_info = model.fit(
            train_generator,
            steps_per_epoch=num_train // batch_size,
            epochs=num_epoch,
            validation_data=validation_generator,
            validation_steps=num_val // batch_size)
    plot_model_history(model_info)
    
    # Save model in src/models
    models_dir = os.path.join('src', 'models')
    os.makedirs(models_dir, exist_ok=True)
    model.save(os.path.join(models_dir, 'model.h5'))

    # Save final metrics in output/plots
    plot_dir = os.path.join('output', 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    metrics_path = os.path.join(plot_dir, 'final_metrics.txt')
    try:
        final_train_acc = model_info.history['accuracy'][-1]
        final_val_acc = model_info.history['val_accuracy'][-1]
        final_train_loss = model_info.history['loss'][-1]
        final_val_loss = model_info.history['val_loss'][-1]
        
        print(f"Final training accuracy: {final_train_acc:.4f}")
        print(f"Final validation accuracy: {final_val_acc:.4f}")
        
        with open(metrics_path, 'w') as f:
            f.write(f"final_train_accuracy={final_train_acc:.4f}\n")
            f.write(f"final_val_accuracy={final_val_acc:.4f}\n")
            f.write(f"final_train_loss={final_train_loss:.4f}\n")
            f.write(f"final_val_loss={final_val_loss:.4f}\n")
        print(f"Metrics successfully saved to {metrics_path}")
    except Exception as e:
        print('Failed to write final metrics:', e)

# emotions will be displayed on your face from the webcam feed
elif mode == "display":
    model.load_weights(os.path.join('src', 'models', 'model.h5'))

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # Find haar cascade to draw bounding box around face
    xml_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
    facecasc = cv2.CascadeClassifier(xml_path)

    # start the webcam feed
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model(cropped_img, training=False)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
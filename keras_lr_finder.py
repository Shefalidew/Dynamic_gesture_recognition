from keras.models import Sequential
from keras import layers 
from keras.optimizers import SGD, Adam
from keras.losses import categorical_crossentropy
import matplotlib.pyplot as plt
from keras_lr_finder.keras_lr_finder.lr_finder import LRFinder
import json

# Model configuration
batch_size = 32
img_width, img_height, img_num_channels = 96, 96, 3
loss_function = categorical_crossentropy
no_classes = 12
no_epochs = 250
start_lr = 0.0001
end_lr = 1
moving_average = 20

RGB_X_train_array = np.load('Master_Thesis_codes/landmark_rgb_X_train.npy',allow_pickle=True)
print((RGB_X_train_array.shape))
RGB_X_train_array_3d = RGB_X_train_array.reshape(RGB_X_train_array.shape[0],RGB_X_train_array.shape[1],RGB_X_train_array.shape[2]*RGB_X_train_array.shape[3])
print(RGB_X_train_array_3d.shape)
x_train = RGB_X_train_array_3d
print(x_train.shape)
y_train = np.load('Master_Thesis_codes/saved_Y_train_scaled.npy')
print(y_train.shape)
RGB_X_test_array = np.load('Master_Thesis_codes/landmark_rgb_X_test.npy',allow_pickle=True)
print((RGB_X_test_array.shape))
RGB_X_test_array_3d = RGB_X_test_array.reshape(RGB_X_test_array.shape[0],RGB_X_test_array.shape[1],RGB_X_test_array.shape[2]*RGB_X_test_array.shape[3])
print(RGB_X_test_array_3d.shape)
x_test = RGB_X_test_array_3d
print(x_test.shape)
y_test = np.load('Master_Thesis_codes/saved_Y_test_scaled.npy')
print(y_test.shape)
print('Data loading finished')

def build_model_bdlstm_1(x_train):
    model = Sequential()
    model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True,activation='relu'),
                                               input_shape=(x_train.shape[1], x_train.shape[2])))   
    #model.add(layers.Dropout(rate=0.4))
    model.add(layers.Bidirectional(layers.LSTM(128,return_sequences=True,activation='relu')))
    #model.add(layers.Dropout(rate=0.4))
    model.add(layers.Bidirectional(layers.LSTM(64,return_sequences=False, activation='relu')))
    #model.add(layers.Dropout(rate=0.4))
    model.add(layers.Dense(256, activation='relu'))
    #model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    #model.add(layers.Dropout(rate=0.4))
    model.add(layers.Dense(12, activation='softmax'))
    #model.summary()
    return model
    
tests = [
  (SGD(), 'SGD optimizer'),
  (Adam(), 'Adam optimizer'),
]
# Set containers for tests
test_learning_rates = []
test_losses = []
test_loss_changes = []
labels = []
model = build_model_bdlstm_1(x_train)

# Perform each test
for test_optimizer, label in tests:
    model.compile(loss=loss_function,
                optimizer=test_optimizer,
                metrics=['categorical_accuracy'])
    # Instantiate the Learning Rate Range Test / LR Finder
    lr_finder = LRFinder(model)
    # Perform the Learning Rate Range Test
    outputs = lr_finder.find(input_train, target_train, start_lr=start_lr, end_lr=end_lr, batch_size=batch_size, epochs=no_epochs)
    # Get values
    learning_rates  = lr_finder.lrs
    losses          = lr_finder.losses
    loss_changes = []
    for i in range(moving_average, len(learning_rates)):
        loss_changes.append((losses[i] - losses[i - moving_average]) / moving_average)
    # Append values to container
    test_learning_rates.append(learning_rates)
    test_losses.append(losses)
    test_loss_changes.append(loss_changes)
    labels.append(label)

with open('training_history_test_learning_rates.json','w') as f:
     json.dump(test_learning_rates, f)
with open('training_history_test_losses.json','w') as f:
    json.dump(test_losses, f)
with open('training_history_test_loss_changes.json','w') as f:
    json.dump(test_loss_changes, f)

# Generate plot for Loss Deltas
for i in range(0, len(test_learning_rates)):
  plt.plot(test_learning_rates[i][moving_average:], test_loss_changes[i], label=labels[i])
plt.xscale('log')
plt.legend(loc='upper left')
plt.ylabel('loss delta')
plt.xlabel('learning rate (log scale)')
plt.title('Results for Learning Rate Range Test / Loss Deltas for Learning Rate')
plt.show()

# Generate plot for Loss Values
for i in range(0, len(test_learning_rates)):
  plt.plot(test_learning_rates[i], test_losses[i], label=labels[i])
plt.xscale('log')
plt.legend(loc='upper left')
plt.ylabel('loss')
plt.xlabel('learning rate (log scale)')
plt.title('Results for Learning Rate Range Test / Loss Values for Learning Rate')
plt.show()


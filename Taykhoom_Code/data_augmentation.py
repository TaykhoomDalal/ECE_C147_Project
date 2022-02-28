import numpy as np

# def sliding_window(X, y, window_size, jump) -> Tuple[np.ndarray, np.ndarray]:
#     '''
#     Sliding Window Data Augmentation technique (inspired by https://github.com/gumpy-bci/gumpy)

#     Input:
#         - X: (Trials, Time_Steps, Channels)
#         - y: (Trials, )
#         - window_size: number of samples in sliding window
#         - jump: size of jump between windows
#         - verbose: print progress
#     Output:
#         - X_augmented: (Trials, (time_steps - window_size + jump - start)//jump ,window_size, Channels)
#         - y_augmented: (Trials, (time_steps - window_size + jump - start)//jump)
#     '''
#     trials, time_steps, channels = X.shape

#     new_time_steps = (time_steps - window_size + jump)//jump

#     X_augmented = np.zeros((trials, new_time_steps, window_size, channels))
#     y_augmented = np.zeros((trials, new_time_steps))

#     for t in range(trials):
#         X_window = np.array([X[t, i : i + window_size] for i in range(0, time_steps - window_size + jump, jump)])
#         y_window = np.repeat(y[t], new_time_steps)
        
#         X_augmented[t] = X_window
#         y_augmented[t] = y_window

#     return X_augmented, y_augmented

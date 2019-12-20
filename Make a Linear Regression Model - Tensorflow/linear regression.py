import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class linear_reg_model(tf.Module):
    def __init__(self,dims,initial_values=None):
        super().__init__()

        if not initial_values:
            self.m = tf.Variable(tf.random.uniform([dims]), name="gradient")
            self.c = tf.Variable(tf.random.uniform([dims]), name="intercept")
        else:
            self.m = tf.Variable(initial_values[0], name="gradient")
            self.c = tf.Variable(initial_values[1], name="intercept")

    def __call__(self,x):
        return (self.m*x) + self.c

def make_random_dataset(seed=0, min=0, max=100, samples=1000, scale=0.1, val_split=0.2):
    np.random.seed(seed)
    m, c = np.random.normal(size=2)

    train_steps, val_steps = int(samples * (1-val_split)), int(samples * val_split)

    x_train, x_val = np.linspace(min,max,train_steps), np.linspace(min,max,val_steps)

    func = np.vectorize(lambda x: (m*x) + c)
    y_train, y_val = func(x_train), func(x_val)

    x_train_noise, y_train_noise = np.random.normal(loc=0,scale=scale, size=(2,train_steps))
    x_val_noise, y_val_noise = np.random.normal(loc=0,scale=scale, size=(2,val_steps))

    x_train += x_train_noise
    x_val += x_val_noise
    y_train += y_train_noise
    y_val += y_val_noise

    return x_train, y_train, x_val, y_val

def make_real_dataset(path="Salary_Data.csv"):
    with open(path,"r") as f:
        raw_data = f.read()
    data = np.array([i.split(",") for i in raw_data.split("\n")][1:-1],dtype=np.float32)
    return data[:,0], data[:,1]
    
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true-y_pred))

def train(model, X_train, y_train, X_val=None, y_val=None, epochs=1, batch_size=1000, optimiser=tf.keras.optimizers.Adam(lr=0.1)):
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    assert X_train.ndim == 1 and y_train.ndim == 1 # Make sure it is only a single dimension
    assert X_train.shape[0] == y_train.shape[0] # Make sure there are the same number of samples

    num_batches_train = X_train.shape[0]//batch_size + (1 if X_train.shape[0]%batch_size != 0 else 0)

    batched_train_data = np.array_split(np.stack([X_train,y_train]),num_batches_train,axis=1) # Split training data into batches
    
    if type(X_val) != type(None) and type(y_val) != type(None):
        assert X_val.ndim == 1 and y_val.ndim == 1 # Make sure it is only a single dimension
        assert X_val.shape[0] == y_val.shape[0] # Make sure there are the same number of samples
        num_batches_val = X_val.shape[0]//batch_size + (1 if X_val.shape[0]%batch_size != 0 else 0)

        batched_val_data = np.array_split(np.stack([X_val,y_val]),num_batches_val,axis=1)
        
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        
        template = "\rEpoch {}, Train Loss: {:.3f}, Val Loss: {:.3f}" + " "*20
        
    else:
        batched_val_data = None
        
        template = "\rEpoch {}, Train Loss: {:.3f}" + " "*20
        
    for e in range(epochs):
        for X, y_true in batched_train_data:
            with tf.GradientTape() as tape:
                y_pred = model(X)
                loss = mse(y_true, y_pred)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimiser.apply_gradients(zip(gradients, model.trainable_variables))
            
            train_loss(loss)

        if batched_val_data:
            for X, y_true in batched_val_data:
                y_pred = model(X)
                loss = mse(y_true, y_pred)

                test_loss(loss)
                print(template.format(e+1, train_loss.result(), test_loss.result()),end="")
                
            test_loss.reset_states()
        else:
            print(template.format(e+1, train_loss.result()),end="")

        train_loss.reset_states()
        
        print()

def graph_model_dataset(model,X,y):
    plt.scatter(X,y,2,"b","x")
    plt.plot(X,model(X),"r")
    plt.show()

def random_train():
    model = linear_reg_model(1)
    X_train, y_train, X_val, y_val = make_random_dataset(seed=42,samples=10000,val_split=0.1,min=0,max=100,scale=2)

    train(
        model,
        X_train, y_train,
        X_val, y_val,
        epochs=20, batch_size = 10000)
    graph_model_dataset(model,X_val,y_val)
    
def real_train():
    X_train, y_train = make_real_dataset()
    estimated_gradient = (max(X_train) - min(X_train)) / (max(y_train) - min(y_train))
    estimated_intercept = min(y_train)
    model = linear_reg_model(1,[estimated_gradient, estimated_intercept])

    train(
        model,
        X_train, y_train,
        epochs=400, batch_size = 30, optimiser=tf.keras.optimizers.Adam(lr=1e+3))
    
    test_value = 6.5
    pred_salary = model(test_value)
    print("Estimated salaray for experience of {} years is £{:.2f}".format(test_value,pred_salary))
    
    graph_model_dataset(model,X_train,y_train)
    
if __name__=="__main__":
    assert tf.__version__ == "2.0.0"
    # random_train()
    real_train()
    